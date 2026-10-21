# 实现计划：vector-map-head — 对齐 UniAD MapFormer 的「真·矢量化」地图头

## 概述

本计划把 design.md 的「分步施工路线」（5 步）展开为可独立验证的编码任务。在不动已验证
image_backbone / bev_encoder 几何、也不替换现有栅格 `map_head` 的前提下，新增一个 **真矢量
地图头** `VectorMapHead`：M 个 map query attend BEV，每 query 回归一条定长折线 (P,2) + 类别
+ 存在置信度。

贯穿主线与 nano-fsd-v2 完全一致——**query 作为信息载体 + 预测/GT 匹配 + 带 mask 的回归
loss**。地图头在结构上是检测头的「孪生兄弟」，刻意对称以最大化复用并降低学习成本：

| 检测头（已有） | 矢量地图头（本计划） |
|----------------|----------------------|
| `DetectionHead` | `VectorMapHead` |
| `match_pred_to_gt` | `match_polyline_to_gt` |
| `detection_loss` | `vector_map_loss` |
| `run_detection_viz`（俯视图叠框） | `run_vectormap_viz`（俯视图叠折线） |

节奏遵循「一次加一个改动、小子集先验证再全量重建」：数据层 → 地图头 → 匹配+loss → 单头
训练验证 → 多任务整合与全量重建。每步都有命令行演示入口与可视化产物。

技术栈：Python + PyTorch，属性测试用 hypothesis，可视化用 matplotlib（Agg 后端、英文图
标题、中文 docstring）。所有张量 batch 维在前，坐标 ego 系（x 前 / y 左）、`SCENE_SCALE=50`
归一化。常量：`MAX_MAP=48`（M）、`MAP_POINTS=20`（P）、`n_cls=3`、BEV 边长 `S=100`、`D=128`、
`BEV_RANGE=50`。代码位于 `v2/`（model.py、build_meta_cache.py、dataset.py、train.py、
visualize.py、tests/）。

设计中的正确性属性（供下文引用）：
1. 形状守恒 · 2. query 顺序/置换等变 · 3. 匹配正确性 · 4. 损失零点 · 5. 顺序鲁棒性 ·
6. mask 单调性 · 7. 重采样等距性 · 8. 空场景安全 · 9. 过拟合可行性 · 10.（可选）Chamfer 一致性

## 任务

- [x] 1. 数据层：矢量折线 GT 抽取 + 重采样 + 缓存重建（小子集）
  - [x] 1.1 在 `v2/build_meta_cache.py` 实现 `resample_polyline`
    - 把任意点数折线按**弧长等距**重采样成定长 n 点：累积弧长 + `linspace` 目标 + 二分
      定段 + 线性插值；端点保持（首点=原首、末点=原末）
    - 退化处理：总弧长为 0（点全重合）返回首点重复 n 次；输入点数 <2 由调用方跳过
    - 中文 docstring 写明前置/后置条件
    - _需求引用：2.1, 2.2, 2.3, 2.5；设计组件 1_
  - [x] 1.2 为 `resample_polyline` 写属性测试与边界单测（hypothesis）
    - **属性 7 重采样等距性**：返回恰好 n 点、首/末点保持、相邻点弧长在容差内近似相等、
      对已均匀输入近似幂等
    - 边界单测：2 点直线、所有点重合、闭合环
    - 文件 `v2/tests/test_vectormap_properties.py`
    - _验证：正确性属性 7；需求 2.1, 2.2, 2.3, 2.5, 10.4_
  - [x] 1.3 在 `v2/build_meta_cache.py` 实现 `build_map_polylines`
    - 复用 `_get_map(city)` 的 `NuScenesMap` 查询 ego 周围 ±`BEV_RANGE` 内图层记录；
      复用 `_ego_g2e` 把点从 global 转当前帧 ego 系（x 前 / y 左）
    - 图层归并 3 类：0=`lane_divider`+`road_divider`、1=`road_segment`/`drivable_area` 外轮廓、
      2=`ped_crossing` 外轮廓（闭合）
    - 裁剪到 `|x|<=BEV_RANGE & |y|<=BEV_RANGE`，跨界几何裁成多段各自成折线；点数 <2 跳过
    - `resample_polyline` 到 `MAP_POINTS`；按到自车距离近者优先，超 `MAX_MAP` 截断
    - 返回定长 pad 字典 `map_pts (MAX_MAP,P,2)` / `map_cls (MAX_MAP,)` / `map_mask (MAX_MAP,)`；
      无元素时 mask 全 False；地图加载失败 try/except 置全 False 并继续
    - _需求引用：1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8；设计组件 1_
  - [x] 1.4 把矢量字段写入缓存 record
    - 在 `main()` 每条 record 追加 `map_pts/map_cls/map_mask`（保留现有 occ/drivable/traj/
      command/agent_* 字段，向后兼容）
    - _需求引用：1.1_
  - [x] 1.5 `v2/dataset.py` 读取矢量字段并归一化容错
    - `__getitem__` 在 agent_* 读取之后追加：读出 `map_pts` 转张量并 `/SCENE_SCALE` 归一化、
      `map_cls` 转 long、GT mask 键命名 `map_poly_mask`（与栅格 `drivable` 互不影响）
    - WHEN record 缺 `map_*`（旧缓存）THEN 填零 + `map_poly_mask` 全 False
    - _需求引用：3.1, 3.2, 3.3, 3.4_
  - [x] 1.6 为数据层写属性测试（hypothesis）
    - **属性 1 形状守恒**：三字段形状 `(48,20,2)/(48,)/(48,)`，无 NaN/Inf
    - **属性 8 空场景安全**：无地图元素样本 `map_poly_mask` 全 False、读取不报错
    - 追加到 `v2/tests/test_vectormap_properties.py`
    - _验证：正确性属性 1、8；需求 1.7, 3.4_
  - [x] 1.7 重建小子集缓存
    - 运行 `python build_meta_cache.py --num 2000 --out cache/meta_2000_vmap.pt`（conda `mcts_debug`）
    - _需求引用：3.5, 9.2_
  - [x] 1.8 演示入口：GT 折线叠路面底图 sanity check
    - 新增 `python build_meta_cache.py --viz-vmap N`（或独立小脚本）：从缓存取样本，复用
      drivable 栅格作底图，在 BEV 俯视图叠 GT 折线（按类着色，横轴=y 左、纵轴=x 前）
    - 保存 `outputs/step1_vmap_gt_*.png`，人眼确认分隔线/边界/人行横道位置与类别合理
    - _需求引用：1.2, 1.3, 8.2, 9.1_

- [x] 2. 地图头 VectorMapHead + 接入 NanoFSDv2
  - [x] 2.1 在 `v2/model.py` 实现 `VectorMapHead`（镜像 `DetectionHead`）
    - `nn.Parameter` map_query_content (M,D)；可选锚点位置编码（BEV 平面均匀铺 M 锚点 → MLP
      → 加到内容 query），暴露 `anchor_xy (M,2)`
    - `kv_pool`（AvgPool2d，`kv_stride` 下采样）+ `bev_pos` 位置编码构 K/V；L 层 decoder =
      cross-attn(query→BEV)+FFN+残差/LN
    - 输出头 MLP：`pts_head`→(B,M,P,2)（锚点基准 + 限幅偏移）、`cls_head`→(B,M,n_cls)、
      `score_head`→(B,M,1)；返回 `query_emb (B,M,D)`
    - _需求引用：4.1, 4.2, 4.3, 4.5, 4.6, 4.7；设计组件 3_
  - [x] 2.2 为地图头写属性测试（hypothesis）
    - **属性 1 形状守恒**：pts/cls/score/query_emb 形状恒定且无 NaN/Inf
    - **属性 2 query 顺序/置换等变**：打乱 batch 内样本顺序，输出对应打乱；同一 query 的
      pts/cls/score/query_emb 顺序对齐
    - 追加到 `v2/tests/test_vectormap_properties.py`
    - _验证：正确性属性 1、2；需求 4.1, 4.2, 4.3, 4.4_
  - [x] 2.3 接入 `NanoFSDv2` dict 输出 + `use_vector_map` 开关
    - `__init__` 新增 `n_map_query/n_map_points/n_map_cls/use_vector_map`；`use_vector_map=True`
      时构造 `self.vector_map`（与 `self.detection` 平行）
    - forward 返回 dict 新增 `vmap` 键（pts/cls/score/query_emb[,anchor_xy]），其余键不变；
      `use_vector_map=False` 时不构造、dict 不含 `vmap`
    - checkpoint 存开关、重建按开关（与 `use_planner_v2` 同款处理）
    - _需求引用：7.1, 7.2, 7.6；设计组件 4_
  - [x] 2.4 顶层整合前向自测
    - batch 跑一次 forward，确认 dict 含 `vmap` 且各形状正确、`use_vector_map=False` 时无 `vmap`
    - _需求引用：7.1, 7.2_

- [x] 3. 折线匹配 + 矢量地图损失
  - [x] 3.1 在 `v2/train.py` 实现 `polyline_distance`
    - 两折线逐点 L2 均值，对 GT 取「正序」与「点序翻转」两种对齐的较小者（线元素无方向）
    - _需求引用：6.1, 6.5；设计组件 5、决策 4_
  - [x] 3.2 在 `v2/train.py` 实现 `match_polyline_to_gt`（贪心，镜像 `match_pred_to_gt`）
    - per-sample，`@torch.no_grad`；距离矩阵基准二选一（anchor 质心距离 / `polyline_distance`）
    - GT 到自车（原点）距离近者优先，贪心取最近且未占用 query；可选类内匹配（异类置 inf）
    - 返回两个等长 1D long 索引；M_gt=0 返回空张量
    - _需求引用：5.2, 5.3, 5.4, 5.5；设计组件 5_
  - [x] 3.3 在 `v2/train.py` 实现 `vector_map_loss`（镜像 `detection_loss`）
    - 逐帧 `match_polyline_to_gt`；匹配 query 算折线回归（`polyline_distance` 之和，按匹配
      总数平均）+ 类别 CE；存在性 BCE 对全部 M query（匹配=1、未匹配=0）
    - 空场景安全：匹配总数=0 时回归/类别贡献 0、分母 clamp(min=1)、不 NaN，存在性 BCE 仍算
    - _需求引用：6.1, 6.2, 6.3, 6.4, 6.7；设计组件 5_
  - [x] 3.4 为匹配与损失写属性测试（hypothesis）
    - **属性 3 匹配正确性**：令部分预测折线精确=GT，匹配必须单射指派到对应 GT（仅来自有效 GT）
    - **属性 4 损失零点**：匹配折线=GT、类别命中、存在性对齐时折线回归分量**精确为 0**
    - **属性 5 顺序鲁棒性**：`polyline_distance(a,reverse(a))==0`；`d(a,b)==d(a,reverse(b))`
    - **属性 6 mask 单调性**：增加 pad 无效折线不改变有效折线 loss
    - **属性 8 空场景安全**：M_gt=0 时 loss 有限、回归/类别贡献 0、反传无 NaN
    - **属性 7 重采样等距性**（如需在 train 侧复用则补测）
    - 追加到 `v2/tests/test_vectormap_properties.py`
    - _验证：正确性属性 3、4、5、6、8；需求 5.4, 5.5, 6.4, 6.5, 6.6, 6.7, 10.3_
  - [ ] 3.5（可选增强）Chamfer 一致性测试
    - IF 实现 Chamfer 版距离 THEN 断言其 ≤ 顺序鲁棒 L2（更松下界）；默认不实现
    - **属性 10 Chamfer 一致性**
    - _验证：正确性属性 10；需求（设计可选项）_

- [x] 4. 检查点 — 确保测试通过
  - 运行 `v2/tests/test_vectormap_properties.py` 全部属性测试，有疑问询问用户再继续。

- [x] 5. 单头训练路径 `--vmap-only` + 可视化 + 过拟合验证
  - [x] 5.1 在 `v2/train.py` 实现 `train_vectormap_only` + `--vmap-only` 分支
    - 镜像 `train_detection_only`：只开矢量地图 loss 的独立单头训练入口
    - _需求引用：7.5, 9.1_
  - [x] 5.2 在 `v2/visualize.py` 实现 `run_vectormap_viz`（镜像 `run_detection_viz`）
    - 加载矢量地图 ckpt，对**验证集**样本（复现相同 seed/val_ratio，只画未训过样本）画
      预测折线（实线按类着色）+ GT 折线（虚线对照）
    - 预测按 `sigmoid(score) > score_thr` 过滤；`pts` 去归一化乘回 `SCENE_SCALE` 到米；
      可选复用 drivable 作底图；`_draw_polyline` 遵循横轴=y 左、纵轴=x 前
    - 保存 `outputs/step_vmap_*.png`
    - _需求引用：8.1, 8.2, 8.3, 8.4；设计组件 6_
  - [x] 5.3 小子集单头训练过拟合 sanity check
    - 用 `cache/meta_2000_vmap.pt` 跑 `--vmap-only`；后台 `nohup python -u train.py ... > log 2>&1`
      （conda `mcts_debug`，单卡 24G）
    - **属性 9 过拟合可行性**：固定小 batch 反复训练，矢量地图 loss 显著下降（BEV→map query
      →折线梯度链路通畅）；跑 `run_vectormap_viz` 确认折线逐渐贴合 GT
    - _验证：正确性属性 9；需求 9.1, 9.3, 9.4, 10.5_

- [ ] 6. 多任务整合与全量重建最终训练
  - [x] 6.1 在 `v2/train.py` 接入多任务总损失 + `--vmap-weight`
    - 新增 `--vmap-weight`（默认 0.5）；多任务循环内与 det 项并列，`w_vmap=0` 短路跳过
      （沿用现有权重为 0 机制）；累计 `acc["vmap"]`
    - _需求引用：7.3, 7.4_
  - [x] 6.2 `v2/visualize.py` 综合图/俯视图叠加矢量地图栏
    - `run_visualization` / `run_topdown_viz` 增加可选矢量地图叠加，`out.get("vmap")` 为 None
      时自动跳过（向后兼容纯旧 ckpt）
    - _需求引用：8.1, 8.5_
  - [x] 6.3 端到端前向反向 + 显存自测
    - 全头开启、batch_size 跑一次前向反向，打印峰值显存确认 ≤ 24G
    - _验证：需求 9.4_
  - [x] 6.4 小子集联合多任务验证
    - 在 `cache/meta_2000_vmap.pt` 上开矢量地图 + 规划/占据/栅格地图/检测/运动联合训练，
      确认 loss 正常下降、可视化产物生成
    - _需求引用：9.1, 9.2_
  - [ ] 6.5 全量缓存重建
    - 小子集三件事（数据对齐 / 单头收敛 / 可视化合理）验证通过后，运行
      `python build_meta_cache.py --out cache/meta_34149_vmap.pt`（全量，含矢量地图字段）
    - _需求引用：3.5, 9.2_
  - [-] 6.6 全量多任务最终训练 + 自动可视化
    - 后台 `nohup python -u train.py --meta cache/meta_34149_vmap.pt --vmap-weight ... > log 2>&1`
      （conda `mcts_debug`，单卡 24G，日志重定向）；训练结束产出综合图/俯视图叠加矢量地图栏
    - _需求引用：9.2, 9.4, 9.5_

- [ ] 7. 最终检查点 — 确保测试通过
  - 确保所有属性测试通过、可视化产物生成，有疑问询问用户再继续。

## 说明

- 核心学习点「query → 矢量元素」集中在任务 2.1（`VectorMapHead`）、3.1~3.3（折线匹配与
  loss），务必亲手跑通并通过属性 2（顺序/置换等变）、属性 5（顺序鲁棒）、属性 9（过拟合
  可行性）验证。
- 刻意对称：`VectorMapHead↔DetectionHead`、`match_polyline_to_gt↔match_pred_to_gt`、
  `vector_map_loss↔detection_loss`、`run_vectormap_viz↔run_detection_viz`，实现时对照检测头。
- 遵循「一次加一个改动、小子集先验证再全量重建」：任务 1~5 在小子集跑通并可视化合理后，
  才做全量重建（任务 6.5~6.6）。
- 任务与子任务标注 `*` 者为可选（属性测试 / 集成测试 / 显存自测），可为快速 MVP 跳过；
  顶层任务不带 `*`，核心实现任务不可跳过。
- 每条正确性属性（1~10）对应到具体子任务并标注编号与所验证需求条款。
- 演示入口：小子集构建（1.7/1.8）、`--vmap-only` 训练（5.1/5.3）、`run_vectormap_viz`（5.2）。
- 训练一律后台 + `python -u` + 日志重定向；conda 环境 `mcts_debug`（终端默认 python 无 torch）；
  matplotlib 用 Agg 后端、英文图标题、中文 docstring；单卡 24G。
- 不改动已验证的 image_backbone 与 bev_encoder 几何；地图头挂在现有 BEV 特征之后，与栅格
  `map_head` 并存不替换。
