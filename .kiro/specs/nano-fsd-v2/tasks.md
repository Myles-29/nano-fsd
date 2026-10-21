# 实现计划：nano-fsd-v2 — 对齐 UniAD 的三头增量

## 概述

本计划把 design.md 的「分步施工路线」（5 步）展开为可独立验证的编码任务，在不推翻现有
BEV 栈的前提下补齐 UniAD 三个核心头：检测头、运动头、交互式规划头。贯穿主线是 **agent
query 从检测→运动→规划的传递**（本次最重要的学习点）。

节奏遵循「一次加一个头、先在小子集验证再推进」：数据层 → 检测头 → 运动头 → 交互式规划头
→ 整合与全量重建训练。每一步都有命令行演示入口与可视化产物。

技术栈沿用：Python + PyTorch，属性测试用 hypothesis，可视化用 matplotlib（Agg 后端、
英文图标题）。所有张量 batch 维在前，坐标 ego 系、`SCENE_SCALE=50` 归一化，`MAX_GT=N=50`。

设计中的正确性属性（供下文引用）：
1. 形状守恒 · 2. query 顺序对齐 · 3. 匹配正确性 · 4. 损失零点 · 5. mask 单调性 ·
6. 空场景安全 · 7. 过拟合可行性 · 8. 规划交互敏感性

## 任务

- [ ] 1. 数据层：结构化 agent 列表 + 缓存重建（小子集）
  - [x] 1.1 在 `build_meta_cache.py` 实现 `build_agent_list`
    - 解析当前帧每个动态 agent：复用 `build_occupancy` 的 global→ego 变换取框中心
      `(x,y)`、朝向 `yaw`、尺寸 `(w,l)`；类别归并为 2 类（0=车、1=人，骑行者就近归入人）
    - 沿 `sample_annotation` 的 instance 链找同一 instance 未来 Tf 帧中心，转到当前帧 ego 系
      并减去当前中心，得相对未来轨迹；instance 消失则对应步 `agent_fut_mask=False`
    - pad 到 `MAX_GT=50`：超出按到自车距离近者优先保留；无动态 agent 时 `agent_mask` 全 False
    - _需求引用：1.1, 1.2, 1.3, 1.4, 1.5, 1.6_
  - [x] 1.2 把新字段写入缓存 record
    - 在 `main()` 的每条 record 加入 `agent_box/agent_cls/agent_mask/agent_fut/agent_fut_mask`
      （保留现有 occ/drivable/traj/command 字段，向后兼容）
    - _需求引用：1.1_
  - [x] 1.3 `dataset.py` 读取新字段并容错
    - `__getitem__` 读出 5 个新字段转张量；坐标用 `SCENE_SCALE` 归一化（与 traj 一致）
    - WHEN record 缺字段（旧缓存）THEN 填零 + mask 全 False
    - `collate` 自动堆叠（已通用，无需改）
    - _需求引用：1.7_
  - [x] 1.4 为数据层写属性测试（hypothesis）
    - **属性 1 形状守恒**：各新字段形状符合 `(50,5)/(50,)/(50,)/(50,Tf,2)/(50,Tf)`
    - **属性 6 空场景安全**：无 agent 样本 `agent_mask` 全 False，读取不报错
    - _验证：正确性属性 1、6；需求 1.6, 1.7_
  - [x] 1.5 重建小子集缓存
    - 运行 `python build_meta_cache.py --num 2000 --out cache/meta_2000_agent.pt`
    - _需求引用：1.8, 7.2_
  - [x] 1.6 演示入口：可视化 agent 真值
    - 新增 `python build_meta_cache.py --viz-agent N`（或独立小脚本）：从缓存取样本，
      在 BEV 俯视图上画 GT agent 框（按类别着色）+ 每个 agent 的 GT 未来轨迹（细线）
    - 保存图到 `outputs/step1_agent_gt_*.png`，人眼确认框位置/朝向/未来轨迹方向合理
    - _需求引用：1.2, 6.1, 7.1_

- [ ] 2. 检测头 DetectionHead + 匹配 + 检测 loss
  - [x] 2.1 在 `model.py` 实现 `DetectionHead`
    - `nn.Parameter` agent_query (N,D)；`L_dec` 层 cross-attn(query→BEV tokens)+FFN+残差/LN
    - BEV (B,D,S,S) 展平成 token 作 K/V，支持 `kv_stride` 平均池化下采样
    - 输出 MLP：`box (B,N,5)`、`cls (B,N,2)`、`score (B,N,1)`，并返回 `query_emb (B,N,D)`
    - _需求引用：2.1, 2.2, 2.3_
  - [x] 2.2 在 `train.py` 实现 `match_pred_to_gt`（最近距离贪心）
    - 预测框中心与有效 GT 中心算距离矩阵，为每个 GT 贪心指派最近且未占用的 query
    - 返回匹配索引对；未匹配 query 作为背景负样本
    - _需求引用：2.4_
  - [x] 2.3 在 `train.py` 实现 `detection_loss`
    - 匹配后：框回归（smooth L1）+ 类别 CE + 存在性 BCE（匹配=正、未匹配=负）
    - WHEN M=0 THEN 返回 0 且不产生 NaN
    - _需求引用：2.5, 2.7_
  - [x] 2.4 为检测头写属性测试（hypothesis）
    - **属性 1 形状守恒**：det 各输出形状恒定
    - **属性 3 匹配正确性**：令部分预测框=GT，匹配必须指派到对应 GT
    - **属性 4 损失零点**：pred=GT、cls=GT 时 detection_loss≈0
    - **属性 6 空场景安全**：M=0 时 loss=0 且反传无 NaN
    - _验证：正确性属性 1、3、4、6；需求 2.6, 2.7_
  - [x] 2.5 接入 NanoFSDv2 并单头过拟合 sanity check
    - 顶层 forward 增加 det 输出（dict 的一部分）；只开检测 loss
    - **属性 7 过拟合可行性**：固定一个 batch 反复训练，检测 loss 显著下降
    - _验证：正确性属性 7；需求 2.8, 5.3_
  - [x] 2.6 演示入口：小子集训练 + 可视化检测框
    - 用 `cache/meta_2000_agent.pt` 跑若干 epoch（仅检测 loss），`visualize.py` 叠加
      高置信度预测框 + GT 框（虚线）到 BEV 栏，存图确认
    - _需求引用：2.8, 6.1, 7.1_

- [ ] 3. 运动头 MotionHead + 运动 loss
  - [x] 3.1 在 `model.py` 实现 `MotionHead`
    - MLP：`query_emb (B,N,D)` → (B,N,Tf*2) → reshape `(B,N,Tf,2)`（相对各 agent 当前中心）
    - 第 i 个 query 的运动与其检测框顺序对齐
    - _需求引用：3.1, 3.2_
  - [x] 3.2 在 `train.py` 实现 `motion_loss`
    - 只在匹配上的 query、`agent_fut_mask` 有效步算 masked MSE
    - _需求引用：3.3_
  - [x] 3.3 为运动头写属性测试（hypothesis）
    - **属性 1 形状守恒**：agent_fut=(B,N,Tf,2)
    - **属性 4 损失零点**：agent_fut=GT 时 motion_loss≈0
    - **属性 5 mask 单调性**：增加无效 agent 不改变有效位 loss
    - _验证：正确性属性 1、4、5；需求 3.4_
  - [x] 3.4 接入模型并叠加训练 sanity check
    - 顶层 forward 增加 agent_fut 输出；检测 + 运动 loss 联合
    - **属性 7 过拟合可行性**：固定 batch 反复训练，运动 loss 显著下降
    - _验证：正确性属性 7；需求 3.5, 5.3_
  - [x] 3.5 演示入口：小子集训练 + 可视化他车预测轨迹
    - `visualize.py` 为每个高置信度预测框画出其预测未来轨迹（细线），GT 未来轨迹对照
    - 存图到 `outputs/step3_motion_*.png`，人眼确认他车轨迹方向合理
    - _需求引用：6.2, 7.1_

- [ ] 4. 交互式规划头 PlanningHeadV2
  - [x] 4.1 在 `model.py` 实现 `PlanningHeadV2`
    - `nn.Parameter` ego_query (1,D)；cross-attn 层1 ego→BEV tokens（kv_stride 下采样）
    - cross-attn 层2 ego→agent query_emb（用 score 作 soft mask）；融合 command embedding
    - MLP → (Tf,2) 增量 → cumsum → `traj (B,Tf,2)`
    - WHEN agent 全背景 THEN 退化为仅 BEV 规划且不报错
    - _需求引用：4.1, 4.2, 4.3_
  - [x] 4.2 顶层加规划头版本开关
    - `NanoFSDv2(use_planner_v2=True/False)`：True 用 V2 交互，False 回退旧 PlanningHead
    - forward 统一走 dict，`dict['traj']` 为当前规划头输出
    - _需求引用：4.6, 5.1_
  - [x] 4.3 为规划头写属性测试（hypothesis）
    - **属性 1 形状守恒**：traj=(B,Tf,2)
    - **属性 8 规划交互敏感性**：「有 agent」vs「agent 全背景」两种输入输出不同
    - _验证：正确性属性 1、8；需求 4.3, 4.4_
  - [x] 4.4 小子集联合多任务训练 + 对比旧规划
    - 规划 + 占据 + 可行驶区 + 检测 + 运动 多任务；
    - **属性 7 过拟合可行性**：固定 batch 规划 loss 显著下降
    - 分别用 `use_planner_v2=True/False` 各训一轮，对比 val ADE（学习对照）
    - _验证：正确性属性 7；需求 4.5, 5.2, 5.3_

- [ ] 5. 整合、训练循环与可视化贯通
  - [x] 5.1 `train.py` 多任务总损失与权重
    - 总损失 `w_plan·L_plan + w_occ·L_occ + w_map·L_map + w_det·L_det + w_mot·L_mot`，
      新增 `--det-weight`、`--mot-weight` 命令行参数（含保守默认）
    - 支持按需只开部分头（分步验证用）
    - _需求引用：5.2, 5.3_
  - [x] 5.2 `model.py` forward 统一 dict 输出 + 调用迁移
    - forward 返回 dict（traj/occ/map/det/agent_fut[,bev]）；train/visualize 迁移到 dict 读取
    - 保留 `return_bev` 行为
    - _需求引用：5.1_
  - [x] 5.3 `visualize.py` 整合检测框 + 他车轨迹到综合图
    - 在现有四栏基础上整合：预测框/GT 框（虚线）、他车预测轨迹；只用验证集样本
    - _需求引用：6.1, 6.2, 6.3_
  - [x] 5.4 显存与端到端前向自测
    - 全头开启、batch_size=4 跑一次前向反向，打印峰值显存确认 ≤ 24G
    - _需求引用：4.4(规划退化)、5.4_

- [ ] 6. 全量重建与最终训练
  - [x] 6.1 重建全量缓存
    - 三个头都在小子集验证通过后，运行
      `python build_meta_cache.py --out cache/meta_34149_agent.pt`（全量，含 agent 列表）
    - _需求引用：1.8, 7.2_
  - [ ] 6.2 全量多任务训练 + 自动可视化
    - 后台 `nohup python -u train.py --meta cache/meta_34149_agent.pt ...`，日志重定向
    - 训练结束自动产出可视化，产物归一到带时间戳+参数的实验目录
    - _需求引用：5.4, 5.5_
  - [ ] 6.3 最终检查点
    - 人眼确认：检测框套住真实车/人、他车预测轨迹方向合理、自车轨迹 vs GT 吻合
    - 对比 use_planner_v2 开/关的 ADE，总结三头带来的变化（学习总结）
    - _需求引用：6.1, 6.2, 6.3, 7.2_

## 说明

- 核心学习点「query 贯穿检测→运动→规划」集中在任务 2.1、3.1、4.1，务必亲手跑通并通过
  属性 2（query 顺序对齐）、属性 8（规划交互敏感性）验证。
- 遵循「一次加一个头、小子集先验证」：任务 2/3/4 各自独立跑通 + 可视化后再进下一步；
  全量重建（任务 6）放在三头都验证通过之后。
- 每步演示入口（1.6、2.6、3.5、4.4、6.2）产物统一输出到 `outputs/` 或实验目录。
- 属性测试统一用 hypothesis，每条正确性属性对应子任务并标注编号。
- 训练一律后台 + `python -u` + 日志重定向；conda 环境用 `mcts_debug`（终端默认 python 无 torch）。
- 不改动已验证的 image_backbone 与 bev_encoder 几何；新头均挂在现有 BEV 特征之后。
