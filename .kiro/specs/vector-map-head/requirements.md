# 需求文档：vector-map-head — 对齐 UniAD MapFormer 的「真·矢量化」地图头

## 简介

本需求基于已确认的 `design.md` 反向提炼，描述在现有 nano-fsd-v2 管线上新增一个
**真矢量化地图头**（对齐 UniAD MapFormer，三档方案中的 **tier 2**）所需满足的能力与
验收标准。

现有 `map_head` 只做栅格可行驶区分割（稠密 BCE，输出像素概率图）；本设计补上「像检测头
输出一个个 agent 框那样，输出一条条折线」的能力：M 个地图 query attend BEV 特征，每个
query 回归一条**定长折线**（P 个有序点）+ 类别（车道分隔线 / 道路边界 / 人行横道）+
存在置信度。

核心价值是**学习**：复用 nano-fsd-v2 的主线思想——**query 作为信息载体 + 预测/GT 匹配 +
带 mask 的回归 loss**，让矢量地图头成为检测头的「孪生兄弟」，易学、易验证、易复用现有
匹配与可视化基建。本设计明确采用若干简化（固定点数 P、贪心匹配、无拓扑/无方向），非目标
是刷 SOTA 指标。既有栅格 `map_head` 保留并存，不替换、不删除。

## 术语

- **map query**：M 个可学习向量，经地图头 attend BEV 后，每个代表「一条候选矢量地图元素」。
- **折线（polyline）**：一串有序点 `(P,2)`，表示一条车道分隔线 / 道路边界 / 人行横道边界。
- **P / MAP_POINTS=20**：每条折线固定点数（弧长等距重采样得到）。
- **MAX_MAP=48**：每帧矢量折线定长上限，与模型 map query 数 M 对齐。
- **n_cls=3**：地图元素类别数：0=车道分隔线、1=道路边界、2=人行横道。
- **折线匹配（match）**：把 M 个预测折线与有效 GT 折线对应起来，才能算回归/类别 loss。
- **SCENE_SCALE=50**：ego 系坐标归一化尺度，与 traj/agent 字段一致。
- **BEV_RANGE=50**：ego 周围地图元素抽取/裁剪的半径（米）。
- **顺序鲁棒距离**：两折线逐点 L2 均值对「正序」与「GT 点序翻转」取较小者（线元素无方向）。
- **VectorMapHead**：本设计新增的 query-based 矢量地图头（镜像 DetectionHead）。

## 需求

### 需求 1：数据层抽取矢量地图 GT 折线

**用户故事**：作为学习者，我要让数据缓存除栅格可行驶区外，还提供「一条条矢量折线」的真值
（点序 + 类别 + mask），这样矢量地图头才有监督信号。

**验收标准**
1. WHEN 构建缓存 THEN 每条样本 SHALL 新增字段 `map_pts (MAX_MAP,P,2)`、
   `map_cls (MAX_MAP,)`、`map_mask (MAX_MAP,)`。
2. WHEN 抽取地图元素 THEN 系统 SHALL 从 nuScenes 地图扩展抽取 ego 周围 ±BEV_RANGE 内的
   `lane_divider`、`road_divider`、`road_segment`/`drivable_area` 外轮廓边界、`ped_crossing`。
3. WHEN 归并类别 THEN 类别 SHALL 为 3 类：0=车道分隔线（`lane_divider`+`road_divider`）、
   1=道路边界（`road_segment`/`drivable_area` 外轮廓）、2=人行横道（`ped_crossing`）。
4. WHEN 解析折线点 THEN 系统 SHALL 用已验证的当前帧 global→ego 变换把每个点转到 ego 系
   （x 前、y 左）。
5. WHEN 一条几何跨出 BEV 范围 THEN 系统 SHALL 裁剪到 `|x|<=BEV_RANGE` 且 `|y|<=BEV_RANGE`
   内的点段，被裁成多段时各段各自成一条折线。
6. WHEN GT 折线数超过 MAX_MAP THEN 系统 SHALL 按到自车（原点）距离近者优先保留，远者丢弃。
7. WHEN 某帧 ego 周围无任何地图元素 THEN `map_mask` SHALL 全 False 且不报错。
8. IF 某城市地图加载失败 THEN 系统 SHALL 将该样本 `map_mask` 置全 False 并继续构建。

### 需求 2：弧长等距重采样为定长折线

**用户故事**：作为学习者，我要把任意点数的折线统一成固定 P 点，这样才能定长 batch、好写
形状守恒的属性测试。

**验收标准**
1. WHEN 对一条折线重采样 THEN `resample_polyline` SHALL 返回恰好 P 个点，形状 `(P,2)`。
2. WHEN 重采样一条折线 THEN 输出首点 SHALL 等于原折线首点、末点 SHALL 等于原折线末点。
3. WHEN 重采样一条折线 THEN 输出相邻点间弧长 SHALL 在数值容差内近似相等（弧长等距）。
4. IF 一条折线落入 BEV 范围的点数小于 2 THEN 系统 SHALL 跳过该段不生成折线。
5. IF 一条折线所有点重合（弧长为 0） THEN `resample_polyline` SHALL 返回首点重复 P 次。

### 需求 3：Dataset 读取矢量字段并归一化

**用户故事**：作为学习者，我要在 Dataset 侧把新字段读成张量并与 traj/agent 一致地归一化，
且旧缓存不报错。

**验收标准**
1. WHEN `__getitem__` 读取样本 THEN 系统 SHALL 输出 `map_pts (MAX_MAP,P,2)`、
   `map_cls (MAX_MAP,)`、`map_poly_mask (MAX_MAP,)` 三个张量。
2. WHEN 读取 `map_pts` THEN 系统 SHALL 将 xy 坐标除以 SCENE_SCALE 归一化（与 traj/agent 一致）。
3. WHERE GT mask 键命名 THE 系统 SHALL 使用 `map_poly_mask`，与栅格 `drivable` 字段互不影响。
4. IF 旧缓存缺少 `map_*` 字段 THEN 系统 SHALL 填零并将 `map_poly_mask` 置全 False（向后兼容）。
5. WHERE 先做小子集验证 THE 系统 SHALL 支持仅重建小子集（如 2000 样本）的带矢量地图缓存。

### 需求 4：VectorMapHead 矢量地图头

**用户故事**：作为学习者，我要用 M 个 map query attend BEV 特征，输出每条折线的点序 /
类别 / 存在置信度，理解 UniAD MapFormer「query → 矢量元素」的范式。

**验收标准**
1. WHEN 输入 BEV 特征 `(B,D,S,S)` THEN 地图头 SHALL 输出 `pts (B,M,P,2)`、`cls (B,M,n_cls)`、
   `score (B,M,1)`、`query_emb (B,M,D)`，且无 NaN/Inf。
2. WHEN 经过若干 decoder 层 THEN 每层后 query 形状 SHALL 恒为 `(B,M,D)` 且顺序一一对应。
3. WHEN 输出同一个 query 的结果 THEN 该 query 的 pts/cls/score/query_emb SHALL 来自同一 query
   并保持顺序对齐。
4. WHEN 打乱 batch 内样本顺序 THEN 地图头输出 SHALL 对应打乱（置换等变）。
5. WHERE 控制显存 THE 地图头 SHALL 支持将 BEV token 作 K/V 时按 `kv_stride` 下采样。
6. WHERE 使用锚点位置编码 THE 地图头 SHALL 暴露固定锚点中心 `anchor_xy (M,2)` 供匹配使用。
7. WHEN 输出折线点 THEN `pts` SHALL 为 SCENE_SCALE 归一化的 ego 系坐标且对输入可微。

### 需求 5：折线贪心匹配

**用户故事**：作为学习者，我要把预测折线与 GT 折线贪心配对，才能对匹配上的 query 算回归 /
类别 loss，并理解「哪条 query 负责哪条 GT」。

**验收标准**
1. WHEN 计算距离矩阵 THEN `polyline_distance` SHALL 返回两折线逐点 L2 均值对「正序」与
   「GT 点序翻转」取较小者。
2. WHEN 执行匹配 THEN `match_polyline_to_gt` SHALL 按 GT 到自车距离近者优先，贪心指派最近且
   未占用的预测 query，返回两个等长 1D 索引张量。
3. WHERE 开启类内匹配 THE 系统 SHALL 把不同类的 (GT, query) 距离置为无穷，仅同类可配。
4. WHEN 部分预测折线精确等于某些 GT 折线 THEN 匹配 SHALL 把这些 GT 指派到对应预测 query
   （单射、仅来自有效 GT）。
5. IF 某帧有效 GT 折线数为 0 THEN 匹配 SHALL 返回空索引张量且不报错。

### 需求 6：矢量地图损失（顺序鲁棒 + 空场景安全）

**用户故事**：作为学习者，我要用「折线回归 + 类别 CE + 存在性 BCE」的组合监督地图头，并让
loss 对折线朝向鲁棒、在空场景下不崩。

**验收标准**
1. WHEN 计算矢量地图 loss THEN `vector_map_loss` SHALL 由折线回归（顺序鲁棒逐点 L2）、
   类别 CE、存在性 BCE 三部分组成。
2. WHEN 计算回归与类别项 THEN 系统 SHALL 仅对匹配上的 query 计算，并按匹配总数平均。
3. WHEN 计算存在性 BCE THEN 系统 SHALL 对全部 M 个 query 计算（匹配为正、未匹配为负）。
4. WHEN 匹配 query 的折线等于 GT 折线 THEN 折线回归分量 SHALL 精确为 0。
5. WHEN 对一条折线做点序翻转 THEN `polyline_distance` 对原折线与翻转折线之间 SHALL 为 0，
   且对任意 a、b 满足 `polyline_distance(a,b)==polyline_distance(a,reverse(b))`。
6. WHEN 增加 pad 的无效折线（`map_poly_mask`=False） THEN 有效折线上的 loss 值 SHALL 不变。
7. IF 某 batch 内匹配总数为 0 THEN 回归与类别贡献 SHALL 为 0 且反向传播不产生 NaN
   （分母 clamp 到至少 1），存在性 BCE 仍正常计算。

### 需求 7：顶层整合与权重开关

**用户故事**：作为学习者，我要把矢量地图头接入现有模型 dict 输出与多任务训练循环，并能用
权重开关做分步验证。

**验收标准**
1. WHEN 调用模型 forward 且 `use_vector_map=True` THEN 返回 dict SHALL 新增 `vmap` 键
   （含 `pts`/`cls`/`score`/`query_emb`[,`anchor_xy`]），其余键 traj/occ/map/det/agent_fut 不变。
2. WHERE `use_vector_map=False` THE 系统 SHALL 不构造矢量地图头且 dict 不含 `vmap` 键，
   下游按 `out.get("vmap")` 容错。
3. WHEN 多任务训练 THEN 总损失 SHALL 新增 `w_vmap·L_vmap` 项，`w_vmap` 可配。
4. IF `w_vmap` 设为 0 THEN 系统 SHALL 短路跳过矢量地图监督（分步验证用）。
5. WHERE 分步验证 THE 系统 SHALL 提供独立可运行的单头训练入口 `--vmap-only`
   （镜像 `train_detection_only`）。
6. IF 旧 checkpoint 无 `vector_map.*` 键 THEN 系统 SHALL 按 `use_vector_map` 开关重建模型并
   容错加载。

### 需求 8：可视化扩展（预测 vs GT 折线）

**用户故事**：作为学习者（无法直接读张量），我要在俯视图上同时看到预测折线与 GT 折线，
人眼确认矢量地图头效果。

**验收标准**
1. WHEN 可视化矢量地图 THEN 系统 SHALL 在俯视图上叠加预测折线（实线按类着色）与 GT 折线
   （虚线对照）。
2. WHEN 画折线 THEN 系统 SHALL 遵循 ego 系 x 前 / y 左 约定（横轴=y 左、纵轴=x 前）。
3. WHEN 过滤预测折线 THEN 系统 SHALL 仅画 `sigmoid(score) > score_thr` 的折线，并把 `pts`
   去归一化乘回 SCENE_SCALE 到米。
4. WHEN 可视化 THEN 系统 SHALL 仅使用模型没训过的验证集样本（复现相同 seed/val_ratio 划分）。
5. WHERE 模型输出不含 `vmap`（纯旧 ckpt） THE 综合图/俯视图 SHALL 自动跳过矢量地图栏
   （向后兼容）。

### 需求 9：分步验证与学习节奏

**用户故事**：作为学习者，我要一次只加一个头、先在小子集验证再全量重建，确保每步可运行、
可视化、可讲清与 UniAD 的对应。

**验收标准**
1. WHEN 推进某一步 THEN 系统 SHALL 提供独立可运行的命令行演示入口与可视化产物。
2. WHEN 在小子集完成「数据对齐 / 单头收敛 / 可视化合理」三项验证 THEN 系统 SHALL 再重建
   全量缓存做最终训练。
3. WHEN 固定一个小 batch 反复训练 THEN 矢量地图 loss SHALL 显著下降（过拟合可行性，证明
   BEV→map query→折线梯度链路通畅）。
4. WHERE 控制资源 THE 全量多任务训练 SHALL 维持单卡峰值显存 ≤ 24G。
5. WHEN 代码风格 THEN 系统 SHALL 沿用现有中文 docstring、Agg 后端、英文图标题约定。

### 需求 10：属性测试覆盖

**用户故事**：作为学习者，我要为每个新组件配套 hypothesis 属性测试，用「随机输入下恒成立
的规则」而非少量例子来保证正确性。

**验收标准**
1. WHEN 新增组件 THEN 系统 SHALL 配套 hypothesis 属性测试，沿用 nano-fsd-v2 的风格。
2. WHEN 测试地图头 THEN 系统 SHALL 覆盖形状守恒与 query 顺序/置换等变属性。
3. WHEN 测试匹配与损失 THEN 系统 SHALL 覆盖匹配正确性、损失零点、顺序鲁棒性、mask 单调性、
   空场景安全。
4. WHEN 测试重采样 THEN 系统 SHALL 覆盖等距性、首末点保持，并对 2 点直线 / 重合点 / 闭合环
   做边界单测。
5. WHERE 做集成验证 THE 系统 SHALL 用 `--vmap-only` 小子集训练跑通前向反向，确认 loss 下降、
   可视化产物生成、显存 ≤ 24G。
