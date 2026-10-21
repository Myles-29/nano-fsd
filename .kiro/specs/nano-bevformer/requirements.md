# 需求文档：nano-bevformer — 把精简版 BEV 升级为「纳米版」UniAD BEVFormer

## 简介

现有 v2 的 BEV 编码器（`v2/bev_encoder.py` 的 `BEVCrossAttention`）是一个「BEVFormer
精简版」：**单帧**、**单融合层**、用**标准 cross-attention**（每个 BEV 格子对其 V×Z 个
投影图像候选做 softmax 加权）、无 BEV 自注意力、Z 为固定高度锚点。它把 6 路图像特征搬到
俯视图，输出稠密 BEV 特征 `(B,D,S,S)` 供下游各任务头使用。

本需求描述把这个精简 BEV 升级为一个结构上对齐 UniAD BEVFormer、但按单卡 24G 显存**缩小**
（分辨率/层数）的「**纳米版 BEVFormer**」所需满足的能力与验收标准。核心升级有四项：

1. **时序 BEV（temporal）**：默认 3 帧（当前 + 过去 2 帧），可配置。用帧间 ego 位姿把历史
   BEV 对齐（warp）到当前帧 ego 系后与当前帧融合，补上单帧缺失的时序/遮挡信息。
2. **可变形 cross-attention（deformable）**：**纯 PyTorch** 自写实现，**不依赖**
   mmcv/mmdet3d（规避 torch 2.11 + cu128 的编译地狱）；要可读、可微。
3. **多层编码器（multi-layer encoder）**：堆叠若干「BEV 自注意力 + 可变形空间 cross-attention
   + FFN」编码层，而非单层融合。
4. **BEV 自注意力（self-attention）**：让 BEV 格子之间互相交换信息。

核心价值仍是**学习**：亲手把 BEVFormer 的四个核心概念（时序对齐、可变形注意力、多层编码、
BEV 自注意力）用可读的纯 PyTorch 写一遍，在单卡 24G 上分步跑通、小子集先验、再全量重建。

**动机记录**：在开环 nuScenes 上，规划头（planner）表现尚可，部分得益于数据集的开环捷径；
但检测（det）/矢量地图（vmap）/运动（motion）等**感知头偏弱**，根因是单帧 BEV 缺少时序与
遮挡信息。补上时序 + 更强的 BEV 预期对这些**感知头**帮助最大。

**非目标（记录在案）**：
- 不以刷 SOTA 指标为目标；
- 不引入重型外部依赖（无 mmcv/mmdet3d/CUDA 自定义算子编译）；
- 不改动各任务头的接口——BEV 输出维持 `(B,D,S,S)`，使下游 detection/vector_map/motion/
  occ/map/planner 全部**无需改动**即可继续工作；
- 不做闭环（closed-loop）评测。

## 术语

- **BEVFormer**：以可学习 BEV query + 时序自注意力 + 空间可变形 cross-attention 多层编码
  得到 BEV 特征的编码器范式（UniAD 的感知基座）。
- **纳米版 BEVFormer / nano-BEVFormer**：本设计中结构对齐 BEVFormer、但分辨率与层数按
  24G 显存缩小的实现，模块名记为 `BEVFormerEncoder`。
- **精简版 BEV / simple BEV**：现有 `BEVCrossAttention`（单帧、标准注意力、单层），本设计
  将其保留为可切换的基线/回退。
- **bev_mode**：BEV 编码器选择开关，取值 `"simple"` | `"bevformer"`。与现有
  `use_planner_v2`/`use_vector_map` 同款：checkpoint 存该开关、重建时按开关构造模型。
- **BEV query**：`S*S` 个可学习格子向量，经编码层更新后 reshape 成 `(B,D,S,S)` BEV 特征。
- **可变形 cross-attention（deformable）**：每个 query 不对所有 K/V 做 softmax，而是预测
  少量采样点偏移与权重，仅在偏移处采样特征后加权求和。
- **空间 cross-attention**：BEV query 经投影几何（`build_sampling_grid`）向 6 路图像特征
  采样信息的可变形 cross-attention。
- **时序自注意力 / 时序融合**：当前帧 BEV query 对「历史 warp 后的 BEV」做可变形注意力或
  融合，获取时序信息。
- **ego 位姿（ego pose）**：某帧车体在全局坐标系下的 4×4 位姿；帧间相对位姿用于 BEV warp。
- **BEV warp**：用帧间相对 ego 位姿把历史帧 BEV 特征网格重采样对齐到当前帧 ego 系。
- **T_frames**：时序帧数（当前 + 历史），默认 3，可配置。
- **S / BEV_SIZE**：BEV 边长（格子数）。**D**：模型维。**V=6**：相机数。**Z**：高度锚点数。
- **N_points**：每个 query 每个参考点的可变形采样点数。
- **N_levels**：可变形注意力使用的特征层数（本纳米版默认单层特征，N_levels=1）。
- **小子集先验**：先只重建小子集缓存（如 2000 样本）验证数据对齐/收敛/显存，再全量重建。

## 需求

### 需求 1：数据层重建以提供历史帧图像与 ego 位姿

**用户故事**：作为学习者，我要让数据缓存在现有单帧字段之外，再提供过去若干帧的图像路径与
各帧 ego 位姿，这样时序 BEV 才有历史输入与对齐所需的几何。

**验收标准**
1. WHEN 构建缓存 THEN 每条样本 SHALL 新增字段 `hist_img_paths (T_hist, V)`（历史帧 6 路
   图像文件路径）、`hist_ego2global (T_hist, 4, 4)`（各历史帧 ego→global 位姿）、
   `cur_ego2global (4, 4)`（当前帧 ego→global 位姿），其中 `T_hist = T_frames - 1`。
2. WHEN 采集历史帧 THEN 系统 SHALL 沿 nuScenes 的 `sample["prev"]` 向过去回溯 `T_hist` 帧。
3. IF 当前样本的历史帧不足 `T_hist`（如场景开头） THEN 系统 SHALL 用最早可得帧重复补齐，
   并在 `hist_valid (T_hist,)` 中将被重复补齐的帧标记为无效。
4. WHEN 保存历史帧相机参数 THEN 系统 SHALL 为每个历史帧保存与当前帧同格式的内参 `K`、
   外参 `cam_R`/`cam_t`（cam→ego），供历史帧各自投影使用。
5. WHEN 旧缓存缺少上述时序字段 THEN Dataset SHALL 用当前帧字段回填历史帧并把 `hist_valid`
   置全 False（向后兼容：退化为单帧行为）。
6. WHERE 先做小子集验证 THE 系统 SHALL 支持仅重建小子集（如 2000 样本）的带时序缓存。
7. WHEN 当前帧字段（图像路径、K/R/t、occ/drivable/traj/command/agent_*/map_*） THEN 系统
   SHALL 保持与现有缓存一致，不破坏已有下游读取。

### 需求 2：Dataset 读取时序字段并计算帧间相对位姿

**用户故事**：作为学习者，我要在 Dataset 侧把历史帧读成张量，并算好「历史帧 ego→当前帧
ego」的相对变换，供模型做 BEV warp。

**验收标准**
1. WHEN `__getitem__` 读取样本 THEN 系统 SHALL 输出当前帧图像 `images (V,3,H,W)` 与历史帧
   图像 `hist_images (T_hist,V,3,H,W)` 两组张量。
2. WHEN 计算相对位姿 THEN 系统 SHALL 输出 `hist2cur (T_hist, 4, 4)`，满足
   `hist2cur = inv(cur_ego2global) @ hist_ego2global`（历史 ego 系点左乘后得当前 ego 系点）。
3. WHEN 输出有效性 THEN 系统 SHALL 输出 `hist_valid (T_hist,)` 布尔张量，与缓存一致。
4. WHERE `T_frames = 1` THE 系统 SHALL 不读取任何历史帧，`hist_images` 的 `T_hist` 维为 0
   且不报错（退化为单帧）。
5. IF 某历史帧图像文件缺失 THEN 系统 SHALL 用当前帧图像回填该历史帧并把对应 `hist_valid`
   置 False，不中断样本读取。
6. WHEN 旧缓存（无时序字段）被读取 THEN 系统 SHALL 返回 `T_hist` 维为 0 的历史张量且
   `hist_valid` 为空，保证单帧路径完全兼容。

### 需求 3：纯 PyTorch 可变形 cross-attention

**用户故事**：作为学习者，我要自己用纯 PyTorch 写一个可读、可微的可变形 cross-attention，
不依赖 mmcv/mmdet3d，这样在 torch 2.11 + cu128 下无需编译自定义算子即可运行。

**验收标准**
1. WHEN 输入 query `(B,Nq,D)`、参考点 `ref_points (B,Nq,N_levels,2)`（归一化到 `[0,1]`）、
   值特征 `value (B,Nv,D)` 与各层空间形状 THEN `DeformableAttention` SHALL 输出
   `(B,Nq,D)`，且无 NaN/Inf。
2. WHEN 前向计算 THEN 模块 SHALL 由 query 线性映射预测采样偏移 `offsets
   (B,Nq,n_heads,N_levels,N_points,2)` 与注意力权重 `weights
   (B,Nq,n_heads,N_levels,N_points)`，权重在 `N_levels*N_points` 维经 softmax 归一化。
3. WHEN 采样特征 THEN 模块 SHALL 用 `F.grid_sample`（`align_corners=False`）在
   `参考点 + 偏移` 处双线性采样 value，再按权重加权求和，**不**使用任何编译型 CUDA 算子。
4. WHEN 对输入求梯度 THEN 输出对 `query`、`value`、`ref_points` SHALL 可微
   （梯度存在且非全零、无 NaN）。
5. WHEN 打乱 batch 内样本顺序 THEN 模块输出 SHALL 对应打乱（batch 置换等变）。
6. WHEN 所有注意力权重集中到单一采样点且该点恰为某 value 位置 THEN 该 query 输出 SHALL
   等于该 value 特征（在数值容差内；可变形注意力退化为点采样的正确性）。
7. WHERE 采样点落在特征图外 THE `grid_sample` SHALL 按 `padding_mode="zeros"` 取零，不报错。

### 需求 4：空间可变形 cross-attention（BEV ← 6 路图像）

**用户故事**：作为学习者，我要用可变形 cross-attention 替代现有标准 cross-attention，让每个
BEV 格子沿投影几何只对少量图像采样点聚合特征，省显存且对齐 BEVFormer。

**验收标准**
1. WHEN 执行空间 cross-attention THEN 系统 SHALL 复用现有 `build_sampling_grid` 产出的
   `sample_uv (B,S*S,V,Z,2)` 与 `sample_valid (B,S*S,V,Z)` 作为参考点与可见性来源。
2. WHEN 某 BEV 格子的 Z 个高度点投到某相机 THEN 该格子 SHALL 以这些投影像素为参考点，在
   对应相机特征图上做可变形采样（偏移相对参考点预测）。
3. IF 某 BEV 格子在所有相机均不可见（`sample_valid` 全 False） THEN 该格子空间
   cross-attention 输出 SHALL 为 0 且不产生 NaN。
4. WHEN 多相机命中同一格子 THEN 系统 SHALL 对各命中相机的采样结果按可见性加权聚合到该格子。
5. WHEN 输入 BEV query `(B,S*S,D)` 与 6 路图像特征 THEN 空间 cross-attention 输出 SHALL 为
   `(B,S*S,D)`，形状与输入 query 一致。

### 需求 5：BEV 自注意力

**用户故事**：作为学习者，我要让 BEV 格子之间互相交换信息（自注意力），补上精简版缺失的
格子间上下文建模。

**验收标准**
1. WHEN 执行 BEV 自注意力 THEN 系统 SHALL 以 BEV query 作 Q/K/V，输出形状 `(B,S*S,D)` 与
   输入一致。
2. WHERE 控制显存 THE BEV 自注意力 SHALL 可配置为可变形自注意力（query 在 BEV 网格上以
   自身位置为参考点做可变形采样），以避免 `(S*S)²` 的全量注意力显存开销。
3. WHEN 打乱 batch 内样本顺序 THEN BEV 自注意力输出 SHALL 对应打乱（batch 置换等变）。
4. WHEN 输入含 BEV 位置编码 THEN 系统 SHALL 为每个格子注入其网格位置信息（位置编码）。

### 需求 6：时序 BEV warp 与融合

**用户故事**：作为学习者，我要把历史帧的 BEV 特征按帧间 ego 位姿对齐到当前帧，再与当前帧
BEV 融合，让 BEV 带上时序信息。

**验收标准**
1. WHEN 对齐历史 BEV THEN `warp_bev` SHALL 用 `hist2cur` 的平面分量把历史 BEV 特征网格重
   采样到当前帧 ego 系，输出形状与输入 BEV `(B,D,S,S)` 一致。
2. WHEN `hist2cur` 为单位变换（历史=当前，无位移无旋转） THEN `warp_bev` 输出 SHALL 等于
   输入 BEV（在数值容差内，恒等 warp）。
3. WHEN warp 使 BEV 平移整数个格子 THEN 输出 SHALL 等于对应整格平移的输入 BEV
   （在边界补零容差内，平移正确性）。
4. WHERE 某历史帧 `hist_valid` 为 False THE 系统 SHALL 不让该历史帧贡献时序信息
   （融合时按有效性加权或跳过），且不产生 NaN。
5. WHEN 融合时序 THEN 系统 SHALL 把「当前帧 BEV」与「warp 后的历史 BEV」通过时序注意力或
   融合层合并，输出形状维持 `(B,D,S,S)`。
6. WHERE `T_frames = 1` THE 系统 SHALL 跳过 warp 与时序融合，直接返回当前帧 BEV（单帧退化）。
7. WHEN warp 的采样点落在当前帧 BEV 范围外 THEN 系统 SHALL 按补零处理，不报错。

### 需求 7：多层 BEVFormer 编码器

**用户故事**：作为学习者，我要把时序自注意力、空间可变形 cross-attention、BEV 自注意力与
FFN 堆成多层编码器，逐层精炼 BEV query，对齐 BEVFormer 的编码结构。

**验收标准**
1. WHEN 构造编码器 THEN `BEVFormerEncoder` SHALL 由 `n_layers`（可配，纳米版默认 ≤3）个
   编码层堆叠，每层含「时序融合/自注意力 + 空间可变形 cross-attention + BEV 自注意力 +
   FFN」，各子层带残差与 LayerNorm。
2. WHEN 逐层前向 THEN 每层后 BEV query 形状 SHALL 恒为 `(B,S*S,D)`（循环不变式）。
3. WHEN 编码器输出 THEN 系统 SHALL 把最终 BEV query reshape 成 `(B,D,S,S)` 的稠密 BEV
   特征图。
4. WHEN 输入 6 路图像特征、`sample_uv`/`sample_valid`、历史 BEV 与 `hist2cur` THEN 编码器
   输出 SHALL 为 `(B,D,S,S)` 且无 NaN/Inf。
5. WHEN `n_layers=1` 且关闭时序与自注意力 THEN 编码器 SHALL 退化为仅空间 cross-attention
   的单层行为（可与精简版对照）。

### 需求 8：bev_mode 开关与 checkpoint 兼容

**用户故事**：作为学习者，我要用一个 `bev_mode` 开关在「精简版」与「纳米 BEVFormer」间切换，
用于 A/B 对照与回退，并像现有开关一样把它存进 checkpoint、按它重建模型。

**验收标准**
1. WHEN 构造 `NanoFSDv2` 且 `bev_mode="bevformer"` THEN 系统 SHALL 用 `BEVFormerEncoder`
   作 BEV 编码器；WHERE `bev_mode="simple"` THE 系统 SHALL 用现有 `BEVCrossAttention`。
2. WHEN BEV 编码器切换 THEN 下游各任务头（occ/map/detection/vector_map/motion/planner）
   SHALL 无需改动即可工作（因 BEV 输出均为 `(B,D,S,S)`）。
3. WHEN 保存 checkpoint THEN 系统 SHALL 把 `bev_mode` 一并写入（与 `use_planner_v2`/
   `use_vector_map` 同款）。
4. WHEN 加载 checkpoint THEN 系统 SHALL 按 checkpoint 内的 `bev_mode` 重建模型后再载入
   权重。
5. IF 旧 checkpoint 不含 `bev_mode` 键 THEN 系统 SHALL 默认按 `"simple"` 重建模型并容错
   加载（向后兼容既有精简版 checkpoint）。
6. WHERE `bev_mode="simple"` THE 系统 SHALL 不构造 `BEVFormerEncoder`、不要求时序输入，
   保持纯单帧前向路径。

### 需求 9：BEV 输出形状不变性（下游头不改）

**用户故事**：作为学习者，我要保证无论用哪种 BEV 编码器，BEV 输出形状与语义约定都不变，
这样现有任务头与可视化不用动。

**验收标准**
1. WHEN 任一 `bev_mode` 下调用 BEV 编码器 THEN 输出 SHALL 恒为 `(B,D,S,S)`，`D`/`S` 与
   模型配置一致。
2. WHEN BEV 输出坐标约定 THEN 系统 SHALL 维持与精简版一致的 ego 系网格约定（x 前、y 左、
   ±BEV_RANGE 范围、`S×S` 格），使 detection/vector_map 的锚点与可视化无需改动。
3. WHEN `bev_mode="bevformer"` 下运行一次完整 forward THEN 返回 dict 的键集合（traj/occ/
   map/det/agent_fut[/vmap][/bev]）SHALL 与 `bev_mode="simple"` 时一致。

### 需求 10：单卡 24G 显存预算与纳米尺寸

**用户故事**：作为学习者，我要让纳米 BEVFormer 在单卡 24G 上可训，用缩小的分辨率/层数/
采样点数把显存压进预算。

**验收标准**
1. WHEN 全量多任务训练使用 `bev_mode="bevformer"` 且默认纳米配置 THEN 单卡峰值显存 SHALL
   ≤ 24G。
2. WHERE 显存吃紧 THE 系统 SHALL 提供可配旋钮（`bev_size`、`n_layers`、`N_points`、
   `T_frames`、空间/时序 K/V 的 `kv_stride`）用于缩减显存。
3. WHEN 默认纳米配置 THEN `T_frames` SHALL 默认为 3、`n_layers` SHALL 默认 ≤3，使显存与算力
   适配 24G。
4. WHEN 提供自测入口 THEN 系统 SHALL 有形状 + 显存自测脚本（沿用现有 `selftest_mem.py`
   风格），打印 BEV 输出形状、参数量与峰值显存。
5. WHEN backbone 处理 THEN 系统 SHALL 保持默认冻结 backbone，并支持 `--unfreeze-backbone`。

### 需求 11：分步验证与小子集先验的学习节奏

**用户故事**：作为学习者，我要一次只加一个改动、先在小子集验证再全量重建，确保每步可运行、
可视化、可讲清与 BEVFormer 的对应。

**验收标准**
1. WHEN 推进某一步 THEN 系统 SHALL 提供独立可运行的命令行演示入口与可视化/日志产物。
2. WHEN 在小子集完成「时序数据对齐 / 单组件收敛 / 显存达标」验证 THEN 系统 SHALL 再重建
   全量时序缓存做最终训练。
3. WHEN 固定一个小 batch 反复训练 `bev_mode="bevformer"` THEN 感知头（det/vmap/motion）
   loss SHALL 显著下降（过拟合可行性，证明图像→BEVFormer→头的梯度链路通畅）。
4. WHEN 做 A/B 对照 THEN 系统 SHALL 支持在相同数据/种子下分别用 `simple` 与 `bevformer`
   训练并比较感知头指标。
5. WHEN 代码风格 THEN 系统 SHALL 沿用现有中文 docstring、Matplotlib Agg 后端、英文图标题、
   hypothesis 属性测试约定。

### 需求 12：属性测试覆盖

**用户故事**：作为学习者，我要为每个新组件配套 hypothesis 属性测试，用「随机输入下恒成立
的规则」而非少量例子来保证正确性，沿用 nano-fsd-v2 / vector-map-head 的测试风格。

**验收标准**
1. WHEN 新增组件 THEN 系统 SHALL 配套 hypothesis 属性测试，沿用现有测试目录与风格
   （`v2/tests/`）。
2. WHEN 测试可变形注意力 THEN 系统 SHALL 覆盖形状守恒、batch 置换等变、对输入可微
   （梯度非 NaN）、单点权重退化为点采样的正确性、越界采样补零安全。
3. WHEN 测试空间 cross-attention THEN 系统 SHALL 覆盖形状守恒、无可见候选格子输出为 0 的
   空场景安全、无 NaN。
4. WHEN 测试 BEV warp THEN 系统 SHALL 覆盖恒等 warp（单位相对位姿输出=输入）、整格平移
   正确性、越界补零、无效历史帧不贡献的安全性。
5. WHEN 测试 BEVFormer 编码器 THEN 系统 SHALL 覆盖 BEV 输出形状恒为 `(B,D,S,S)`、逐层
   query 形状不变、`T_frames=1` 单帧退化、无 NaN/Inf。
6. WHEN 测试数据层时序字段 THEN 系统 SHALL 覆盖历史张量形状守恒、`hist2cur` 关系
   （`hist2cur @ cur = hist` 的逆关系在容差内成立）、历史不足补齐与 `hist_valid` 标记、
   旧缓存向后兼容（`T_hist` 维为 0）。
7. WHERE 做集成验证 THE 系统 SHALL 用小子集训练跑通 `bev_mode="bevformer"` 前向反向，确认
   loss 下降、可视化产物生成、显存 ≤ 24G。
