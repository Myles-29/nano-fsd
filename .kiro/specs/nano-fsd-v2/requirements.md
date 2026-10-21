# 需求文档：nano-fsd-v2 — 对齐 UniAD 的「五脏俱全」端到端自动驾驶

## 简介

本需求基于已确认的 `design.md` 反向提炼，描述 v2 在现有「图像 → BEV → 规划/占据/
可行驶区」管线上，补齐 UniAD 三个核心「脏器」所需满足的能力与验收标准：

1. 目标级检测头（query-based，2 类：车/人）
2. 他车运动预测头（每 agent 未来轨迹）
3. 交互式规划头（ego query 与 BEV + agent query 交互）

核心价值是**学习**：理解 UniAD 的 query 贯穿思想（同一批 agent query 从检测→运动→
规划传递），并在单卡 24G 上亲手跑通、分步验证、出可视化。非目标是刷 SOTA 指标。

## 术语

- **agent query**：N 个可学习向量，经检测头 attend BEV 后代表「一个候选目标」。
- **匹配（match）**：把 N 个预测与 M 个 GT agent 对应起来，才能算检测/运动 loss。
- **MAX_GT=N=50**：每帧结构化 agent 列表的定长上限，与模型 query 数对齐。
- **ADE**：有效未来步上预测与真值轨迹的平均欧氏距离（米）。

## 需求

### 需求 1：数据层提供结构化 agent 列表

**用户故事**：作为学习者，我要让数据缓存除占据栅格外，还提供「一个个 agent」的结构化
真值（框 + 类别 + 未来轨迹），这样检测头和运动头才有监督信号。

**验收标准**
1. WHEN 构建缓存 THEN 每条样本 SHALL 新增字段 `agent_box (MAX_GT,5)`、
   `agent_cls (MAX_GT,)`、`agent_mask (MAX_GT,)`、`agent_fut (MAX_GT,Tf,2)`、
   `agent_fut_mask (MAX_GT,Tf)`。
2. WHEN 解析 agent 框 THEN 坐标 SHALL 用已验证的 global→ego 变换转到当前帧 ego 系
   （x 前、y 左）。
3. WHEN 归并类别 THEN 类别 SHALL 为 2 类：0=车（vehicle）、1=人（human，含骑行者就近归并）。
4. WHEN GT agent 数超过 MAX_GT THEN 系统 SHALL 按到自车距离近者优先保留。
5. WHEN 某 agent 在未来帧消失 THEN 其 `agent_fut_mask` 对应步 SHALL 置 False。
6. WHEN 某帧无动态 agent THEN `agent_mask` SHALL 全 False 且不报错。
7. WHEN 旧缓存缺这些字段 THEN Dataset SHALL 填零并置 mask 全 False（向后兼容）。
8. WHEN 先做小子集验证 THEN 系统 SHALL 支持只重建小子集（如 2000 样本）缓存。

### 需求 2：query-based 检测头

**用户故事**：作为学习者，我要用 N 个 agent query attend BEV 特征输出目标框/类别/
置信度，理解 DETR/UniAD 的 set prediction 范式。

**验收标准**
1. WHEN 输入 BEV 特征 (B,D,S,S) THEN 检测头 SHALL 输出 `box (B,N,5)`、`cls (B,N,2)`、
   `score (B,N,1)`、`query_emb (B,N,D)`。
2. WHEN 经过若干 decoder 层 THEN 每层后 query 形状 SHALL 恒为 (B,N,D) 且顺序一一对应。
3. WHEN 控制显存 THEN BEV token 作 K/V SHALL 支持 `kv_stride` 下采样。
4. WHEN 计算检测 loss THEN 系统 SHALL 先用「最近距离贪心匹配」把预测与有效 GT 配对。
5. WHEN 匹配完成 THEN detection_loss SHALL 含框回归 + 类别 CE + 存在性 BCE
   （匹配为正、未匹配为负）。
6. WHEN 预测框=GT 且类别=GT THEN detection_loss 在有效位上 SHALL ≈0。
7. WHEN 某帧 M=0 THEN detection_loss SHALL 返回 0 且不产生 NaN。
8. WHEN 固定一个 batch 反复训练 THEN 检测 loss SHALL 显著下降（过拟合可行性）。

### 需求 3：他车运动预测头

**用户故事**：作为学习者，我要让每个 agent query 回归该 agent 的未来轨迹，理解
MotionFormer「从 query 预测运动」的思路。

**验收标准**
1. WHEN 输入 `query_emb (B,N,D)` THEN 运动头 SHALL 输出 `agent_fut (B,N,Tf,2)`
   （相对各 agent 当前中心）。
2. WHEN 第 i 个 query THEN 其运动预测 SHALL 与同一 query 的检测框顺序对齐。
3. WHEN 计算运动 loss THEN 系统 SHALL 只在匹配上的 query 且有效未来步算 masked MSE。
4. WHEN `agent_fut`=GT 未来轨迹 THEN motion_loss 在有效位上 SHALL ≈0。
5. WHEN 固定一个 batch 反复训练 THEN 运动 loss SHALL 显著下降。

### 需求 4：交互式规划头

**用户故事**：作为学习者，我要让自车规划从「BEV 全局池化」升级为「ego query attend
BEV + agent query」，让规划能看见具体的周围车，对齐 UniAD 规划交互。

**验收标准**
1. WHEN 输入 BEV + agent query THEN 规划头 SHALL 输出自车轨迹 `traj (B,Tf,2)`
   （增量 cumsum 成绝对轨迹）。
2. WHEN 融合导航意图 THEN 规划头 SHALL 支持可选 command embedding（None 时兼容不加）。
3. WHEN agent 全为背景 THEN 规划头 SHALL 退化为仅 BEV 规划且不报错。
4. WHEN 「有 agent」与「agent 全置背景」两种输入 THEN 规划输出 SHALL 不同
   （证明确实用到了 agent 信息）。
5. WHEN 固定一个 batch 反复训练 THEN 规划 loss SHALL 显著下降。
6. WHEN 需要对照 THEN 系统 SHALL 支持开关回退到旧 PlanningHead（全局池化）。

### 需求 5：顶层整合与多任务训练

**用户故事**：作为学习者，我要把三个新头接入现有模型与训练循环，用多任务 loss
联合训练，并保持单卡 24G 可训。

**验收标准**
1. WHEN 调用模型 forward THEN 返回 SHALL 为 dict，含 traj/occ/map/det/agent_fut。
2. WHEN 多任务训练 THEN 总损失 SHALL 为
   `w_plan·L_plan + w_occ·L_occ + w_map·L_map + w_det·L_det + w_mot·L_mot`，各权重可配。
3. WHEN 分步验证 THEN 系统 SHALL 支持按需只开部分头（先检测、再运动、再规划 v2）。
4. WHEN 全量训练 THEN batch_size SHALL 可维持 4 且峰值显存 ≤ 24G。
5. WHEN 训练结束 THEN 系统 SHALL 自动产出可视化并把日志/checkpoint/图归一到实验目录。

### 需求 6：可视化扩展（感知/预测质量可见）

**用户故事**：作为学习者（无法读张量），我要在可视化里直接看到检测框和他车预测轨迹，
人眼确认效果。

**验收标准**
1. WHEN 可视化验证样本 THEN BEV 栏 SHALL 叠加高置信度预测框 + GT 框（虚线对照）。
2. WHEN 可视化他车运动 THEN 每个预测框 SHALL 画出其预测未来轨迹（细线）。
3. WHEN 可视化 THEN SHALL 只用模型没训过的验证样本（复现相同 seed/val_ratio 划分）。

### 需求 7：分步验证与学习节奏

**用户故事**：作为学习者，我要一次只加一个头、先在小子集验证再推进，确保每步都讲清
「做什么/怎么做/目的/与 UniAD 的对应」。

**验收标准**
1. WHEN 推进某一步 THEN 系统 SHALL 提供独立可运行的演示入口（命令行）与可视化产物。
2. WHEN 三个头都在小子集验证通过 THEN 系统 SHALL 再重建全量缓存做最终训练。
3. WHEN 新增组件 THEN SHALL 配套 hypothesis 属性测试（形状/匹配/损失零点/空场景安全等）。
4. WHEN 代码风格 THEN SHALL 沿用现有中文 docstring、Agg 后端、英文图标题约定。
