# nano-fsd v2 — 路线 B1：图像 → BEV → 端到端规划

v2 从 v1 的"纯向量输入"升级为**从 6 路环视相机图像出发**，做一个精简但真实的
图像 → BEV 特征 → 端到端规划 pipeline（B1：用预训练 CNN 骨干 + 精简 BEV 投影，
不追求完整 BEVFormer / SOTA 精度，目标是能训、能看效果、概念完整）。

## 与 v1 的关系

- v1（`../v1/`）：纯向量输入（agent 状态），backbone 是小 transformer。保留作基线。
- v2：从图像出发，backbone = 预训练 CNN（逐相机提特征）+ 精简 BEV 投影。

## 数据前提（已确认）

- nuScenes trainval：850 scene / 34149 关键帧，6 路相机图像**全量齐全**。
- 向量地图（map expansion v1.3）已就位，可读。
- dataroot（元数据）：`/home/miaozijian/life/mzj_planning/data/v1.0-trainval/v1.0-trainval_meta`
- 图像 blob：`/home/miaozijian/life/mzj_planning/data/v1.0-trainval/v1.0-trainval{01..10}_blobs/`
- GPU：24 GB（BEV 模型吃显存，batch 需很小）。

## 环境

conda 环境 `mcts_debug`（含 torch 2.11 + cu128、nuscenes-devkit）。

## 推进计划（分阶段，每步独立验证）

1. [进行中] 数据地基：单样本读 6 路图 + 相机内外参 + 自车位姿 + 真值，验证标定/坐标正确。
2. 图像 backbone：预训练 ResNet 逐相机提特征。
3. 精简 BEV 投影：把 6 路特征投到 BEV 网格。
4. 任务头：BEV 上接规划头（吃导航指令），可选检测/建图辅助。
5. 端到端训练：借鉴 UniAD —— 不冻结、联合微调。

> 当前阶段只做第 1 步的数据探查，不涉及建模/训练。
