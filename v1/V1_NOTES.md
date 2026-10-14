# nano-fsd v1 — 版本快照

本目录是 nano-fsd 第一版（v1）的**可独立运行**冻结快照。用途：作为后续优化的基线，
随时可回退、对比。所有命令都以 **在本目录内运行**（即先 `cd v1`）为前提。

## 这一版是什么

一个 nanoGPT 量级的端到端自动驾驶模仿学习项目：向量化 BEV 场景 → 共享 Transformer
backbone → 解耦的感知头 / 规划头 → 分阶段冻结训练（阶段一联合训练 backbone+感知头，
阶段二冻结感知栈只训规划头）→ 轨迹预测与 BEV 可视化。

本快照还提供两条额外路径（除分阶段冻结外）：
- **端到端训练 `--stage e2e`**：不加载感知预训练、不冻结，全部参数一起用规划 loss
  从头训（对照"先感知预训练再冻结"的价值）。
- **大数据集支持**：data.py 可预处理 nuScenes v1.0-trainval（约 3 万样本），train.py /
  viz.py 用 `--cache` 切换缓存。

架构要点（详见 docs/design.md）：
- 输入：自车过去 4 帧 `(Th=4, 4)` + 周围至多 N=32 个 agent 各过去 4 帧 `(N, Th, C=9)`。
- Backbone：自车=token 0、各 agent=token 1..N，各自展平过两层 MLP 投影到 D=128，加
  区分自车/他车的**类型 embedding**（刻意不加顺序位置编码），过 3 层 TransformerEncoder
  做全体 token 交互，输出 `(B, N+1, 128)`。约 63.5 万参数（占全模型 95%）。
- 感知头：取 agent token 过两层 MLP → 每个 agent 相对位置 `(B, N, 2)`。
- 规划头：取 ego token 过两层 MLP → 自车未来 Tf=12 步轨迹 `(B, Tf, 2)`。
- 冻结开关 `freeze_perception_stack()`：阶段二冻结 backbone+感知头（约 65 万参数），
  只训规划头（约 1.9 万参数）。

## v1 指标（val ADE，米；seed=0，val_ratio=0.2；数据集 = nuScenes mini）

- 感知（perception_best.pt）：best val ADE ≈ **0.12 m**（1000 epoch，lr=5e-4）。
- 规划（planning_best.pt）：best val ADE ≈ **2.91 m**（1000 epoch，lr=5e-4，
  冻结自上面的感知权重）。

逐 epoch 曲线见 checkpoints/perception_history.json、checkpoints/planning_history.json。

> 归档的两个权重是在 **mini** 上训的。trainval 与 e2e 的实验权重未纳入快照（跑完后
> 存在各自的 runs/ 目录里），需要时按下方命令自行训练。

## 目录内容

```
v1/
├── config.py model.py data.py train.py viz.py   # 源码
├── requirements.txt
├── tests/                                        # 测试（单元 + 属性 + sanity）
├── docs/design.md docs/tasks.md                  # 设计与任务文档
├── data/cache_mini.pt                            # 预处理好的 mini 缓存（可直接训练/可视化）
├── checkpoints/
│   ├── perception_best.pt   planning_best.pt     # 本版最优权重（mini 上训练）
│   └── perception_history.json planning_history.json
└── V1_NOTES.md
```

> 说明：`data/cache_mini.pt` 是 nuScenes v1.0-mini 预处理后的向量缓存，不含原始
> nuScenes 图像/点云。有它即可直接训练、评估、可视化，无需安装 nuscenes-devkit。
> trainval 缓存（`data/cache_trainval.pt`，约 3 万样本）体积较大，未纳入快照，需自行
> 预处理生成（见下方命令）。

## 环境

项目依赖的 torch 在 conda 环境 `mcts_debug` 里。运行前先激活：

```bash
conda activate mcts_debug
# 或直接用其解释器： ~/miniconda3/envs/mcts_debug/bin/python
```

（终端默认的 `python` 指向 base 环境，没有 torch，会报 ModuleNotFoundError。）

## 如何复现 / 使用（均需先 cd v1）

跑测试（在 v1 目录内：29 passed, 2 skipped —— 跳过的 2 个依赖根目录 mini 缓存路径，
正常现象）：
```bash
python -m pytest tests/ -q
```

用归档权重看效果（输出到 outputs/）：
```bash
# 感知：GT(绿) vs 预测(红) agent 位置，灰线连预测→真值
python viz.py --task perception --ckpt checkpoints/perception_best.pt

# 规划：GT(绿) vs 预测(红) 未来轨迹，自动挑直行+转弯样本
python viz.py --task planning --ckpt checkpoints/planning_best.pt
```

从头重训（mini；会在 runs/ 下生成带时间戳的实验目录）：
```bash
# 阶段一：联合训练 backbone + 感知头（正式模式，带 val / 最优权重 / 实验目录）
python train.py --stage 1 --train --epochs 1000 --lr 5e-4 --seed 0

# 阶段二：冻结上一步的感知权重，只训规划头
python train.py --stage 2 --train --epochs 1000 --lr 5e-4 --seed 0 \
    --stage1-ckpt runs/stage1_<时间戳>/checkpoint_stage1_best.pt

# 端到端（不冻结、全参数 + 规划 loss；始终为正式训练模式）
python train.py --stage e2e --epochs 100 --lr 5e-4 --seed 0
```

### 用 nuScenes trainval（约 3 万样本，需自行预处理）

trainval 原始数据按官方分包解压后，元数据在 `*_meta` 目录；数据层只读向量标注，
不碰图像。先预处理成缓存：
```bash
python data.py \
    --dataroot <trainval根>/v1.0-trainval_meta \
    --version v1.0-trainval \
    --out data/cache_trainval.pt
```
再用 `--cache` 切换到该缓存训练 / 可视化（三条路径都支持）：
```bash
python train.py --stage 1   --train --cache data/cache_trainval.pt --epochs 100 ...
python train.py --stage 2   --train --cache data/cache_trainval.pt --stage1-ckpt <...> ...
python train.py --stage e2e         --cache data/cache_trainval.pt --epochs 100 ...
python viz.py   --task planning     --cache data/cache_trainval.pt --ckpt <...>
```
> 若 trainval 缓存不在 v1/data/ 下，`--cache` 用绝对路径最稳妥。

### 命令行参数速查（train.py）

- `--stage {1,2,e2e}`：1=感知；2=冻结感知栈只训规划；e2e=端到端不冻结。
- `--train`：对 stage 1/2 启用正式训练（val 划分 / 最优权重 / 实验目录）；e2e 恒为正式模式。
- `--cache`：数据缓存路径（默认 data/cache_mini.pt）。
- `--stage1-ckpt`：stage 2 要冻结加载的感知权重。
- `--epochs / --lr / --batch-size / --val-ratio / --seed / --run-dir / --patience`：训练超参。

## 已知局限（即 v2 的优化方向）

1. 规划头只读 ego token，**看不见周围 agent**。无法学"因前车刹车而减速"这类交互。
   候选优化：规划头加对 agent token 的 cross-attention。（最高优先级）
2. 规划仍偏欠拟合，转弯场景误差明显大于直行（转弯样本少）。候选：加数据 / 加大模型。
3. 规划头一次性 MLP 回归整条未来轨迹，无时序建模。候选：自回归 / 逐步 rollout。
4. 感知任务偏简单：`agent_gt`（当前帧位置）几乎已包含在输入 `agents` 里，感知头接近
   "抄答案"，ADE 低不代表感知能力强；它主要作用是作为辅助任务把 backbone 的场景
   表示训好供规划复用。候选进阶：预测 agent 未来位置 / BEV 占据栅格。

> 关于数据：v1 默认用 mini（约 364 样本）。本快照已支持 trainval（约 3 万样本），
> 加数据能缓解规划"欠拟合"，但解决不了第 1 条的结构瓶颈——到一定程度规划 ADE 会
> 卡在由规划头结构决定的平台上。因此建议把 trainval 当"确立基线 + 判断是否触顶"的
> 实验，之后转向改规划头结构（cross-attention）。

## 对照实验建议（stage2 冻结 vs e2e 端到端）

用同一 `--cache`、同一 `--seed` 各跑一次，比各自 history.json 的 `best_val_ade`：
- stage 1 → stage 2：先感知预训练，再冻结感知栈训规划。
- e2e：从随机初始化，全参数 + 规划 loss 直接端到端。
这能量化"感知预训练 + 冻结"相对"直接端到端"的增量，判断分阶段训练是否值得。

## 独立运行说明

本快照已在本目录内实测通过：`python -m pytest tests/`（29 passed, 2 skipped）、感知与
规划可视化均能用归档权重出图、`--stage e2e` 端到端入口冒烟测试通过。源码中的默认路径
（如 data/cache_mini.pt）按"在 v1 目录内运行"设计；权重用显式 `--ckpt` / `--stage1-ckpt`
指到 checkpoints/ 下即可。
