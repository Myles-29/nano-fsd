#!/usr/bin/env bash
# 感知训练（stage 1）—— 联合训练 backbone + 感知头，用 trainval 大数据集。
# 后台运行，日志写入实验目录下的 train.log，同时在当前目录留一份 nohup 输出便于 tail。
#
# 用法:
#   bash scripts/train_perception.sh
# 查看进度:
#   脚本启动后会打印实验目录与日志路径，用 tail -f <日志> 跟踪。
set -euo pipefail

# ---- 固定配置（可按需改） ----
CONDA_ENV="mcts_debug"
CACHE="/home/miaozijian/life/mzj_planning/data/cache_trainval.pt"
EPOCHS=30
LR=5e-4
SEED=0
VAL_RATIO=0.2
PATIENCE=8          # 感知收敛快，早停防浪费

# ---- 切到 v1 目录（代码与相对路径均以此为工作目录） ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_DIR="$(dirname "$SCRIPT_DIR")"
cd "$V1_DIR"

# ---- 激活 conda 环境（终端默认 python 无 torch） ----
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

STAMP="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG="$V1_DIR/nohup_perception_${STAMP}.log"

echo "[perception] 启动感知训练（stage 1, trainval）"
echo "[perception] cache=$CACHE  epochs=$EPOCHS  lr=$LR  seed=$SEED  patience=$PATIENCE"

# -u 关键：无缓冲输出，nohup 日志实时可见
nohup python -u train.py --stage 1 --train \
    --cache "$CACHE" \
    --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" \
    --val-ratio "$VAL_RATIO" --patience "$PATIENCE" \
    > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "[perception] 后台 PID=$PID"
echo "[perception] 实时日志: tail -f $NOHUP_LOG"
echo "[perception] 训练完实验目录（含 train.log / checkpoints / 可视化）会在 runs/ 下，"
echo "             名为 <时间戳>_stage1_trainval_ep${EPOCHS}_lr${LR}_bs*_seed${SEED}"
