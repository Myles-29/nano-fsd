#!/usr/bin/env bash
# 端到端训练（e2e）—— 不冻结、全参数、仅规划 loss，从头训，用 trainval 大数据集。
# 无需感知预训练权重；e2e 入口本身即正式训练模式（带 val / 最优权重 / 实验目录）。
#
# 用法:
#   bash scripts/train_e2e.sh
set -euo pipefail

# ---- 固定配置（可按需改） ----
CONDA_ENV="mcts_debug"
CACHE="/home/miaozijian/life/mzj_planning/data/cache_trainval.pt"
EPOCHS=100
LR=5e-4
SEED=0
VAL_RATIO=0.2
PATIENCE=15

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_DIR="$(dirname "$SCRIPT_DIR")"
cd "$V1_DIR"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

STAMP="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG="$V1_DIR/nohup_e2e_${STAMP}.log"

echo "[e2e] 启动端到端训练（不冻结, 全参数 + 规划 loss, trainval）"
echo "[e2e] cache=$CACHE  epochs=$EPOCHS  lr=$LR  seed=$SEED  patience=$PATIENCE"

# 注意：e2e 不需要 --train（它恒为正式训练模式）
nohup python -u train.py --stage e2e \
    --cache "$CACHE" \
    --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" \
    --val-ratio "$VAL_RATIO" --patience "$PATIENCE" \
    > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "[e2e] 后台 PID=$PID"
echo "[e2e] 实时日志: tail -f $NOHUP_LOG"
echo "[e2e] 实验目录会在 runs/ 下，名为 <时间戳>_e2e_trainval_ep${EPOCHS}_lr${LR}_bs*_seed${SEED}"
