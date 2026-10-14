#!/usr/bin/env bash
# 规划训练（stage 2）—— 冻结感知栈，只训规划头，用 trainval 大数据集。
# 依赖一个已训好的感知权重（stage 1 的 *_best.pt），请先在下方 STAGE1_CKPT 填好路径。
#
# 用法:
#   1) 先改下面的 STAGE1_CKPT 指向你的感知 best 权重
#   2) bash scripts/train_planning.sh
set -euo pipefail

# ---- 必填：感知 best 权重路径（相对 v1 目录，或绝对路径） ----
# 例: runs/20261013_xxxxxx_stage1_trainval_ep30_lr5e-4_bs64_seed0/checkpoint_stage1_best.pt
STAGE1_CKPT="/home/miaozijian/life/mzj_planning/v1/runs/20261013_120543_stage1_trainval_ep30_lr0.0005_bs64_seed0/checkpoint_stage1_best.pt"

# ---- 固定配置（可按需改） ----
CONDA_ENV="mcts_debug"
CACHE="/home/miaozijian/life/mzj_planning/data/cache_trainval.pt"
EPOCHS=100
LR=5e-4
SEED=0              # 与感知训练同 seed，保证 val 划分一致（冻结 backbone 未见过 val）
VAL_RATIO=0.2
PATIENCE=15

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_DIR="$(dirname "$SCRIPT_DIR")"
cd "$V1_DIR"

# ---- 校验感知权重已填且存在 ----
if [[ "$STAGE1_CKPT" == *REPLACE_ME* ]]; then
    echo "[planning] 错误：请先编辑本脚本，把 STAGE1_CKPT 改成你的感知 best 权重路径。" >&2
    exit 1
fi
if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "[planning] 错误：找不到感知权重 $STAGE1_CKPT" >&2
    echo "            先运行 scripts/train_perception.sh，再把其 *_best.pt 路径填进来。" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

STAMP="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG="$V1_DIR/nohup_planning_${STAMP}.log"

echo "[planning] 启动规划训练（stage 2, 冻结感知栈, trainval）"
echo "[planning] stage1_ckpt=$STAGE1_CKPT"
echo "[planning] cache=$CACHE  epochs=$EPOCHS  lr=$LR  seed=$SEED  patience=$PATIENCE"

nohup python -u train.py --stage 2 --train \
    --stage1-ckpt "$STAGE1_CKPT" \
    --cache "$CACHE" \
    --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" \
    --val-ratio "$VAL_RATIO" --patience "$PATIENCE" \
    > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "[planning] 后台 PID=$PID"
echo "[planning] 实时日志: tail -f $NOHUP_LOG"
echo "[planning] 实验目录会在 runs/ 下，名为 <时间戳>_stage2_trainval_ep${EPOCHS}_..."
