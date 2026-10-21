"""检测头单头过拟合 sanity check（任务 2.5 / 正确性属性 7：过拟合可行性）。

从带 agent 真值的缓存（cache/meta_2000_agent.pt）取**一个固定样本**，反复训练**仅检测
loss**（框回归 + 类别 CE + 存在性 BCE），确认检测 loss 显著下降 => 「图像->BEV->检测头」
整条梯度链路通畅、可训练。

只开检测 loss：规划/占据/可行驶区头不参与反传（它们的输出此处忽略），通过
`model(...)["det"]` 拿到检测头 det 输出，配合 train.py 的 detection_loss 计算。

坐标/单位沿用全局约定：ego 系、SCENE_SCALE=50 归一化、MAX_GT=N=50、n_cls=2。
"""

import argparse
import os

import torch

from dataset import NuScenesBEVDataset, HEIGHTS, collate, geometry_from_batch
from model import NanoFSDv2
from train import detection_loss

IMG_HW = (224, 400)
FUT_STEPS = 12


def _pick_sample_with_agents(ds, want_idx):
    """选一个有有效 agent 的样本：指定了 --sample-idx 则直接用；否则扫描找第一个
    agent_mask 有有效位的样本（否则检测 loss 没有正样本，过拟合不出信号）。"""
    if want_idx is not None:
        return want_idx
    for i in range(len(ds)):
        if bool(ds[i]["agent_mask"].any()):
            return i
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="cache/meta_2000_agent.pt")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="固定过拟合的样本下标；默认自动找第一个含有效 agent 的样本")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[overfit-det] device={dev}；加载缓存 {args.meta} ...", flush=True)

    ds = NuScenesBEVDataset(args.meta)
    idx = _pick_sample_with_agents(ds, args.sample_idx)
    sample = ds[idx]
    n_valid_gt = int(sample["agent_mask"].sum().item())
    print(f"[overfit-det] 固定样本 idx={idx}，有效 GT agent 数 M={n_valid_gt}", flush=True)
    if n_valid_gt == 0:
        print("[overfit-det] 警告：该样本无有效 agent，检测 loss 无正样本，过拟合无意义。")

    # 组装成单样本 batch（B=1），并在 device 上预算几何采样网格（固定输入，训练中不变）
    batch = collate([sample])
    images = batch["images"].to(dev)                                   # (1,6,3,H,W)
    uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
    command = batch["command"].to(dev) if "command" in batch else None
    gt_box = batch["agent_box"].to(dev)                                # (1,MAX_GT,5)
    gt_cls = batch["agent_cls"].to(dev)                                # (1,MAX_GT)
    gt_mask = batch["agent_mask"].to(dev)                              # (1,MAX_GT)

    # 模型 + 优化器（backbone 冻结；只训 BEV 编码器 + 检测头等可训练参数）
    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, n_query=50, n_cls=2,
                      pretrained=True, freeze_backbone=True).to(dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    # 余弦退火：后期降 lr 让贪心匹配稳定下来（预测不再大幅跳动，query 与 GT 配对固定），
    # 否则匹配每步重排、各 query 的回归目标反复切换，loss 难以压到底。
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(f"[overfit-det] 开始单样本过拟合 {args.steps} 步（仅检测 loss）...", flush=True)
    first_loss = None
    best_loss = float("inf")
    last_loss = None
    for step in range(args.steps):
        # 只取 det 输出，忽略 traj/occ/drivable（只开检测 loss）
        det = model(images, uv, valid, command=command)["det"]
        loss = detection_loss(det, gt_box, gt_cls, gt_mask)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        lv = loss.item()
        if first_loss is None:
            first_loss = lv
        last_loss = lv
        best_loss = min(best_loss, lv)
        if (step + 1) % 50 == 0 or step == 0:
            print(f"  step {step+1:>4}/{args.steps}  det_loss={lv:.6f}  best={best_loss:.6f}",
                  flush=True)

    drop = (first_loss - best_loss) / max(abs(first_loss), 1e-8) * 100.0
    print(f"[overfit-det] 初始 det_loss={first_loss:.6f} -> 最终 det_loss={last_loss:.6f} "
          f"（最优 best_loss={best_loss:.6f}，相对初始下降 {drop:.1f}%）", flush=True)
    # 属性 7：固定 batch 反复训练，检测 loss 显著下降（相对初始下降 ≥50% 视为通过）。
    # 用 best_loss 判定——贪心匹配带来的 loss 抖动不影响「梯度链路是否通畅」的结论。
    verdict = "显著下降 => 检测头梯度链路通畅、可过拟合（属性 7 通过）" \
        if best_loss < first_loss * 0.5 else "下降不足，需检查梯度链路"
    print(f"[overfit-det] 判定：{verdict}", flush=True)


if __name__ == "__main__":
    main()
