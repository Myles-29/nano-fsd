"""台阶2-第2.6步演示入口：检测头「仅检测 loss」小子集训练。

目的：在带结构化 agent 列表的小子集缓存（cache/meta_2000_agent.pt）上，只用
detection_loss 训练 DetectionHead（backbone 冻结、其余头不参与反传），验证检测头在
真实数据上能收敛（框/类别/存在性 loss 随 epoch 下降），并存一个 checkpoint 供
visualize.py 叠加预测框可视化。

与 train.py 的关系：train.py 负责「规划 + 占据 + 可行驶区」多任务，多任务 loss 的
统一接线是后续任务 5.1。本脚本是一个**独立、轻量**的检测头专用训练入口，不改动
train.py 现有的规划训练逻辑（遵循「一次加一个改动」节奏）。

记号：N=50 query，n_cls=2（0=车 1=人），BEV 100，SCENE_SCALE=50。

用法:
  python -u train_det.py --meta cache/meta_2000_agent.pt --epochs 30 \
      --run-dir runs/det_sub2000 > runs/det_sub2000.log 2>&1 &
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

from dataset import HEIGHTS, geometry_from_batch, make_loaders
from model import NanoFSDv2
from train import _Tee, detection_loss

IMG_HW = (224, 400)
FUT_STEPS = 12


@torch.no_grad()
def evaluate_det(model, loader, device):
    """验证集平均检测 loss（仅 detection_loss）。"""
    model.eval()
    loss_sum, nb = 0.0, 0
    for batch in loader:
        images = batch["images"].to(device)
        uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, device)
        gt_box = batch["agent_box"].to(device)
        gt_cls = batch["agent_cls"].to(device)
        gt_mask = batch["agent_mask"].to(device)
        # 只要检测分支：forward 返回 dict，读 det（含 agent_fut），其余输出不参与
        det = model(images, uv, valid, command=None)["det"]
        loss = detection_loss(det, gt_box, gt_cls, gt_mask)
        loss_sum += loss.item(); nb += 1
    model.train()
    return loss_sum / max(nb, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="cache/meta_2000_agent.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = os.path.splitext(os.path.basename(args.meta))[0]
    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join(
            "runs", f"{stamp}_det_{tag}_ep{args.epochs}_lr{args.lr:g}_bs{args.batch_size}")
    os.makedirs(args.run_dir, exist_ok=True)

    _logf = open(os.path.join(args.run_dir, "train.log"), "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _logf)
    sys.stderr = _Tee(sys.__stderr__, _logf)

    train_loader, val_loader, (ntr, nva) = make_loaders(
        args.meta, batch_size=args.batch_size, val_ratio=args.val_ratio,
        seed=args.seed, num_workers=args.num_workers)
    print(f"[det] device={dev}  train={ntr} val={nva}  bs={args.batch_size} "
          f"lr={args.lr} epochs={args.epochs}  run_dir={args.run_dir}", flush=True)

    # backbone 冻结：检测头只在 BEV 特征之上学习，省显存/时间
    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      pretrained=True, freeze_backbone=True).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_epoch = float("inf"), -1
    best_path = os.path.join(args.run_dir, "checkpoint_best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, nb = 0.0, 0
        for batch in train_loader:
            images = batch["images"].to(dev)
            uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
            gt_box = batch["agent_box"].to(dev)
            gt_cls = batch["agent_cls"].to(dev)
            gt_mask = batch["agent_mask"].to(dev)
            det = model(images, uv, valid, command=None)["det"]
            loss = detection_loss(det, gt_box, gt_cls, gt_mask)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); nb += 1
        train_loss = loss_sum / max(nb, 1)

        val_loss = evaluate_det(model, val_loader, dev)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        improved = val_loss < best_val
        if improved:
            best_val, best_epoch = val_loss, epoch + 1
            torch.save({"model": model.state_dict()}, best_path)
        print(f"[det] epoch {epoch+1:>3}/{args.epochs}  train_det={train_loss:.6f}  "
              f"val_det={val_loss:.6f}{' *' if improved else ''}  ({time.time()-t0:.1f}s)",
              flush=True)

    torch.save({"model": model.state_dict()}, os.path.join(args.run_dir, "checkpoint_last.pt"))
    with open(os.path.join(args.run_dir, "history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch, "best_val_det": best_val,
                   "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"[det] 完成。最优 val_det={best_val:.6f} (epoch {best_epoch}) -> {best_path}",
          flush=True)


if __name__ == "__main__":
    main()
