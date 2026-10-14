"""v2 端到端训练：图像 -> BEV -> 轨迹，用 planning loss 全量/子集训练。

沿用 v1 风格：固定 seed、train/val 划分、每 epoch 评估规划 ADE(米)、按 val ADE 存
最优权重、带时间戳实验目录、训练后自动可视化。backbone 默认冻结（省显存/时间）。

记号：ADE = 有效未来步上预测与真值轨迹的平均欧氏距离（米）。
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from dataset import SCENE_SCALE, geometry_from_batch, make_loaders
from model import NanoFSDv2

IMG_HW = (224, 400)
FUT_STEPS = 12


import sys

import torch.nn.functional as F


class _Tee:
    """stdout 同时写终端 + 文件，用于把训练日志落盘到 run_dir/train.log。"""

    def __init__(self, *streams):
        self._s = streams

    def write(self, data):
        for s in self._s:
            s.write(data); s.flush()
        return len(data)

    def flush(self):
        for s in self._s:
            s.flush()


def planning_loss(pred, gt, mask):
    """masked MSE：只在有效未来步上算。pred/gt (B,Tf,2), mask (B,Tf)。"""
    err = ((pred - gt) ** 2).sum(-1)                 # (B,Tf)
    m = mask.float()
    return (err * m).sum() / m.sum().clamp(min=1)


def occupancy_loss(occ_logit, occ_gt):
    """占据头 BCE。occ_logit (B,1,S,S), occ_gt (B,S,S) in {0,1}。

    占据很稀疏(~1%)，给正类加权（pos_weight）平衡，否则模型会全预测 0。
    """
    occ_gt = occ_gt.unsqueeze(1)                     # (B,1,S,S)
    pos = occ_gt.mean().clamp(min=1e-4)
    pos_weight = ((1 - pos) / pos).clamp(max=100.0)  # 正类稀疏 -> 加权
    return F.binary_cross_entropy_with_logits(
        occ_logit, occ_gt, pos_weight=pos_weight)


@torch.no_grad()
def evaluate(model, loader, device):
    """验证集平均规划 ADE(米) 与 loss。"""
    model.eval()
    dist_sum, n_valid, loss_sum, weight = 0.0, 0, 0.0, 0.0
    for batch in loader:
        images = batch["images"].to(device)
        traj_gt = batch["traj_gt"].to(device)
        mask = batch["traj_mask"].to(device)
        uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, device)
        command = batch["command"].to(device)
        pred, _occ, _map = model(images, uv, valid, command=command)
        loss_sum += planning_loss(pred, traj_gt, mask).item() * mask.sum().item()
        weight += mask.sum().item()
        err_m = (pred - traj_gt) * SCENE_SCALE
        d = err_m.pow(2).sum(-1).sqrt()              # (B,Tf) 米
        dist_sum += d[mask].sum().item()
        n_valid += int(mask.sum().item())
    model.train()
    if n_valid == 0:
        return float("nan"), float("nan")
    return loss_sum / weight, dist_sum / n_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="cache/meta_2000.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="联合微调 ResNet（更强但更吃显存/更慢）")
    parser.add_argument("--occ-weight", type=float, default=1.0,
                        help="占据 loss 权重")
    parser.add_argument("--map-weight", type=float, default=1.0,
                        help="可行驶区分割 loss 权重")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = os.path.splitext(os.path.basename(args.meta))[0]
    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join(
            "runs", f"{stamp}_v2_{tag}_ep{args.epochs}_lr{args.lr:g}_bs{args.batch_size}")
    os.makedirs(args.run_dir, exist_ok=True)

    # 训练日志同时落盘到 run_dir/train.log（行缓冲，实时可见）
    _logf = open(os.path.join(args.run_dir, "train.log"), "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _logf)
    sys.stderr = _Tee(sys.__stderr__, _logf)

    train_loader, val_loader, (ntr, nva) = make_loaders(
        args.meta, batch_size=args.batch_size, val_ratio=args.val_ratio,
        seed=args.seed, num_workers=args.num_workers)
    print(f"[train] device={dev}  train={ntr} val={nva}  bs={args.batch_size} "
          f"lr={args.lr} epochs={args.epochs}  run_dir={args.run_dir}", flush=True)

    from dataset import HEIGHTS
    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      pretrained=True, freeze_backbone=not args.unfreeze_backbone).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_ade": []}
    best_ade, best_epoch = float("inf"), -1
    best_path = os.path.join(args.run_dir, "checkpoint_best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, nb = 0.0, 0
        for batch in train_loader:
            images = batch["images"].to(dev)
            traj_gt = batch["traj_gt"].to(dev)
            mask = batch["traj_mask"].to(dev)
            uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
            occ_gt = batch["occ"].to(dev)
            map_gt = batch["drivable"].to(dev)
            command = batch["command"].to(dev)
            pred, occ_logit, map_logit = model(images, uv, valid, command=command)
            l_plan = planning_loss(pred, traj_gt, mask)
            l_occ = occupancy_loss(occ_logit, occ_gt)
            l_map = occupancy_loss(map_logit, map_gt)        # 可行驶区分割，同为带权 BCE
            loss = l_plan + args.occ_weight * l_occ + args.map_weight * l_map
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += l_plan.item(); nb += 1               # 记录规划 loss 便于和之前对比
        train_loss = loss_sum / max(nb, 1)

        val_loss, val_ade = evaluate(model, val_loader, dev)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ade"].append(val_ade)
        improved = val_ade < best_ade
        if improved:
            best_ade, best_epoch = val_ade, epoch + 1
            torch.save({"model": model.state_dict()}, best_path)
        print(f"[train] epoch {epoch+1:>3}/{args.epochs}  train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  val_ADE={val_ade:.3f} m"
              f"{' *' if improved else ''}  ({time.time()-t0:.1f}s)", flush=True)

    torch.save({"model": model.state_dict()}, os.path.join(args.run_dir, "checkpoint_last.pt"))
    with open(os.path.join(args.run_dir, "history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch, "best_val_ade": best_ade,
                   "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"[train] 完成。最优 val_ADE={best_ade:.3f} m (epoch {best_epoch}) -> {best_path}")

    # 训练完自动可视化：用 best 权重对验证集出三栏图，存到 run_dir/viz/
    try:
        from visualize import run_visualization
        viz_dir = os.path.join(args.run_dir, "viz")
        print(f"[train] 自动可视化（用最优权重）-> {viz_dir}", flush=True)
        run_visualization(best_path, meta=args.meta, num=6, seed=args.seed,
                          val_ratio=args.val_ratio, out_dir=viz_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[train] 自动可视化失败（不影响训练产物）: {e}", flush=True)


if __name__ == "__main__":
    main()
