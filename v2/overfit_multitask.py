"""属性 7 过拟合可行性快检：固定一个 batch，全头多任务 loss 反复训练，确认规划 loss
显著下降（证明 use_planner_v2=True 下 planner_v2 的梯度链路通畅）。

对齐 design.md 属性 7 与任务 4.4。只跑一个固定 batch 若干步，打印规划 loss 轨迹。
"""

import sys

import torch

from dataset import HEIGHTS, geometry_from_batch, make_loaders
from model import NanoFSDv2
from train import (detection_loss, motion_loss, occupancy_loss, planning_loss)

IMG_HW = (224, 400)
FUT_STEPS = 12


def main():
    meta = sys.argv[1] if len(sys.argv) > 1 else "cache/meta_2000_agent.pt"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    use_v2 = not (len(sys.argv) > 3 and sys.argv[3] == "v1")

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _, _ = make_loaders(meta, batch_size=4, val_ratio=0.2,
                                      seed=0, num_workers=2)
    batch = next(iter(train_loader))  # 固定一个 batch 反复训练

    images = batch["images"].to(dev)
    traj_gt = batch["traj_gt"].to(dev)
    mask = batch["traj_mask"].to(dev)
    uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
    occ_gt = batch["occ"].to(dev)
    map_gt = batch["drivable"].to(dev)
    command = batch["command"].to(dev)
    gt_box = batch["agent_box"].to(dev)
    gt_cls = batch["agent_cls"].to(dev)
    gt_mask = batch["agent_mask"].to(dev)
    gt_fut = batch["agent_fut"].to(dev)
    gt_fut_mask = batch["agent_fut_mask"].to(dev)

    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      use_planner_v2=use_v2, pretrained=True, freeze_backbone=True).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    print(f"[overfit] meta={meta} steps={steps} use_planner_v2={use_v2}", flush=True)
    first_plan = None
    for i in range(steps):
        out = model(images, uv, valid, command=command)
        pred, occ_logit, map_logit, det = out["traj"], out["occ"], out["map"], out["det"]
        l_plan = planning_loss(pred, traj_gt, mask)
        l_occ = occupancy_loss(occ_logit, occ_gt)
        l_map = occupancy_loss(map_logit, map_gt)
        l_det = detection_loss(det, gt_box, gt_cls, gt_mask)
        l_mot = motion_loss(det["agent_fut"], det, gt_fut, gt_fut_mask, gt_box, gt_mask)
        loss = l_plan + 1.0 * l_occ + 1.0 * l_map + 0.5 * l_det + 1.0 * l_mot
        opt.zero_grad(); loss.backward(); opt.step()
        if first_plan is None:
            first_plan = l_plan.item()
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  step {i+1:>3}/{steps}  plan={l_plan.item():.6f} occ={l_occ.item():.4f} "
                  f"map={l_map.item():.4f} det={l_det.item():.4f} mot={l_mot.item():.6f}",
                  flush=True)
    last_plan = l_plan.item()
    drop = (first_plan - last_plan) / max(first_plan, 1e-9) * 100
    print(f"[overfit] 规划 loss: {first_plan:.6f} -> {last_plan:.6f}  (下降 {drop:.1f}%)",
          flush=True)
    assert last_plan < first_plan * 0.5, "规划 loss 未显著下降，过拟合可行性(属性7)不满足"
    print("[overfit] 属性 7 过拟合可行性：规划 loss 显著下降（>50%）✓", flush=True)


if __name__ == "__main__":
    main()
