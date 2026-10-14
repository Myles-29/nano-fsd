"""v2 成果可视化：加载训练好的 checkpoint，对验证集样本出三类图。

严谨性：用与训练相同的 seed/val_ratio 复现验证集划分，只对**模型没训过的验证样本**
可视化。对每个选中样本出一张综合图：
  (左) BEV 特征能量图（训练后，可与随机初始化对比看结构是否浮现）
  (中) BEV 俯视：预测轨迹(红) vs 真值(绿)
  (右) 前视相机图：预测/真值轨迹投影回道路

用法:
  python visualize.py --ckpt runs/v2_sub2000_ep40/checkpoint_best.pt \
      --meta cache/meta_2000.pt --num 6
"""

import argparse
import os

import numpy as np
import torch

from bev_geometry import CAMS, DATAROOT, VERSION, _resolve_image_path, project_ego_to_cam
from dataset import (BEV_RANGE, BEV_SIZE, HEIGHTS, SCENE_SCALE,
                     NuScenesBEVDataset, geometry_from_batch)
from model import NanoFSDv2

IMG_HW = (224, 400)
ORIG_H, ORIG_W = 900, 1600
FUT_STEPS = 12


def val_indices(n, val_ratio, seed):
    """复现训练时的 val 划分（与 dataset.make_loaders 完全一致）。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_ratio))
    return perm[:n_val]


def densify(xy, k=6):
    out = []
    for i in range(len(xy) - 1):
        for a in np.linspace(0, 1, k, endpoint=False):
            out.append(xy[i] * (1 - a) + xy[i + 1] * a)
    out.append(xy[-1])
    return np.array(out)


def draw_traj_front(ax, rec, pred_m, gt_m):
    """前视相机图上画 预测(红)/真值(绿) 轨迹（z=0 贴地，原图尺度）。"""
    from PIL import Image
    cam_i = CAMS.index("CAM_FRONT")
    # rec 的 K 是缩放后的；投到原图要用原图内参 => 把缩放系数还原
    sx, sy = ORIG_W / IMG_HW[1], ORIG_H / IMG_HW[0]
    K = rec["K"][cam_i].copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    R = rec["R"][cam_i]; t = rec["t"][cam_i]
    ax.imshow(Image.open(_resolve_image_path(rec["img_paths"][cam_i])))
    for pts, color, lab in [(gt_m, "lime", "GT"), (pred_m, "red", "Pred")]:
        p3 = densify(pts)
        p3 = np.concatenate([p3, np.zeros((len(p3), 1))], 1)   # z=0
        uv, valid = project_ego_to_cam(p3, K, R, t, ORIG_W, ORIG_H)
        vu = uv[valid]
        if len(vu) >= 2:
            ax.plot(vu[:, 0], vu[:, 1], "-", c=color, lw=3, label=lab)
    ax.set_xlim(0, ORIG_W); ax.set_ylim(ORIG_H, 0); ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)


def run_visualization(ckpt, meta="cache/meta_2000.pt", num=6, seed=0,
                      val_ratio=0.2, out_dir="outputs/viz_sub2000"):
    """加载 ckpt，对验证集前 num 个样本出三栏图（占据/轨迹/投影），存到 out_dir。

    可被 train.py 在训练结束后直接调用，实现"训练完自动可视化"。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = NuScenesBEVDataset(meta)
    vidx = val_indices(len(ds), val_ratio, seed)
    print(f"[viz] 验证集 {len(vidx)} 个（seed={seed}）；取前 {num} 个可视化")

    model = NanoFSDv2(bev_size=BEV_SIZE, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, pretrained=True, freeze_backbone=True).to(dev)
    state = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[viz] 已加载 {ckpt}")

    os.makedirs(out_dir, exist_ok=True)
    ades = []
    for n, idx in enumerate(vidx[:num]):
        sample = ds[idx]
        rec = ds.records[idx]
        images = sample["images"].unsqueeze(0).to(dev)
        K = sample["K"].unsqueeze(0); R = sample["R"].unsqueeze(0); t = sample["t"].unsqueeze(0)
        uv, valid = geometry_from_batch(K, R, t, IMG_HW, dev)
        command = sample.get("command")
        command = command.unsqueeze(0).to(dev) if command is not None else None
        with torch.no_grad():
            pred, occ_logit, map_logit, bev = model(images, uv, valid,
                                                    command=command, return_bev=True)
        pred_m = pred[0].cpu().numpy() * SCENE_SCALE          # (Tf,2) 米
        gt_m = rec["traj"]                                     # (Tf,2) 米
        mask = rec["traj_mask"]
        ade = np.linalg.norm((pred_m - gt_m)[mask], axis=1).mean()
        ades.append(ade)
        occ_prob = torch.sigmoid(occ_logit[0, 0]).cpu().numpy()   # (S,S) 预测占据
        map_prob = torch.sigmoid(map_logit[0, 0]).cpu().numpy()   # (S,S) 预测可行驶区
        occ_gt = rec["occ"].astype(np.float32)
        map_gt = rec.get("drivable", np.zeros_like(occ_gt)).astype(np.float32)

        fig, (a0, a1, a2, a3) = plt.subplots(1, 4, figsize=(26, 6))
        ext = [-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE]
        axes_x = np.linspace(-BEV_RANGE, BEV_RANGE, occ_gt.shape[0])
        # 第0栏：占据 预测(热力) + 真值轮廓(青)
        im0 = a0.imshow(occ_prob.T, origin="lower", cmap="hot", extent=ext, vmin=0, vmax=1)
        a0.contour(axes_x, axes_x, occ_gt.T, levels=[0.5], colors="cyan", linewidths=1)
        a0.plot(0, 0, "b^", ms=9); a0.set_title("occupancy: pred(hot) vs GT(cyan)")
        fig.colorbar(im0, ax=a0, fraction=0.046)
        # 第1栏：可行驶区 预测(热力) + 真值轮廓(青)
        im1 = a1.imshow(map_prob.T, origin="lower", cmap="hot", extent=ext, vmin=0, vmax=1)
        a1.contour(axes_x, axes_x, map_gt.T, levels=[0.5], colors="cyan", linewidths=1)
        a1.plot(0, 0, "b^", ms=9); a1.set_title("drivable: pred(hot) vs GT(cyan)")
        fig.colorbar(im1, ax=a1, fraction=0.046)

        gm, pm = gt_m[mask], pred_m[mask]
        a2.plot(gm[:, 0], gm[:, 1], "g-o", label="GT")
        a2.plot(pm[:, 0], pm[:, 1], "r-x", label="Pred")
        a2.plot(0, 0, "b^", ms=12, label="Ego")
        a2.set_aspect("equal"); a2.legend(); a2.grid(alpha=0.3)
        a2.set_xlabel("x fwd (m)"); a2.set_ylabel("y left (m)")
        a2.set_title(f"trajectory (ADE={ade:.2f} m)")

        draw_traj_front(a3, rec, pred_m, gt_m)
        a3.set_title("CAM_FRONT: pred(red) vs GT(green)")

        fig.suptitle(f"v2 val sample idx={idx}  ADE={ade:.2f} m")
        out = os.path.join(out_dir, f"val_{n:02d}_idx{idx}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] sample idx={idx}  ADE={ade:.2f} m  -> {out}")

    print(f"[viz] 平均 ADE（这 {len(ades)} 个样本）= {np.mean(ades):.3f} m")
    print(f"[viz] 共 {len(ades)} 张保存到 {out_dir}/")
    return float(np.mean(ades)) if ades else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--meta", type=str, default="cache/meta_2000.pt")
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="outputs/viz_sub2000")
    args = parser.parse_args()
    run_visualization(args.ckpt, args.meta, args.num, args.seed,
                      args.val_ratio, args.out_dir)


if __name__ == "__main__":
    main()
