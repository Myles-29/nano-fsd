"""一次性验证：改变 BEV 范围，看相机覆盖区的边界形状。

假设："±50m 看到方形"是因为相机在 50m 内无死角、覆盖被方形 BEV 边界裁掉。
验证：把范围放大到 ±150m，若假设成立，远处相机够不到，覆盖区边界应从方形
      变成相机视野极限的圆弧（环视 6 扇形拼合）。

只画"覆盖图"（哪些格子被相机覆盖），不跑 cross-attention（不需要特征，只看几何）。
"""

import argparse
import os

import numpy as np
import torch

from bev_encoder import build_sampling_grid
from bev_geometry import CAMS, DATAROOT, VERSION

IMG_H, IMG_W = 224, 400
ORIG_H, ORIG_W = 900, 1600
HEIGHTS = [0.0, -0.5, -1.0, 0.5]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    parser = argparse.ArgumentParser()
    parser.add_argument("--range", type=float, default=150.0, help="BEV 半径(米)")
    parser.add_argument("--size", type=int, default=150, help="BEV 边长(格子数)")
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--out", type=str, default="outputs/coverage_range150.png")
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cov] device={dev}；range=±{args.range}m  size={args.size}  加载 devkit...",
          flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    sample = nusc.sample[args.sample_idx if args.sample_idx is not None
                         else len(nusc.sample) // 2]

    # 读 6 路相机的缩放后内参 + 外参（和 verify_bev_real 一致）
    sx, sy = IMG_W / ORIG_W, IMG_H / ORIG_H
    Ks, Rs, ts = [], [], []
    for cam in CAMS:
        sd = nusc.get("sample_data", sample["data"][cam])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(calib["camera_intrinsic"], dtype=np.float64)
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        Ks.append(torch.from_numpy(K).float())
        Rs.append(torch.from_numpy(Quaternion(calib["rotation"]).rotation_matrix).float())
        ts.append(torch.tensor(calib["translation"], dtype=torch.float32))
    K = torch.stack(Ks).to(dev)
    R = torch.stack(Rs).to(dev)
    t = torch.stack(ts).to(dev)

    # 指定范围/分辨率的 BEV 网格
    xs = torch.linspace(-args.range, args.range, args.size, device=dev)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    bev_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    heights = torch.tensor(HEIGHTS, device=dev)

    # 特征图尺寸这里不重要（只看可见性 mask，不采样），给个占位 (7,13)
    _, sample_valid = build_sampling_grid(
        bev_xy, heights, K, R, t, (7, 13), (IMG_H, IMG_W))
    cover = sample_valid.any(dim=2).any(dim=1)        # (Q,) 每格是否被任一相机覆盖
    cover_map = cover.reshape(args.size, args.size).cpu().numpy()
    cell_cov = cover.float().mean().item()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cover_map.T, origin="lower", cmap="gray",
              extent=[-args.range, args.range, -args.range, args.range])
    ax.plot(0, 0, "r^", ms=10)
    ax.set_aspect("equal")
    ax.set_title(f"Camera coverage at range=±{args.range}m  (covered={cell_cov:.1%})")
    ax.set_xlabel("x forward (m)"); ax.set_ylabel("y left (m)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[cov] 覆盖格子占比={cell_cov:.3f}  已保存 {args.out}")
    print("[cov] 预期：范围放大后，远处相机够不到，白色覆盖区边界从方形变成 6 扇形拼成的放射状/圆弧。")


if __name__ == "__main__":
    main()
