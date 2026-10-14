"""可视化 BEV 特征图（当前为随机初始化、未训练状态，作为训练前对照基线）。

对一个真实样本跑完整 图像->BEV，把 (128,100,100) 的 BEV 特征用三种方式画出来：
  1) 能量图：每格 128 维向量的模长（看哪些格子特征强）。
  2) 单通道图：挑若干通道各画一张热力图（看单个特征的空间分布）。
  3) PCA 伪彩色：把 128 维压到 3 维映射成 RGB（看高维特征的整体空间结构）。

说明：此时 cross-attention 为随机初始化，特征"内容"尚无语义，预期偏噪声；
几何是对的（覆盖/投影已验证）。训练后再画一张对比，即可看到结构浮现。
"""

import argparse
import os

import numpy as np
import torch

from bev_encoder import BEVCrossAttention, build_sampling_grid
from bev_geometry import DATAROOT, VERSION
from image_backbone import ImageBackbone
from verify_bev_real import (BEV_RANGE, BEV_SIZE, HEIGHTS, load_real_sample)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuscenes.nuscenes import NuScenes

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="模型随机初始化种子")
    parser.add_argument("--out", type=str, default="outputs/bev_feature_untrained.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[feat] device={dev}；加载 devkit...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    sample = nusc.sample[args.sample_idx if args.sample_idx is not None
                         else len(nusc.sample) // 2]

    # 真实样本 -> 图像特征 -> 采样网格 -> BEV cross-attention
    images, K, R, t = load_real_sample(nusc, sample)
    images, K, R, t = images.to(dev), K.to(dev), R.to(dev), t.to(dev)
    bb = ImageBackbone(pretrained=True, freeze=True).to(dev).eval()
    with torch.no_grad():
        feats = bb(images.unsqueeze(0))
    _, V, Cf, Hf, Wf = feats.shape

    xs = torch.linspace(-BEV_RANGE, BEV_RANGE, BEV_SIZE, device=dev)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    bev_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    heights = torch.tensor(HEIGHTS, device=dev)
    sample_uv, sample_valid = build_sampling_grid(
        bev_xy, heights, K, R, t, (Hf, Wf), (224, 400))

    enc = BEVCrossAttention(d_model=128, feat_channels=Cf, bev_size=BEV_SIZE,
                            n_heights=len(HEIGHTS), n_cams=V).to(dev).eval()
    with torch.no_grad():
        bev = enc(feats, sample_uv.unsqueeze(0), sample_valid.unsqueeze(0))[0]  # (128,S,S)
    print(f"[feat] BEV 特征 {tuple(bev.shape)}（随机初始化、未训练）")

    S = BEV_SIZE
    ext = [-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE]
    bev_np = bev.cpu().numpy()                         # (128,S,S)

    # 画布：1 张能量 + 4 张单通道 + 1 张 PCA 伪彩色 = 6 格
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1) 能量图
    energy = np.sqrt((bev_np ** 2).sum(0))
    a = axes[0, 0]
    im = a.imshow(energy.T, origin="lower", cmap="viridis", extent=ext)
    a.plot(0, 0, "r^", ms=9); a.set_title("feature energy |feat|")
    fig.colorbar(im, ax=a, fraction=0.046)

    # 2) 4 个单通道
    for i, ch in enumerate([0, 1, 2, 3]):
        a = axes.ravel()[1 + i]
        im = a.imshow(bev_np[ch].T, origin="lower", cmap="coolwarm", extent=ext)
        a.plot(0, 0, "k^", ms=8); a.set_title(f"channel {ch}")
        fig.colorbar(im, ax=a, fraction=0.046)

    # 3) PCA 伪彩色（128 -> 3 -> RGB）
    flat = bev_np.reshape(128, -1).T                  # (S*S, 128)
    flat = flat - flat.mean(0, keepdims=True)
    # 简单 PCA：取协方差前 3 主成分
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    rgb = (u[:, :3] * s[:3])                           # (S*S, 3)
    rng3 = rgb.max(0) - rgb.min(0)
    rgb = (rgb - rgb.min(0)) / (rng3 + 1e-8)           # 归一化到 [0,1]
    rgb = rgb.reshape(S, S, 3)
    a = axes[1, 2]
    a.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=ext)
    a.plot(0, 0, "w^", ms=9); a.set_title("PCA pseudo-color (128->3 RGB)")

    for a in axes.ravel():
        a.set_xlabel("x fwd (m)"); a.set_ylabel("y left (m)")
    fig.suptitle(f"BEV feature maps (UNTRAINED, random init) — sample {sample['token'][:8]}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[feat] 已保存 BEV 特征可视化到 {args.out}")
    print("[feat] 预期：未训练，特征偏噪声、无明显语义结构；覆盖区内有能量，盲区为零。")


if __name__ == "__main__":
    main()
