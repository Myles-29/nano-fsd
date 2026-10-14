"""v2 第 3b 真实数据端到端验证：真实样本 6 路图 -> CNN -> BEV cross-attention。

第一次把整条"图像->BEV"串起来跑一个真实样本，并画 BEV 覆盖图，确认：
  1) 用真实相机内外参时，可见采样点占比不再是 0（相机确实覆盖了部分 BEV 网格）。
  2) 输出 BEV 特征图：被相机覆盖的格子非零、无覆盖格子为零。

关键细节：内参 K 是原图(1600x900)尺度，图像缩放到 (224,400) 喂 CNN，故 K 要按
缩放比例同步缩放，否则投影像素和特征图对不上。
"""

import argparse
import os

import numpy as np
import torch

from bev_encoder import BEVCrossAttention, build_sampling_grid
from bev_geometry import CAMS, DATAROOT, VERSION, _resolve_image_path
from image_backbone import ImageBackbone

IMG_H, IMG_W = 224, 400       # 喂 CNN 的缩放后尺寸
ORIG_H, ORIG_W = 900, 1600    # nuScenes 原图尺寸
BEV_RANGE, BEV_SIZE = 50.0, 100
HEIGHTS = [0.0, -0.5, -1.0, 0.5]


def load_real_sample(nusc, sample):
    """读一个样本的 6 路图(缩放后张量) + 缩放后内参 K + 外参(cam->ego R/t)。"""
    from PIL import Image
    from pyquaternion import Quaternion

    imgs, Ks, Rs, ts = [], [], [], []
    sx, sy = IMG_W / ORIG_W, IMG_H / ORIG_H     # 缩放比例
    for cam in CAMS:
        sd = nusc.get("sample_data", sample["data"][cam])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        img = Image.open(_resolve_image_path(sd["filename"])).convert("RGB")
        img = img.resize((IMG_W, IMG_H))
        arr = torch.from_numpy(np.asarray(img)).float() / 255.0   # (H,W,3) [0,1]
        imgs.append(arr.permute(2, 0, 1))                          # (3,H,W)

        K = np.array(calib["camera_intrinsic"], dtype=np.float64)
        # 内参随图像缩放：fx,cx 乘 sx；fy,cy 乘 sy
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        Ks.append(torch.from_numpy(K).float())
        Rs.append(torch.from_numpy(Quaternion(calib["rotation"]).rotation_matrix).float())
        ts.append(torch.tensor(calib["translation"], dtype=torch.float32))

    images = torch.stack(imgs)       # (V,3,H,W)
    return images, torch.stack(Ks), torch.stack(Rs), torch.stack(ts)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuscenes.nuscenes import NuScenes

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--out", type=str, default="outputs/bev_coverage.png")
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] device={dev}；加载 devkit...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    sample = nusc.sample[args.sample_idx if args.sample_idx is not None
                         else len(nusc.sample) // 2]

    # 1) 读真实样本
    images, K, R, t = load_real_sample(nusc, sample)
    images, K, R, t = images.to(dev), K.to(dev), R.to(dev), t.to(dev)
    print(f"[verify] images {tuple(images.shape)}  K {tuple(K.shape)}")

    # 2) 图像 backbone 提特征
    bb = ImageBackbone(pretrained=True, freeze=True).to(dev).eval()
    with torch.no_grad():
        feats = bb(images.unsqueeze(0))        # (1,V,512,Hf,Wf)
    _, V, Cf, Hf, Wf = feats.shape
    print(f"[verify] 图像特征 {tuple(feats.shape)}")

    # 3) 用真实内外参算采样网格（投影几何的张量版）
    xs = torch.linspace(-BEV_RANGE, BEV_RANGE, BEV_SIZE, device=dev)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    bev_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)   # (Q,2)
    heights = torch.tensor(HEIGHTS, device=dev)
    sample_uv, sample_valid = build_sampling_grid(
        bev_xy, heights, K, R, t, (Hf, Wf), (IMG_H, IMG_W))
    cov = sample_valid.float().mean().item()
    cell_cov = sample_valid.any(dim=(1, 2)).float().mean().item()   # 至少一个采样点可见的格子占比
    print(f"[verify] 可见采样点占比={cov:.3f}  被覆盖的 BEV 格子占比={cell_cov:.3f}")

    # 4) cross-attention 融合
    enc = BEVCrossAttention(d_model=128, feat_channels=Cf, bev_size=BEV_SIZE,
                            n_heights=len(HEIGHTS), n_cams=V).to(dev).eval()
    with torch.no_grad():
        bev = enc(feats, sample_uv.unsqueeze(0), sample_valid.unsqueeze(0))  # (1,128,S,S)
    print(f"[verify] BEV 特征 {tuple(bev.shape)}")

    # 5) 画两张图：各相机覆盖的 BEV 格子 + BEV 特征能量图
    coverage = sample_valid.any(dim=2)                 # (Q,V) 每格哪些相机可见
    cover_map = coverage.any(dim=1).reshape(BEV_SIZE, BEV_SIZE).cpu().numpy()
    energy = bev[0].pow(2).sum(0).sqrt().reshape(BEV_SIZE, BEV_SIZE).cpu().numpy()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    a1.imshow(cover_map.T, origin="lower", cmap="gray",
              extent=[-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE])
    a1.plot(0, 0, "r^", ms=10)
    a1.set_title("BEV cells covered by any camera (white=covered)")
    a1.set_xlabel("x forward (m)"); a1.set_ylabel("y left (m)")
    im = a2.imshow(energy.T, origin="lower", cmap="viridis",
                   extent=[-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE])
    a2.plot(0, 0, "r^", ms=10)
    a2.set_title("BEV feature energy (|feat| per cell)")
    a2.set_xlabel("x forward (m)"); a2.set_ylabel("y left (m)")
    fig.colorbar(im, ax=a2, fraction=0.046)
    fig.suptitle(f"Image->BEV end-to-end on real sample {sample['token'][:8]}  "
                 f"(cell coverage={cell_cov:.1%})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[verify] 已保存覆盖/能量图到 {args.out}")
    print("[verify] 核对：覆盖图应呈 6 相机环视的扇形拼合；能量图在覆盖区非零、"
          "无覆盖区(通常是远处角落)为零。")


if __name__ == "__main__":
    main()
