"""v2 单样本过拟合 sanity check：验证"图像->BEV->轨迹"整条链能训练。

取一个真实样本（6 路图 + 真值未来轨迹 + 相机几何），反复训练几百步，看规划 loss
能否被压到接近 0。能 => 梯度链路（图像->CNN->投影->注意力->规划头）通畅、可训练。
最后画 BEV 俯视图：预测轨迹(红) vs 真值轨迹(绿)，以及训练后的 BEV 特征能量。

坐标/单位：ego 系（x 前 y 左），轨迹除以 SCENE_SCALE=50 标准化（与 v1 一致）。
"""

import argparse
import os

import numpy as np
import torch

from bev_encoder import build_sampling_grid
from bev_geometry import DATAROOT, VERSION
from model import NanoFSDv2
from verify_bev_real import (BEV_RANGE, BEV_SIZE, HEIGHTS, load_real_sample)

SCENE_SCALE = 50.0
FUT_STEPS = 12


def get_future_traj_ego(nusc, sample, fut_steps: int) -> np.ndarray:
    """提取自车未来 fut_steps 帧轨迹，转到当前帧 ego 系（x前y左，米）。不足则 pad。"""
    from pyquaternion import Quaternion

    def ego_pose_of(s):
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        return nusc.get("ego_pose", sd["ego_pose_token"])

    cur = ego_pose_of(sample)
    # 当前帧 ego->global 的逆（global->ego）
    m = np.eye(4)
    m[:3, :3] = Quaternion(cur["rotation"]).rotation_matrix
    m[:3, 3] = np.array(cur["translation"])
    g2e = np.linalg.inv(m)

    traj = np.zeros((fut_steps, 2), dtype=np.float32)
    tok = sample["next"]
    k = 0
    while tok and k < fut_steps:
        s = nusc.get("sample", tok)
        p = np.array(ego_pose_of(s)["translation"] + [1.0])    # (4,) global homo
        pe = (g2e @ p)[:2]
        traj[k] = pe
        tok = s["next"]
        k += 1
    return traj      # (fut_steps, 2) 米


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuscenes.nuscenes import NuScenes

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="outputs/overfit_one.png")
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[overfit] device={dev}；加载 devkit...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    sample = nusc.sample[args.sample_idx if args.sample_idx is not None
                         else len(nusc.sample) // 2]

    # --- 固定输入：6 路图 + 相机几何（训练中不变）---
    images, K, R, t = load_real_sample(nusc, sample)
    images, K, R, t = images.to(dev), K.to(dev), R.to(dev), t.to(dev)
    xs = torch.linspace(-BEV_RANGE, BEV_RANGE, BEV_SIZE, device=dev)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    bev_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    heights = torch.tensor(HEIGHTS, device=dev)
    sample_uv, sample_valid = build_sampling_grid(
        bev_xy, heights, K, R, t, (7, 13), (224, 400))
    images = images.unsqueeze(0)
    sample_uv = sample_uv.unsqueeze(0)
    sample_valid = sample_valid.unsqueeze(0)

    # --- 真值轨迹（标准化）---
    traj_m = get_future_traj_ego(nusc, sample, FUT_STEPS)      # (Tf,2) 米
    traj_gt = torch.from_numpy(traj_m / SCENE_SCALE).float().to(dev).unsqueeze(0)
    print(f"[overfit] 真值轨迹(米) 末点={traj_m[-1].round(2).tolist()}  "
          f"（终点离自车 {np.linalg.norm(traj_m[-1]):.1f} m）")

    # --- 模型 + 优化器（只训 BEV 编码器 + 规划头；backbone 冻结）---
    model = NanoFSDv2(bev_size=BEV_SIZE, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, pretrained=True, freeze_backbone=True).to(dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    print(f"[overfit] 开始单样本过拟合 {args.steps} 步...")
    losses = []
    for step in range(args.steps):
        traj_pred = model(images, sample_uv, sample_valid)["traj"]  # (1,Tf,2)
        loss = ((traj_pred - traj_gt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0 or step == 0:
            # 换算成米制 ADE，直观
            ade_m = ((traj_pred - traj_gt).detach() * SCENE_SCALE
                     ).pow(2).sum(-1).sqrt().mean().item()
            print(f"  step {step+1:>4}/{args.steps}  loss={loss.item():.6f}  "
                  f"ADE={ade_m:.3f} m", flush=True)

    # --- 画图：轨迹对比 + 训练后 BEV 能量 ---
    model.eval()
    with torch.no_grad():
        _out = model(images, sample_uv, sample_valid, return_bev=True)
        traj_pred, bev = _out["traj"], _out["bev"]
    pred = (traj_pred[0].cpu().numpy()) * SCENE_SCALE
    gt = traj_m
    energy = bev[0].pow(2).sum(0).sqrt().cpu().numpy()

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 6))
    a1.plot(losses); a1.set_yscale("log"); a1.set_title("overfit loss (log)")
    a1.set_xlabel("step"); a1.set_ylabel("MSE")

    a2.plot(gt[:, 0], gt[:, 1], "g-o", label="GT")
    a2.plot(pred[:, 0], pred[:, 1], "r-x", label="Pred")
    a2.plot(0, 0, "b^", ms=12, label="Ego")
    a2.set_aspect("equal"); a2.legend(); a2.grid(alpha=0.3)
    a2.set_title("trajectory: pred vs GT (meters)")
    a2.set_xlabel("x forward (m)"); a2.set_ylabel("y left (m)")

    im = a3.imshow(energy.T, origin="lower", cmap="viridis",
                   extent=[-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE])
    a3.plot(0, 0, "r^", ms=10); a3.set_title("BEV feature energy (after overfit)")
    fig.colorbar(im, ax=a3, fraction=0.046)

    final_ade = np.linalg.norm(pred - gt, axis=1).mean()
    fig.suptitle(f"v2 single-sample overfit — sample {sample['token'][:8]}  "
                 f"final ADE={final_ade:.3f} m")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[overfit] 最终 ADE={final_ade:.3f} m；已保存 {args.out}")
    print("[overfit] 判定：loss 显著下降、轨迹两线基本重合 => 整条链梯度通、可训练。")


if __name__ == "__main__":
    main()
