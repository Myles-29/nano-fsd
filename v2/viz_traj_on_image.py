"""把预测/真值轨迹投影回相机图像：直观看"模型想开的线"画在真实道路上。

复用单样本过拟合（outputs 已验证整条链可训练）：训练后得到预测轨迹，连同真值轨迹
一起用相机投影几何（project_ego_to_cam，与 BEV 同一套）投到图像平面。
主图：前视 CAM_FRONT 大图；附图：6 路小图。轨迹点取地面高度 z=0（贴地）。

红=预测(Pred)，绿=真值(GT)，蓝三角=自车。相邻轨迹点间插值加密使曲线平滑。
"""

import argparse
import os

import numpy as np
import torch

from bev_encoder import build_sampling_grid
from bev_geometry import CAMS, DATAROOT, VERSION, _resolve_image_path, project_ego_to_cam
from model import NanoFSDv2
from overfit_one import SCENE_SCALE, FUT_STEPS, get_future_traj_ego
from verify_bev_real import BEV_RANGE, BEV_SIZE, HEIGHTS, load_real_sample

ORIG_H, ORIG_W = 900, 1600       # 投到原图尺寸画（清晰）


def densify(traj_xy: np.ndarray, k: int = 6) -> np.ndarray:
    """相邻点间线性插值加密，(T,2) -> (~T*k, 2)，让投影曲线平滑。"""
    out = []
    for i in range(len(traj_xy) - 1):
        for a in np.linspace(0, 1, k, endpoint=False):
            out.append(traj_xy[i] * (1 - a) + traj_xy[i + 1] * a)
    out.append(traj_xy[-1])
    return np.array(out)


def traj_to_3d(traj_xy_m: np.ndarray, z: float = 0.0) -> np.ndarray:
    """轨迹 (T,2) 米 -> ego 系 3D 点 (T,3)，z 为地面高度。"""
    t = densify(traj_xy_m)
    return np.concatenate([t, np.full((len(t), 1), z)], axis=1)


def _draw_traj_on_ax(ax, nusc, sample, cam, pred3d, gt3d):
    """在某相机图上画出 pred/gt 轨迹的可见部分。返回可见点数(pred)。"""
    from PIL import Image
    from pyquaternion import Quaternion

    sd = nusc.get("sample_data", sample["data"][cam])
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(calib["camera_intrinsic"], dtype=np.float64)   # 原图内参
    R = Quaternion(calib["rotation"]).rotation_matrix
    t = np.array(calib["translation"], dtype=np.float64)
    ax.imshow(Image.open(_resolve_image_path(sd["filename"])))

    for pts3d, color, label in [(gt3d, "lime", "GT"), (pred3d, "red", "Pred")]:
        uv, valid = project_ego_to_cam(pts3d, K, R, t, ORIG_W, ORIG_H)
        vu = uv[valid]
        if len(vu) >= 2:
            ax.plot(vu[:, 0], vu[:, 1], "-", c=color, lw=3, label=label)
        elif len(vu) == 1:
            ax.scatter(vu[:, 0], vu[:, 1], c=color, s=30, label=label)
    ax.set_xlim(0, ORIG_W); ax.set_ylim(ORIG_H, 0); ax.axis("off")
    uv0, v0 = project_ego_to_cam(pred3d, K, R, t, ORIG_W, ORIG_H)
    return int(v0.sum())


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuscenes.nuscenes import NuScenes

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out", type=str, default="outputs/traj_on_image.png")
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[traj] device={dev}；加载 devkit...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    sample = nusc.sample[args.sample_idx if args.sample_idx is not None
                         else len(nusc.sample) // 2]

    # 固定输入 + 几何
    images, K, R, t = load_real_sample(nusc, sample)
    images, K, R, t = images.to(dev), K.to(dev), R.to(dev), t.to(dev)
    xs = torch.linspace(-BEV_RANGE, BEV_RANGE, BEV_SIZE, device=dev)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    bev_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    heights = torch.tensor(HEIGHTS, device=dev)
    sample_uv, sample_valid = build_sampling_grid(bev_xy, heights, K, R, t, (7, 13), (224, 400))
    images = images.unsqueeze(0); sample_uv = sample_uv.unsqueeze(0); sample_valid = sample_valid.unsqueeze(0)

    traj_m = get_future_traj_ego(nusc, sample, FUT_STEPS)
    traj_gt = torch.from_numpy(traj_m / SCENE_SCALE).float().to(dev).unsqueeze(0)

    # 单样本过拟合（和 overfit_one 一致），得到预测轨迹
    model = NanoFSDv2(bev_size=BEV_SIZE, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, pretrained=True, freeze_backbone=True).to(dev)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    print(f"[traj] 单样本过拟合 {args.steps} 步...")
    for step in range(args.steps):
        pred = model(images, sample_uv, sample_valid)["traj"]
        loss = ((pred - traj_gt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(images, sample_uv, sample_valid)["traj"][0].cpu().numpy() * SCENE_SCALE
    print(f"[traj] 过拟合后预测末点(米)={pred[-1].round(2).tolist()}")

    pred3d = traj_to_3d(pred, z=0.0)
    gt3d = traj_to_3d(traj_m, z=0.0)

    # 主图：前视大图
    fig = plt.figure(figsize=(18, 10))
    ax_main = fig.add_subplot(2, 1, 1)
    nvis = _draw_traj_on_ax(ax_main, nusc, sample, "CAM_FRONT", pred3d, gt3d)
    ax_main.legend(loc="upper right", fontsize=12)
    ax_main.set_title(f"CAM_FRONT — predicted(red) vs GT(green) trajectory on road "
                      f"(pred visible pts={nvis})")

    # 附图：6 路小图
    for i, cam in enumerate(CAMS):
        ax = fig.add_subplot(2, 6, 7 + i)
        _draw_traj_on_ax(ax, nusc, sample, cam, pred3d, gt3d)
        ax.set_title(cam, fontsize=8)

    fig.suptitle(f"Trajectory projected onto camera images — sample {sample['token'][:8]}  "
                 f"(z=0 ground; overfit model so red≈green)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[traj] 已保存 {args.out}")
    print("[traj] 核对：轨迹应沿前方道路延伸、贴着路面；过拟合模型下红(pred)≈绿(GT)。")


if __name__ == "__main__":
    main()
