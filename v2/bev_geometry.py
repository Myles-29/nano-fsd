"""v2 第 3a 步：BEV 相机投影几何（先验证几何，再上 cross-attention）。

提供把 BEV 网格点（ego 系，带高度锚点）投影到 6 路相机图像的核心函数，以及一个
单样本可视化：把 BEV 网格点按投影落点画到 6 张原图上，人眼确认几何正确。

几何约定（务必核对）:
  - ego 系: x 向前, y 向左, z 向上, 原点在自车。
  - calibrated_sensor 给的是 cam->ego 的 (R=rotation 四元数, t=translation)。
    即 p_ego = R @ p_cam + t。因此 ego->cam 为 p_cam = R^T @ (p_ego - t)。
  - 内参 K(3x3): pixel_homo = K @ p_cam, 再除以深度 z 得像素 (u, v)。
  - 可见性: p_cam 的 z>0（相机前方）且 (u,v) 落在图像内。
"""

import argparse
import os

import numpy as np

DATAROOT = "/home/miaozijian/life/mzj_planning/data/v1.0-trainval/v1.0-trainval_meta"
BLOB_ROOT = "/home/miaozijian/life/mzj_planning/data/v1.0-trainval"
VERSION = "v1.0-trainval"

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# BEV 配置（对齐 UniAD 几何、规模适配 24G）
BEV_RANGE = 50.0        # ego 周围 ±50m
BEV_SIZE = 100          # 100x100 网格
HEIGHTS = [-1.0, 0.0, 1.0, 2.0]   # 多高度锚点（BEVFormer 风格的 pillar）


def bev_grid_centers(bev_range: float = BEV_RANGE, bev_size: int = BEV_SIZE) -> np.ndarray:
    """返回 BEV 网格中心的 ego 系 (x, y) 坐标，形状 (bev_size, bev_size, 2)。

    行索引对应 x（向前），列索引对应 y（向左）；范围 [-bev_range, bev_range]。
    """
    xs = np.linspace(-bev_range, bev_range, bev_size)
    ys = np.linspace(-bev_range, bev_range, bev_size)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")       # (S, S)
    return np.stack([gx, gy], axis=-1)                # (S, S, 2)


def project_ego_to_cam(points_ego: np.ndarray, K: np.ndarray,
                       cam_to_ego_R: np.ndarray, cam_to_ego_t: np.ndarray,
                       img_w: int, img_h: int):
    """把 ego 系 3D 点 (M,3) 投影到某相机像素，返回 (uv (M,2), valid (M,) bool)。

    cam_to_ego_R/t: calibrated_sensor 的外参（cam->ego）。内部取逆做 ego->cam。
    valid: 点在相机前方(z>0) 且像素落在图像内。
    """
    pts = np.asarray(points_ego, dtype=np.float64)         # (M, 3)
    # ego -> cam: p_cam = R^T (p_ego - t)
    p_cam = (pts - cam_to_ego_t[None, :]) @ cam_to_ego_R   # (M,3)，等价 (R^T @ v)
    z = p_cam[:, 2]
    in_front = z > 1e-3
    # K @ p_cam，再除以 z
    uvw = p_cam @ K.T                                      # (M,3)
    uv = np.full((pts.shape[0], 2), -1.0)
    safe = in_front & (np.abs(uvw[:, 2]) > 1e-6)
    uv[safe] = uvw[safe, :2] / uvw[safe, 2:3]
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    valid = in_front & in_img
    return uv, valid


def _resolve_image_path(filename: str) -> str | None:
    for i in range(1, 11):
        p = os.path.join(BLOB_ROOT, f"v1.0-trainval{i:02d}_blobs", filename)
        if os.path.exists(p):
            return p
    return None


# 3D 框 8 个角点的连线（12 条棱），用于画立体框
_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),      # 顶面
              (4, 5), (5, 6), (6, 7), (7, 4),      # 底面
              (0, 4), (1, 5), (2, 6), (3, 7)]      # 竖棱


def _ego_from_global_matrix(ego_pose) -> np.ndarray:
    """global->ego 的 4x4 齐次矩阵（ego_pose 给的是 ego->global，这里取逆）。"""
    from pyquaternion import Quaternion

    m = np.eye(4)
    m[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
    m[:3, 3] = np.array(ego_pose["translation"])
    return np.linalg.inv(m)


def _draw_one_sample(nusc, sample, out_png: str) -> None:
    """把一个 sample 的 3D 框投到 6 图并存图（devkit 已在外部加载好，复用）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuscenes.utils.data_classes import Box
    from PIL import Image
    from pyquaternion import Quaternion

    ego = nusc.get("ego_pose",
                   nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
    ego_from_global = _ego_from_global_matrix(ego)

    boxes_ego, ground_zs = [], []
    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        homo = np.concatenate([box.corners().T, np.ones((8, 1))], axis=1)
        corners_ego = (homo @ ego_from_global.T)[:, :3]
        boxes_ego.append(corners_ego)
        ground_zs.extend(np.sort(corners_ego[:, 2])[:4].tolist())

    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    for ax, cam in zip(axes.ravel(), CAMS):
        sd = nusc.get("sample_data", sample["data"][cam])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(calib["camera_intrinsic"], dtype=np.float64)
        R = Quaternion(calib["rotation"]).rotation_matrix
        t = np.array(calib["translation"], dtype=np.float64)
        W, H = sd["width"], sd["height"]
        ax.imshow(Image.open(_resolve_image_path(sd["filename"])))
        n_drawn = 0
        for corners_ego in boxes_ego:
            uv, valid = project_ego_to_cam(corners_ego, K, R, t, W, H)
            if valid.sum() < 8:
                continue
            n_drawn += 1
            for a, b in _BOX_EDGES:
                ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], c="lime", lw=1.2)
        ax.set_title(f"{cam}  boxes={n_drawn}")
        ax.axis("off")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    ground = float(np.median(ground_zs)) if ground_zs else float("nan")
    fig.suptitle(f"3D GT boxes -> 6 cameras — sample {sample['token'][:8]}  "
                 f"(boxes should wrap real cars/people).  ground z(ego) ~= {ground:.2f} m")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[bev] {out_png}  (boxes={len(boxes_ego)}, ground z≈{ground:.3f} m)")


def visualize_boxes_many(num: int, out_dir: str, seed: int = 0) -> None:
    """均匀抽 num 个样本，各画一张 3D 框投影图，便于批量确认几何。"""
    from nuscenes.nuscenes import NuScenes

    print(f"[bev] 加载 devkit（{VERSION}）...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    n = len(nusc.sample)
    # 均匀间隔取样（跨不同场景），避免都挑到相邻帧
    idxs = np.linspace(0, n - 1, num).astype(int).tolist()
    print(f"[bev] 从 {n} 个样本中均匀取 {num} 个: {idxs}")
    for k, si in enumerate(idxs):
        _draw_one_sample(nusc, nusc.sample[si], os.path.join(out_dir, f"box_check_{k:02d}_s{si}.png"))
    print(f"[bev] 共 {num} 张已保存到 {out_dir}/  —— 核对绿框是否套住真实车/人。")


def visualize_boxes(sample_idx: int | None, out_png: str) -> None:
    """把当前帧 agent 的 3D 标注框投影到 6 图并画框线；同时统计框底 z 得地面高度。

    几何链路：box(global) --ego_pose逆--> ego --外参逆--> cam --内参--> 像素。
    判定：框线套住图像里真实的车/人 => 投影几何全对。地面高度 = 框底角点 z 的中位数。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuscenes.nuscenes import NuScenes
    from PIL import Image
    from pyquaternion import Quaternion

    print(f"[bev] 加载 devkit（{VERSION}）...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    if sample_idx is not None:
        sample = nusc.sample[sample_idx]
    else:
        scene = nusc.scene[0]
        tok = scene["first_sample_token"]
        chain = []
        while tok:
            chain.append(tok)
            tok = nusc.get("sample", tok)["next"]
        sample = nusc.get("sample", chain[len(chain) // 2])

    ego = nusc.get("ego_pose",
                   nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
    ego_from_global = _ego_from_global_matrix(ego)

    # 预先把所有标注框的 8 角点转到 ego 系（corners 用 devkit Box 保证正确）
    from nuscenes.utils.data_classes import Box

    boxes_ego = []        # 每个元素: (8,3) ego 系角点
    ground_zs = []        # 框底 4 角的 z，用于估地面高度
    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        corners_global = box.corners().T                    # (8,3) global
        homo = np.concatenate([corners_global, np.ones((8, 1))], axis=1)
        corners_ego = (homo @ ego_from_global.T)[:, :3]     # (8,3) ego
        boxes_ego.append(corners_ego)
        # Box.corners(): 索引 2,3,7,6 为底面（z 最小的 4 个角）；稳妥起见直接取 z 最小的 4 个
        zs = np.sort(corners_ego[:, 2])[:4]
        ground_zs.extend(zs.tolist())

    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    for ax, cam in zip(axes.ravel(), CAMS):
        sd = nusc.get("sample_data", sample["data"][cam])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(calib["camera_intrinsic"], dtype=np.float64)
        R = Quaternion(calib["rotation"]).rotation_matrix        # cam->ego
        t = np.array(calib["translation"], dtype=np.float64)
        W, H = sd["width"], sd["height"]

        ax.imshow(Image.open(_resolve_image_path(sd["filename"])))
        n_drawn = 0
        for corners_ego in boxes_ego:
            uv, valid = project_ego_to_cam(corners_ego, K, R, t, W, H)
            if valid.sum() < 8:          # 要求 8 角全部可见，避免半截框误导
                continue
            n_drawn += 1
            for a, b in _BOX_EDGES:
                ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]],
                        c="lime", lw=1.2)
        ax.set_title(f"{cam}  boxes={n_drawn}")
        ax.axis("off")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    ground = float(np.median(ground_zs)) if ground_zs else float("nan")
    fig.suptitle(f"3D GT boxes projected to 6 cameras — sample {sample['token'][:8]}  "
                 f"(boxes should wrap real cars/people).  ground z(ego) ~= {ground:.2f} m")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[bev] 已保存 3D 框投影验证到 {out_png}")
    print(f"[bev] 当前帧标注框数={len(boxes_ego)}；估算地面高度(ego 系 z 中位数)≈ {ground:.3f} m")
    print("[bev] 核对：绿色框线是否套住图像里真实的车/人。套住=投影几何正确。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="单样本模式：指定 sample 序号")
    parser.add_argument("--num", type=int, default=None,
                        help="多样本模式：均匀抽 N 个样本各出一张图")
    parser.add_argument("--out", type=str, default="outputs/bev_box_check.png")
    parser.add_argument("--out-dir", type=str, default="outputs/box_checks")
    args = parser.parse_args()
    if args.num:
        visualize_boxes_many(args.num, args.out_dir)
    else:
        visualize_boxes(args.sample_idx, args.out)
