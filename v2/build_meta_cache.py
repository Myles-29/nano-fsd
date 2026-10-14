"""v2 台阶2-第1步：构建"元数据缓存"（不含图像像素）。

遍历样本，把每个样本的轻量信息存成一个 .pt，供 Dataset 快速索引：
  - 6 路图像的文件绝对路径（图像在线读，不缓存像素）
  - 6 路相机的缩放后内参 K、外参 cam->ego 的 R/t
  - 真值未来轨迹（ego 系，米），以及有效步 mask

不缓存 sample_uv（太大，每样本 ~2MB）；它由相机参数在训练循环里 GPU 上现算，很快。
devkit 加载一次（~20s），之后训练完全不碰 devkit。

用法:
  python build_meta_cache.py --num 2000 --out cache/meta_2000.pt   # 小子集
  python build_meta_cache.py --out cache/meta_full.pt              # 全量
"""

import argparse
import os

import numpy as np
import torch

from bev_geometry import CAMS, DATAROOT, VERSION, _resolve_image_path

ORIG_H, ORIG_W = 900, 1600
IMG_H, IMG_W = 224, 400
FUT_STEPS = 12


def future_traj_ego(nusc, sample, fut_steps):
    """未来 fut_steps 帧自车轨迹转到当前帧 ego 系（米），返回 (traj (Tf,2), mask (Tf,))。"""
    from pyquaternion import Quaternion

    def pose(s):
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        return nusc.get("ego_pose", sd["ego_pose_token"])

    cur = pose(sample)
    m = np.eye(4)
    m[:3, :3] = Quaternion(cur["rotation"]).rotation_matrix
    m[:3, 3] = np.array(cur["translation"])
    g2e = np.linalg.inv(m)

    traj = np.zeros((fut_steps, 2), dtype=np.float32)
    mask = np.zeros(fut_steps, dtype=bool)
    tok, k = sample["next"], 0
    while tok and k < fut_steps:
        s = nusc.get("sample", tok)
        p = np.array(pose(s)["translation"] + [1.0])
        traj[k] = (g2e @ p)[:2]
        mask[k] = True
        tok, k = s["next"], k + 1
    return traj, mask


# BEV 占据图配置（和模型 BEV 对齐）
BEV_RANGE, BEV_SIZE = 50.0, 100
# 只给动态交通参与者生成占据（车/人/骑行者），和 v1 的类别过滤一致
_DYNAMIC_PREFIX = ("vehicle", "human")


_MAP_CACHE = {}       # city -> NuScenesMap，避免每样本重载


def _get_map(city: str):
    """按城市缓存 NuScenesMap（首次加载慢，之后复用）。"""
    from nuscenes.map_expansion.map_api import NuScenesMap
    if city not in _MAP_CACHE:
        print(f"[meta]   加载地图 {city} ...", flush=True)
        _MAP_CACHE[city] = NuScenesMap(dataroot=DATAROOT, map_name=city)
    return _MAP_CACHE[city]


def build_drivable_bev(nusc, sample, city: str) -> np.ndarray:
    """生成以自车为中心、ego 对齐的"可行驶区"BEV mask (BEV_SIZE,BEV_SIZE) uint8。

    用 map API 的 get_map_mask：以自车全局 (x,y,yaw) 为中心取一个
    2R×2R 的 patch，栅格化 drivable_area 图层；再转成"x前y左"的 ego 朝向。
    """
    from pyquaternion import Quaternion

    ego = nusc.get("ego_pose",
                   nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
    ex, ey = ego["translation"][0], ego["translation"][1]
    yaw = Quaternion(ego["rotation"]).yaw_pitch_roll[0]      # 全局 yaw（弧度）
    yaw_deg = np.degrees(yaw)

    nmap = _get_map(city)
    # patch_box=(center_x, center_y, height, width)；patch_angle=自车 yaw（经诊断确认）
    patch_box = (ex, ey, 2 * BEV_RANGE, 2 * BEV_RANGE)
    mask = nmap.get_map_mask(patch_box, yaw_deg, ["drivable_area"],
                             canvas_size=(BEV_SIZE, BEV_SIZE))[0]
    # 诊断(diag_map_angle2)确认：转置 T 后，可行驶区与"行=x前,列=y左"的 ego BEV 对齐
    # （GT 轨迹/占据落在灰色路面内）。固化此校正。
    drivable = mask.T.astype(np.uint8)                       # (BEV_SIZE,BEV_SIZE)
    return drivable


def build_occupancy(nusc, sample) -> np.ndarray:
    """当前帧 agent 的 3D 框投到 BEV 俯视 100x100 栅格，框覆盖的格子=1。

    流程：框(global) --当前帧 ego_pose 逆--> ego 系 -> 取框底面 4 角 (x,y) ->
    栅格化（框的旋转矩形内的格子标 1）。返回 (BEV_SIZE, BEV_SIZE) float32。
    """
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    ego = nusc.get("ego_pose",
                   nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
    m = np.eye(4)
    m[:3, :3] = Quaternion(ego["rotation"]).rotation_matrix
    m[:3, 3] = np.array(ego["translation"])
    g2e = np.linalg.inv(m)

    occ = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    # BEV 栅格坐标：行=x(前)、列=y(左)，范围 [-R,R]
    cell = (2 * BEV_RANGE) / BEV_SIZE

    def to_ij(x, y):
        i = int((x + BEV_RANGE) / cell)
        j = int((y + BEV_RANGE) / cell)
        return i, j

    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        if not ann["category_name"].startswith(_DYNAMIC_PREFIX):
            continue
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        corners_g = box.corners().T                       # (8,3) global
        homo = np.concatenate([corners_g, np.ones((8, 1))], 1)
        corners_e = (homo @ g2e.T)[:, :3]                 # (8,3) ego
        # 底面 4 角的 (x,y)（z 最小的 4 个）
        order = np.argsort(corners_e[:, 2])[:4]
        poly = corners_e[order][:, :2]                    # (4,2) ego 系 (x,y)
        # 用框的 AABB 包围盒遍历候选格子，再判点是否在旋转矩形内
        xmin, ymin = poly.min(0); xmax, ymax = poly.max(0)
        i0, j0 = to_ij(xmin, ymin); i1, j1 = to_ij(xmax, ymax)
        for i in range(max(i0, 0), min(i1 + 1, BEV_SIZE)):
            for j in range(max(j0, 0), min(j1 + 1, BEV_SIZE)):
                px = -BEV_RANGE + (i + 0.5) * cell
                py = -BEV_RANGE + (j + 0.5) * cell
                if _point_in_quad(px, py, poly):
                    occ[i, j] = 1.0
    return occ


def _point_in_quad(px, py, quad) -> bool:
    """点 (px,py) 是否在凸四边形 quad(4,2) 内（叉积同号法）。"""
    # 按角度排序成凸多边形顺序
    c = quad.mean(0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]
    sign = None
    for k in range(4):
        ax, ay = q[k]
        bx, by = q[(k + 1) % 4]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def main():
    from nuscenes.nuscenes import NuScenes

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=None, help="取前 N 个样本；默认全量")
    parser.add_argument("--out", type=str, default="cache/meta_full.pt")
    args = parser.parse_args()

    print(f"[meta] 加载 devkit（{VERSION}）...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    sx, sy = IMG_W / ORIG_W, IMG_H / ORIG_H

    n_total = len(nusc.sample)
    n = n_total if args.num is None else min(args.num, n_total)
    print(f"[meta] 处理 {n}/{n_total} 个样本...", flush=True)

    recs = []
    skipped = 0
    for i in range(n):
        sample = nusc.sample[i]
        # 只保留未来至少 1 帧的样本（能提供规划监督）
        traj, mask = future_traj_ego(nusc, sample, FUT_STEPS)
        if not mask.any():
            skipped += 1
            continue

        img_paths, Ks, Rs, ts = [], [], [], []
        ok = True
        for cam in CAMS:
            sd = nusc.get("sample_data", sample["data"][cam])
            calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            p = _resolve_image_path(sd["filename"])
            if p is None:
                ok = False
                break
            img_paths.append(p)
            from pyquaternion import Quaternion
            K = np.array(calib["camera_intrinsic"], dtype=np.float32)
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
            Ks.append(K)
            Rs.append(Quaternion(calib["rotation"]).rotation_matrix.astype(np.float32))
            ts.append(np.array(calib["translation"], dtype=np.float32))
        if not ok:
            skipped += 1
            continue

        # 导航指令（UniAD 风格，从真值轨迹推）：看末端有效点横向 y（ego 系 y 向左）
        #   y > +thr -> 左转(0)；y < -thr -> 右转(2)；否则 -> 直行(1)
        last = np.where(mask)[0].max()
        y_end = float(traj[last, 1])
        _CMD_THR = 4.0
        command = 0 if y_end > _CMD_THR else (2 if y_end < -_CMD_THR else 1)

        occ = build_occupancy(nusc, sample)               # (100,100) float32
        # 当前帧所属城市（用于取对应地图）
        log = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])
        city = log["location"]
        try:
            drivable = build_drivable_bev(nusc, sample, city)
        except Exception as e:  # noqa: BLE001
            print(f"[meta]   地图生成失败({city}): {e}，该样本 drivable 置零")
            drivable = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        recs.append({
            "token": sample["token"],
            "img_paths": img_paths,                       # list[str] 长度 6
            "K": np.stack(Ks),                            # (6,3,3)
            "R": np.stack(Rs),                            # (6,3,3)
            "t": np.stack(ts),                            # (6,3)
            "traj": traj,                                 # (Tf,2) 米
            "traj_mask": mask,                            # (Tf,)
            "occ": occ.astype(np.uint8),                  # (100,100) BEV 占据真值
            "drivable": drivable.astype(np.uint8),        # (100,100) 可行驶区真值
            "command": int(command),                      # 导航指令 0=左 1=直 2=右
        })
        if (i + 1) % 500 == 0:
            print(f"[meta]   {i+1}/{n}  已收集 {len(recs)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"records": recs, "img_hw": (IMG_H, IMG_W),
                "orig_hw": (ORIG_H, ORIG_W), "fut_steps": FUT_STEPS}, args.out)
    print(f"[meta] 完成：有效 {len(recs)}，跳过 {skipped} -> {args.out}")


if __name__ == "__main__":
    main()
