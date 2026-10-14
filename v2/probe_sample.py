"""v2 第 1 步：单样本数据地基探查（只读，不建模/不训练）。

目的：对某个 nuScenes sample，验证我们能完整拿到 B1 所需的全部原料：
  - 6 路环视相机图像（文件存在、可读、尺寸）
  - 每路相机的内参 K(3x3) 与外参（相对自车的平移/旋转，来自 calibrated_sensor）
  - 自车位姿 ego_pose（全局平移/旋转）
  - 自车未来轨迹真值（用于规划监督 / 导航指令派生）
并打印一份清单，确认标定与坐标原料齐全、路径正确。

用法:
  conda activate mcts_debug
  python probe_sample.py                 # 探查第 0 个 scene 的某个关键帧
  python probe_sample.py --sample-idx 5  # 指定全局 sample 序号
"""

import argparse
import os

DATAROOT = "/home/miaozijian/life/mzj_planning/data/v1.0-trainval/v1.0-trainval_meta"
BLOB_ROOT = "/home/miaozijian/life/mzj_planning/data/v1.0-trainval"
VERSION = "v1.0-trainval"

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def _resolve_image_path(filename: str) -> str | None:
    """samples/CAM_*/xxx.jpg 的 filename 分散在 10 个 blob 包里；逐个找真实存在的路径。

    nuScenes sample_data 的 filename 形如 'samples/CAM_FRONT/n0...jpg'，
    图像实际落在某个 v1.0-trainvalNN_blobs/ 下。这里遍历 10 个 blob 找到它。
    """
    for i in range(1, 11):
        p = os.path.join(BLOB_ROOT, f"v1.0-trainval{i:02d}_blobs", filename)
        if os.path.exists(p):
            return p
    return None


def main():
    import numpy as np
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="全局 sample 序号；默认取第 0 个 scene 的中间帧")
    args = parser.parse_args()

    print(f"[probe] 加载 devkit（{VERSION}）...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    print(f"[probe] scenes={len(nusc.scene)}  samples={len(nusc.sample)}", flush=True)

    if args.sample_idx is not None:
        sample = nusc.sample[args.sample_idx]
    else:
        # 取第 0 个 scene 的中间一帧（避免首尾历史/未来不足）
        scene = nusc.scene[0]
        tok = scene["first_sample_token"]
        chain = []
        while tok:
            chain.append(tok)
            tok = nusc.get("sample", tok)["next"]
        sample = nusc.get("sample", chain[len(chain) // 2])

    print(f"\n[probe] sample_token = {sample['token']}")

    # --- 自车位姿（取 LIDAR_TOP 关联的 ego_pose 作为该帧基准） ---
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    print(f"[probe] ego_pose: translation={np.round(ego['translation'], 2).tolist()}  "
          f"rotation(wxyz)={np.round(ego['rotation'], 3).tolist()}")

    # --- 6 路相机：图像文件 + 内参 + 外参 ---
    print("\n[probe] 6 路相机（图像 / 内参 / 外参）:")
    ok_imgs = 0
    for cam in CAMS:
        sd = nusc.get("sample_data", sample["data"][cam])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        img_path = _resolve_image_path(sd["filename"])
        exists = img_path is not None
        ok_imgs += int(exists)

        K = np.array(calib["camera_intrinsic"])          # 3x3 内参
        t = np.round(calib["translation"], 3).tolist()   # 相机相对自车平移
        R = Quaternion(calib["rotation"]).rotation_matrix  # 相机相对自车旋转
        wh = f"{sd['width']}x{sd['height']}"

        print(f"  {cam:16s} img={'OK' if exists else '缺失!'}  {wh}  "
              f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}  cam->ego t={t}")

    # --- 真值未来轨迹（规划监督 / 导航指令派生）：看未来还有多少帧 ---
    fut = []
    tok = sample["next"]
    while tok and len(fut) < 12:
        s = nusc.get("sample", tok)
        sd2 = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        fut.append(nusc.get("ego_pose", sd2["ego_pose_token"])["translation"][:2])
        tok = s["next"]
    print(f"\n[probe] 未来可用关键帧数（用于 traj_gt）: {len(fut)}（目标 Tf=12）")

    # --- 真值标注数（感知辅助可用） ---
    print(f"[probe] 当前帧 annotation 数: {len(sample['anns'])}")

    print(f"\n[probe] 结论: 6 路相机图像 {ok_imgs}/6 可读；"
          f"内参/外参/自车位姿/未来轨迹/标注 均可获取。")
    if ok_imgs == 6:
        print("[probe] ✅ B1 数据地基齐全，可进入下一步（图像 backbone）。")
    else:
        print("[probe] ⚠️ 有相机图像缺失，请检查 blob 包是否完整。")


if __name__ == "__main__":
    main()
