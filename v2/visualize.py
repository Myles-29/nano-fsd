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


def _draw_perception_panel(ax, sample, det, score_thr=0.3, vmap=None):
    """在综合图的「BEV 感知栏」里整合画：预测框（实线按类着色）+ GT 框（虚线）+
    他车预测未来轨迹（细实线）+ 匹配 GT 未来轨迹（细虚线）。

    复用 `_draw_box`/`_draw_future`（与 step2/step3 单独栏一致），并用 train.py 的
    `match_pred_to_gt`（训练同一套最近距离贪心匹配）把预测 query 配到 GT，保证综合图里
    预测↔GT 的未来轨迹对照与训练/单独可视化口径一致（需求 6.1、6.2）。

    det：模型 forward 返回的 dict['det']（含 box/cls/score/agent_fut），batch 维取第 0 个。
    vmap：模型 forward 返回的 dict['vmap']（可选）。不为 None 时在同一感知栏叠加高置信度
      **预测矢量地图折线**（实线按类着色），直观看静态地图元素与动态目标的空间关系；为
      None（纯旧 ckpt）时整段跳过，与原四/五栏行为一致（需求 8.1/8.5）。
    返回 (n_gt, n_keep, n_matched) 供标题展示。
    """
    from train import match_pred_to_gt  # 复用训练同一套贪心匹配

    # 预测框/类别/置信度/未来（去归一化到米）
    pred_box = det["box"][0].cpu().numpy().copy()            # (N,5) [x,y,yaw,w,l]
    pred_cls = det["cls"][0].argmax(-1).cpu().numpy()        # (N,)
    pred_score = torch.sigmoid(det["score"][0, :, 0]).cpu().numpy()  # (N,)
    pred_fut = det["agent_fut"][0].cpu().numpy()             # (N,Tf,2) 相对中心（归一化）
    pred_box_m = pred_box.copy()
    pred_box_m[:, 0:2] *= SCENE_SCALE
    pred_fut_m = pred_fut * SCENE_SCALE                      # 相对位移去归一化到米

    # GT 框/未来（去归一化到米）
    gt_box = sample["agent_box"].numpy().copy()              # (MAX_GT,5)
    gt_box_m = gt_box.copy(); gt_box_m[:, 0:2] *= SCENE_SCALE
    gt_cls = sample["agent_cls"].numpy()
    gt_mask = sample["agent_mask"].numpy()
    gt_fut = sample["agent_fut"].numpy() * SCENE_SCALE       # (MAX_GT,Tf,2) 米
    gt_fut_mask = sample["agent_fut_mask"].numpy()           # (MAX_GT,Tf)
    n_gt = int(gt_mask.sum())

    # 预测 query -> GT 匹配（坐标用归一化系的相对距离即可，与训练一致）
    pred_box_t = det["box"][0].cpu()                         # (N,5) 归一化坐标
    gt_box_t = sample["agent_box"]                           # (MAX_GT,5)
    gt_mask_t = sample["agent_mask"]
    pred_idx, gt_idx = match_pred_to_gt(pred_box_t, gt_box_t, gt_mask_t)
    gt_for_pred = {int(p): int(g) for p, g in zip(pred_idx.tolist(), gt_idx.tolist())}

    ax.plot(0, 0, "b^", ms=11, zorder=6, label="ego")

    # GT 框（虚线对照）
    seen_gt = set()
    for k in range(len(gt_mask)):
        if not gt_mask[k]:
            continue
        cls = int(gt_cls[k])
        lab = f"GT {_CLS_NAMES.get(cls, cls)}" if cls not in seen_gt else None
        seen_gt.add(cls)
        _draw_box(ax, gt_box_m[k], _CLS_COLORS.get(cls, "gray"), "--", label=lab, lw=1.2)

    # 高置信度预测框（实线）+ 其预测未来轨迹（细实线）+ 匹配 GT 未来（细虚线）
    keep = np.where(pred_score > score_thr)[0]
    seen_pred, seen_fut = set(), set()
    for k in keep:
        cls = int(pred_cls[k])
        color = _CLS_COLORS.get(cls, "gray")
        lab_box = f"Pred {_CLS_NAMES.get(cls, cls)}" if cls not in seen_pred else None
        seen_pred.add(cls)
        _draw_box(ax, pred_box_m[k], color, "-", label=lab_box, lw=1.7)
        _draw_future(ax, pred_box_m[k, 0:2], pred_fut_m[k], color, "-", lw=1.0)
        if k in gt_for_pred:
            g = gt_for_pred[k]
            vstep = gt_fut_mask[g]
            if vstep.any():
                lab_gt = "GT future" if "gt" not in seen_fut else None
                seen_fut.add("gt")
                _draw_future(ax, gt_box_m[g, 0:2], gt_fut[g][vstep], "black", "--",
                             lw=1.0, label=lab_gt)

    # 矢量地图叠加（可选）：模型含矢量地图头时叠加高置信度预测折线；纯旧 ckpt 为 None 跳过。
    if vmap is not None:
        _overlay_vmap_pred(ax, vmap, score_thr)

    ax.set_xlim(BEV_RANGE, -BEV_RANGE)   # 左(+y)在左侧
    ax.set_ylim(-BEV_RANGE, BEV_RANGE)   # 前(+x)在上
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("y left (m)"); ax.set_ylabel("x forward (m)")
    ax.legend(loc="upper right", fontsize=6)
    return n_gt, len(keep), len(gt_for_pred)


def run_visualization(ckpt, meta="cache/meta_2000.pt", num=6, seed=0,
                      val_ratio=0.2, out_dir="outputs/viz_sub2000",
                      use_planner_v2=None, score_thr=0.3):
    """加载 ckpt，对验证集前 num 个样本出**综合图**，存到 out_dir。

    综合图整合五栏（需求 6.1/6.2/6.3，任务 5.3）：
      0 占据：预测(热力) vs GT(青轮廓)
      1 可行驶区：预测(热力) vs GT(青轮廓)
      2 自车轨迹：预测(红) vs GT(绿)（BEV 俯视）
      3 CAM_FRONT：预测/真值轨迹投影回道路
      4 **BEV 感知**：预测框(实线按类着色) + GT 框(虚线) + 他车预测未来轨迹(细线)
        + 匹配 GT 未来轨迹(细虚线对照)

    只用模型没训过的**验证集**样本（val_indices 复现相同 seed/val_ratio 划分，需求 6.3）。
    可被 train.py 在训练结束后直接调用，实现"训练完自动可视化"。

    向后兼容：若 checkpoint/模型不含检测头输出（如纯 v1 旧 ckpt），第 4 栏自动跳过并
    回退为原四栏综合图，不报错。

    use_planner_v2：规划头版本（任务 4.4）。None 时优先读 checkpoint 里存的
    `use_planner_v2` 字段（多任务训练会写入），都没有则默认 True。必须与训练该
    checkpoint 时的规划头版本一致，否则 state_dict 键对不上（v1 无 planner_v2.* 键）。
    score_thr：BEV 感知栏里预测框的置信度阈值（sigmoid(score) 高于此值才画）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = NuScenesBEVDataset(meta)
    vidx = val_indices(len(ds), val_ratio, seed)
    print(f"[viz] 验证集 {len(vidx)} 个（seed={seed}）；取前 {num} 个可视化")

    state = torch.load(ckpt, map_location=dev, weights_only=True)
    if use_planner_v2 is None:
        use_planner_v2 = bool(state.get("use_planner_v2", True))
    # 矢量地图头开关：默认 False，这样旧 checkpoint（无 vector_map.* 键、也无该字段）重建
    # 时不构造该头，load_state_dict 严格匹配仍成立（加一个 ckpt 没有的头会破坏 strict 加载）。
    # 新的多任务 ckpt 保存时写入 use_vector_map=True，重建时才带上矢量地图头。
    use_vector_map = bool(state.get("use_vector_map", False))
    model = NanoFSDv2(bev_size=BEV_SIZE, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, use_planner_v2=use_planner_v2,
                      use_vector_map=use_vector_map,
                      pretrained=True, freeze_backbone=True).to(dev)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[viz] 已加载 {ckpt}  (use_planner_v2={use_planner_v2}, "
          f"use_vector_map={use_vector_map})")

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
            out = model(images, uv, valid, command=command, return_bev=True)
        pred, occ_logit, map_logit, bev = out["traj"], out["occ"], out["map"], out["bev"]
        # 检测/运动头输出（综合图第 4 栏用）；纯 v1 ckpt 可能无 det，向后兼容
        det = out.get("det")
        has_perception = det is not None and ("agent_fut" in det) and ("agent_box" in sample)
        pred_m = pred[0].cpu().numpy() * SCENE_SCALE          # (Tf,2) 米
        gt_m = rec["traj"]                                     # (Tf,2) 米
        mask = rec["traj_mask"]
        ade = np.linalg.norm((pred_m - gt_m)[mask], axis=1).mean()
        ades.append(ade)
        occ_prob = torch.sigmoid(occ_logit[0, 0]).cpu().numpy()   # (S,S) 预测占据
        map_prob = torch.sigmoid(map_logit[0, 0]).cpu().numpy()   # (S,S) 预测可行驶区
        occ_gt = rec["occ"].astype(np.float32)
        map_gt = rec.get("drivable", np.zeros_like(occ_gt)).astype(np.float32)

        n_panels = 5 if has_perception else 4
        fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6))
        a0, a1, a2, a3 = axes[0], axes[1], axes[2], axes[3]
        a4 = axes[4] if has_perception else None
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

        # 第4栏：BEV 感知（检测框 + 他车预测/GT 未来轨迹整合），仅在有检测头输出时画。
        # 若模型含矢量地图头（out["vmap"] 不为 None），在同栏叠加预测矢量地图折线；纯旧
        # ckpt 无该头则 vmap=None，_draw_perception_panel 自动跳过矢量地图叠加（需求 8.5）。
        vmap = out.get("vmap")
        if has_perception:
            n_gt, n_keep, n_matched = _draw_perception_panel(
                a4, sample, det, score_thr, vmap=vmap)
            _vmap_note = "  +vmap" if vmap is not None else ""
            a4.set_title(f"perception (BEV): pred box+future(solid) vs GT(dashed){_vmap_note}\n"
                         f"GT={n_gt}  pred={n_keep}  matched={n_matched}")

        fig.suptitle(f"v2 val sample idx={idx}  ADE={ade:.2f} m"
                     f"{'  (+perception)' if has_perception else ''}")
        png = os.path.join(out_dir, f"val_{n:02d}_idx{idx}.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] sample idx={idx}  ADE={ade:.2f} m  -> {png}")

    print(f"[viz] 平均 ADE（这 {len(ades)} 个样本）= {np.mean(ades):.3f} m")
    print(f"[viz] 共 {len(ades)} 张保存到 {out_dir}/")
    return float(np.mean(ades)) if ades else float("nan")


# ---------------------------------------------------------------------------
# 检测框可视化（任务 2.6）：高置信度预测框 + GT 框（虚线）叠加到 BEV 俯视
# ---------------------------------------------------------------------------

# 2 类配色（与 viz_agent_gt.py 一致）：0=车 vehicle，1=人 human
_CLS_COLORS = {0: "tab:orange", 1: "tab:cyan"}
_CLS_NAMES = {0: "vehicle", 1: "human"}


def _oriented_box_corners(x, y, yaw, w, l):
    """由中心 (x,y)、朝向 yaw、尺寸 (w,l) 算 ego 系下框 4 角 (4,2)。

    nuScenes size=[w,l,h]：w 为宽（沿车体 y）、l 为长（沿车体前进方向 x）；yaw=0 时车头
    朝 ego +x。与 viz_agent_gt.py 的实现保持一致，便于预测/GT 框并排对照。
    """
    dx, dy = l / 2.0, w / 2.0
    local = np.array([[+dx, -dy], [+dx, +dy], [-dx, +dy], [-dx, -dy]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return (local @ rot.T) + np.array([x, y])


def _draw_box(ax, box, color, linestyle, label=None, lw=1.5):
    """在 BEV 俯视子图上画一个朝向框（+ 车头指示线）。box=[x,y,yaw,w,l]（米，ego 系）。

    俯视图统一约定：横轴 = y 左、纵轴 = x 前。故所有点在画到坐标轴时取 (y, x) 顺序，
    即横坐标用 ego 的 y、纵坐标用 ego 的 x（与道路底图 pcolormesh 的 (Xy=y, Yx=x) 一致）。
    """
    from matplotlib.patches import Polygon
    x, y, yaw, w, l = box
    corners = _oriented_box_corners(x, y, yaw, w, l)   # (4,2) 列0=x前, 列1=y左
    # 画到画布：横轴取 y(列1)、纵轴取 x(列0)
    poly_xy = np.stack([corners[:, 1], corners[:, 0]], axis=-1)
    ax.add_patch(Polygon(poly_xy, closed=True, fill=False, edgecolor=color,
                         linewidth=lw, linestyle=linestyle, label=label, zorder=3))
    head = np.array([x, y]) + (l / 2.0) * np.array([np.cos(yaw), np.sin(yaw)])
    ax.plot([y, head[1]], [x, head[0]], linestyle, c=color, lw=0.9, zorder=4)


def run_detection_viz(ckpt, meta="cache/meta_2000_agent.pt", num=6, seed=0,
                      val_ratio=0.2, out_dir="outputs", score_thr=0.5):
    """加载检测头 ckpt，对**验证集**样本把高置信度预测框 + GT 框（虚线）叠到 BEV 俯视。

    严谨性：用与训练相同的 seed/val_ratio 复现验证划分（见 val_indices），只可视化模型
    没训过的验证样本（需求 6.3）。预测框按 sigmoid(score) > score_thr 过滤为「高置信度」
    并按类别着色（实线）；GT 框用虚线对照（需求 6.1）。保存到 out_dir/step2_det_*.png。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = NuScenesBEVDataset(meta)
    vidx = val_indices(len(ds), val_ratio, seed)
    print(f"[det-viz] 验证集 {len(vidx)} 个（seed={seed}）；取前 {num} 个可视化")

    model = NanoFSDv2(bev_size=BEV_SIZE, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, pretrained=True, freeze_backbone=True).to(dev)
    state = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[det-viz] 已加载 {ckpt}")

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for n, idx in enumerate(vidx[:num]):
        sample = ds[idx]
        images = sample["images"].unsqueeze(0).to(dev)
        K = sample["K"].unsqueeze(0); R = sample["R"].unsqueeze(0); t = sample["t"].unsqueeze(0)
        uv, valid = geometry_from_batch(K, R, t, IMG_HW, dev)
        with torch.no_grad():
            det = model(images, uv, valid, command=None)["det"]
        pred_box = det["box"][0].cpu().numpy().copy()        # (N,5) [x,y,yaw,w,l]
        pred_cls = det["cls"][0].argmax(-1).cpu().numpy()    # (N,)
        pred_score = torch.sigmoid(det["score"][0, :, 0]).cpu().numpy()  # (N,)
        # 预测框的 x,y 与 GT 同为 SCENE_SCALE 归一化 -> 乘回米；yaw/w/l 本就是弧度/米
        pred_box[:, 0:2] *= SCENE_SCALE

        gt_box = sample["agent_box"].numpy().copy()          # (MAX_GT,5)
        gt_box[:, 0:2] *= SCENE_SCALE
        gt_cls = sample["agent_cls"].numpy()
        gt_mask = sample["agent_mask"].numpy()
        n_gt = int(gt_mask.sum())

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        ax.plot(0, 0, "b^", ms=12, zorder=5, label="ego")

        # GT 框（虚线对照）
        seen_gt = set()
        for k in range(len(gt_mask)):
            if not gt_mask[k]:
                continue
            cls = int(gt_cls[k])
            lab = f"GT {_CLS_NAMES.get(cls, cls)}" if cls not in seen_gt else None
            seen_gt.add(cls)
            _draw_box(ax, gt_box[k], _CLS_COLORS.get(cls, "gray"), "--", label=lab, lw=1.3)

        # 高置信度预测框（实线，按类别着色）
        keep = np.where(pred_score > score_thr)[0]
        seen_pred = set()
        for k in keep:
            cls = int(pred_cls[k])
            lab = f"Pred {_CLS_NAMES.get(cls, cls)}" if cls not in seen_pred else None
            seen_pred.add(cls)
            _draw_box(ax, pred_box[k], _CLS_COLORS.get(cls, "gray"), "-", label=lab, lw=1.8)

        ax.set_xlim(BEV_RANGE, -BEV_RANGE)   # 左(+y)在左侧
        ax.set_ylim(-BEV_RANGE, BEV_RANGE)   # 前(+x)在上
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("y left (m)"); ax.set_ylabel("x forward (m)")
        ax.set_title(f"Detection (BEV): pred(solid, score>{score_thr}) vs GT(dashed)  "
                     f"idx={idx}  GT={n_gt}  pred={len(keep)}")
        ax.legend(loc="upper right", fontsize=7)

        png = os.path.join(out_dir, f"step2_det_{n:02d}_idx{idx}.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[det-viz] idx={idx}  GT={n_gt}  high-score pred={len(keep)}  -> {png}")
        saved.append(png)

    print(f"[det-viz] 共保存 {len(saved)} 张到 {out_dir}/")
    return saved


# ---------------------------------------------------------------------------
# 他车预测轨迹可视化（任务 3.5）：高置信度预测框 + 其预测未来轨迹（细线），GT 轨迹对照
# ---------------------------------------------------------------------------


def _draw_future(ax, center_xy, fut_rel_m, color, linestyle, lw, label=None):
    """从框中心画一条未来轨迹细线。

    center_xy: 该 agent 当前框中心 (x,y)（米，ego 系）。
    fut_rel_m: (Tf,2) 相对当前中心的未来位移（米）；加回中心得绝对未来点。
    画「中心 -> 各未来步」的折线（细线），末端点一个小标记指示终点方向。
    """
    pts = center_xy[None, :] + fut_rel_m          # (Tf,2) 绝对未来位置（列0=x前,列1=y左）
    # 俯视图：横轴=y左、纵轴=x前，故横坐标用列1(y)、纵坐标用列0(x)
    xcoord = np.concatenate([[center_xy[0]], pts[:, 0]])   # x 前（纵轴）
    ycoord = np.concatenate([[center_xy[1]], pts[:, 1]])   # y 左（横轴）
    ax.plot(ycoord, xcoord, linestyle, c=color, lw=lw, label=label, zorder=4)
    ax.plot(pts[-1, 1], pts[-1, 0], marker="o", c=color, ms=3, zorder=5)


def run_motion_viz(ckpt, meta="cache/meta_2000_agent.pt", num=6, seed=0,
                   val_ratio=0.2, out_dir="outputs", score_thr=0.3):
    """加载检测+运动 ckpt，对**验证集**样本画高置信度预测框的预测未来轨迹（细线）+ GT 对照。

    严谨性：用与训练相同的 seed/val_ratio 复现验证划分（见 val_indices），只可视化模型
    没训过的验证样本（需求 7.1）。对每个 sigmoid(score) > score_thr 的预测 query：
      - 画其预测框（实线，按类别着色）与**预测未来轨迹**（细实线，由 agent_fut 去归一化
        乘回 SCENE_SCALE 并加回框中心）；
      - 用与训练一致的最近距离贪心匹配，为该预测找到对应 GT，叠加该 GT 的**未来轨迹**
        （细虚线，仅在 agent_fut_mask 有效步）对照（需求 6.2）。
    保存到 out_dir/step3_motion_*.png，人眼确认他车轨迹方向合理。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from train import match_pred_to_gt  # 复用训练同一套贪心匹配，保证预测↔GT 配对一致

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = NuScenesBEVDataset(meta)
    vidx = val_indices(len(ds), val_ratio, seed)
    print(f"[mot-viz] 验证集 {len(vidx)} 个（seed={seed}）；取前 {num} 个可视化")

    model = NanoFSDv2(bev_size=BEV_SIZE, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, pretrained=True, freeze_backbone=True).to(dev)
    state = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[mot-viz] 已加载 {ckpt}")

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for n, idx in enumerate(vidx[:num]):
        sample = ds[idx]
        images = sample["images"].unsqueeze(0).to(dev)
        K = sample["K"].unsqueeze(0); R = sample["R"].unsqueeze(0); t = sample["t"].unsqueeze(0)
        uv, valid = geometry_from_batch(K, R, t, IMG_HW, dev)
        with torch.no_grad():
            det = model(images, uv, valid, command=None)["det"]

        # GT（张量，用于匹配）；坐标仍是 SCENE_SCALE 归一化，匹配只用 x,y 相对距离即可
        gt_box_t = sample["agent_box"].unsqueeze(0)          # (1,MAX_GT,5)
        gt_mask_t = sample["agent_mask"].unsqueeze(0)        # (1,MAX_GT)
        pred_box_t = det["box"][0].cpu()                     # (N,5) 归一化坐标
        # 用训练同一匹配：预测 query -> GT（per-sample）
        pred_idx, gt_idx = match_pred_to_gt(pred_box_t, gt_box_t[0], gt_mask_t[0])
        gt_for_pred = {int(p): int(g) for p, g in zip(pred_idx.tolist(), gt_idx.tolist())}

        # 预测框/类别/置信度（去归一化到米）
        pred_box = det["box"][0].cpu().numpy().copy()        # (N,5)
        pred_cls = det["cls"][0].argmax(-1).cpu().numpy()    # (N,)
        pred_score = torch.sigmoid(det["score"][0, :, 0]).cpu().numpy()  # (N,)
        pred_fut = det["agent_fut"][0].cpu().numpy()         # (N,Tf,2) 相对中心（归一化）
        pred_box[:, 0:2] *= SCENE_SCALE
        pred_fut = pred_fut * SCENE_SCALE                    # 相对位移去归一化到米

        # GT 框/未来轨迹（去归一化到米）
        gt_box = sample["agent_box"].numpy().copy()          # (MAX_GT,5)
        gt_box[:, 0:2] *= SCENE_SCALE
        gt_cls = sample["agent_cls"].numpy()
        gt_mask = sample["agent_mask"].numpy()
        gt_fut = sample["agent_fut"].numpy() * SCENE_SCALE   # (MAX_GT,Tf,2) 米
        gt_fut_mask = sample["agent_fut_mask"].numpy()       # (MAX_GT,Tf)
        n_gt = int(gt_mask.sum())

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(0, 0, "b^", ms=12, zorder=6, label="ego")

        keep = np.where(pred_score > score_thr)[0]
        seen_pred, seen_gt = set(), set()
        for k in keep:
            cls = int(pred_cls[k])
            color = _CLS_COLORS.get(cls, "gray")
            # 预测框（实线）
            lab_box = f"Pred {_CLS_NAMES.get(cls, cls)}" if cls not in seen_pred else None
            seen_pred.add(cls)
            _draw_box(ax, pred_box[k], color, "-", label=lab_box, lw=1.6)
            # 预测未来轨迹（细实线，从框中心出发）
            _draw_future(ax, pred_box[k, 0:2], pred_fut[k], color, "-", lw=1.0)
            # 匹配到的 GT 未来轨迹（细虚线，仅有效步）对照
            if k in gt_for_pred:
                g = gt_for_pred[k]
                vstep = gt_fut_mask[g]
                if vstep.any():
                    lab_gt = "GT future" if "gt" not in seen_gt else None
                    seen_gt.add("gt")
                    _draw_future(ax, gt_box[g, 0:2], gt_fut[g][vstep], "black", "--",
                                 lw=1.0, label=lab_gt)

        ax.set_xlim(BEV_RANGE, -BEV_RANGE)   # 左(+y)在左侧
        ax.set_ylim(-BEV_RANGE, BEV_RANGE)   # 前(+x)在上
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("y left (m)"); ax.set_ylabel("x forward (m)")
        ax.set_title(f"Agent motion (BEV): pred box+future(solid) vs GT future(dashed)  "
                     f"idx={idx}  GT={n_gt}  pred={len(keep)}  matched={len(gt_for_pred)}")
        ax.legend(loc="upper right", fontsize=7)

        png = os.path.join(out_dir, f"step3_motion_{n:02d}_idx{idx}.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[mot-viz] idx={idx}  GT={n_gt}  pred={len(keep)}  matched={len(gt_for_pred)}"
              f"  -> {png}")
        saved.append(png)

    print(f"[mot-viz] 共保存 {len(saved)} 张到 {out_dir}/")
    return saved


# ---------------------------------------------------------------------------
# 统一俯视图（单栏综合）：自车矩形 + 自车预测/真值轨迹 + 障碍框 + 障碍未来轨迹
# ---------------------------------------------------------------------------

# 自车典型尺寸（米）：宽 1.9、长 4.5（画自车矩形用；自车在原点、朝 +x 前方）
_EGO_W, _EGO_L = 1.9, 4.5


def run_topdown_viz(ckpt, meta="cache/meta_2000_agent.pt", num=6, seed=0,
                    val_ratio=0.2, out_dir="outputs/topdown", score_thr=0.3,
                    use_planner_v2=None):
    """统一俯视图：一张 BEV 俯视图里整合所有要素，直观看规划 + 感知 + 预测。

    每个验证样本画一张俯视图（ego 系，x 前向上、y 左向左），包含：
      - 自车：原点处的矩形（蓝），朝 +x；
      - 自车轨迹：预测(红实线) vs 真值(绿实线)；
      - 障碍（他车/人）：预测框(实线按类着色) + GT 框(虚线)；
      - 障碍未来轨迹：每个高置信度预测框的预测未来(细实线) + 匹配 GT 未来(细虚线)。

    只用模型没训过的验证集样本（val_indices 复现 seed/val_ratio 划分）。use_planner_v2
    为 None 时优先读 checkpoint 内保存的字段，保证按同版本规划头重建模型。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from train import match_pred_to_gt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = NuScenesBEVDataset(meta)
    vidx = val_indices(len(ds), val_ratio, seed)
    print(f"[topdown] 验证集 {len(vidx)} 个（seed={seed}）；取前 {num} 个可视化")

    state = torch.load(ckpt, map_location=dev, weights_only=True)
    if use_planner_v2 is None:
        use_planner_v2 = bool(state.get("use_planner_v2", True))
    # 矢量地图头开关：默认 False（见 run_visualization 的说明）——旧 ckpt 无该头、无该字段，
    # 重建时不构造，严格加载仍成立；新多任务 ckpt 带 use_vector_map=True 才重建矢量地图头。
    use_vector_map = bool(state.get("use_vector_map", False))
    model = NanoFSDv2(bev_size=BEV_SIZE, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, use_planner_v2=use_planner_v2,
                      use_vector_map=use_vector_map,
                      pretrained=True, freeze_backbone=True).to(dev)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[topdown] 已加载 {ckpt}  (use_planner_v2={use_planner_v2}, "
          f"use_vector_map={use_vector_map})")

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for n, idx in enumerate(vidx[:num]):
        sample = ds[idx]
        rec = ds.records[idx]
        images = sample["images"].unsqueeze(0).to(dev)
        K = sample["K"].unsqueeze(0); R = sample["R"].unsqueeze(0); t = sample["t"].unsqueeze(0)
        uv, valid = geometry_from_batch(K, R, t, IMG_HW, dev)
        command = sample.get("command")
        command = command.unsqueeze(0).to(dev) if command is not None else None
        with torch.no_grad():
            out = model(images, uv, valid, command=command)
        traj_pred = out["traj"]; det = out["det"]

        # ---- 自车轨迹（米）：预测 vs 真值 ----
        pred_m = traj_pred[0].cpu().numpy() * SCENE_SCALE        # (Tf,2)
        gt_m = rec["traj"]                                       # (Tf,2) 米
        tmask = rec["traj_mask"]
        ade = np.linalg.norm((pred_m - gt_m)[tmask], axis=1).mean()

        # ---- 障碍：预测框/类别/置信度/未来；GT 框/未来（去归一化到米）----
        pred_box_t = det["box"][0].cpu()                         # (N,5) 归一化
        gt_box_t = sample["agent_box"]; gt_mask_t = sample["agent_mask"]
        pred_idx, gt_idx = match_pred_to_gt(pred_box_t, gt_box_t, gt_mask_t)
        gt_for_pred = {int(p): int(g) for p, g in zip(pred_idx.tolist(), gt_idx.tolist())}

        pred_box = det["box"][0].cpu().numpy().copy(); pred_box[:, 0:2] *= SCENE_SCALE
        pred_cls = det["cls"][0].argmax(-1).cpu().numpy()
        pred_score = torch.sigmoid(det["score"][0, :, 0]).cpu().numpy()
        pred_fut = det["agent_fut"][0].cpu().numpy() * SCENE_SCALE   # (N,Tf,2)
        gt_box = sample["agent_box"].numpy().copy(); gt_box[:, 0:2] *= SCENE_SCALE
        gt_cls = sample["agent_cls"].numpy(); gt_mask = sample["agent_mask"].numpy()
        gt_fut = sample["agent_fut"].numpy() * SCENE_SCALE
        gt_fut_mask = sample["agent_fut_mask"].numpy()
        n_gt = int(gt_mask.sum())

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        # 道路真值底图：可行驶区栅格 (S,S)，road[i,j] 中 i=x前方向索引、j=y左方向索引。
        # 为避免 imshow 的行列↔轴映射歧义，直接用 pcolormesh 按**物理坐标**铺栅格：
        # 横轴 = y 左（俯视图里已反转，左侧为 +y）、纵轴 = x 前。构造每个格中心的 (y,x)
        # 网格，road 原样作为颜色值——方向由物理坐标决定，不会镜像/转置错位。
        drivable_gt = rec.get("drivable")
        if drivable_gt is not None:
            road = np.asarray(drivable_gt, dtype=np.float32)        # (S,S) 行 i=x前, 列 j=y左
            S = road.shape[0]
            axis = np.linspace(-BEV_RANGE, BEV_RANGE, S)            # 格中心物理坐标
            # pcolormesh(X, Y, C) 约定：C[i,j] 的行 i 配 Y(纵轴)、列 j 配 X(横轴)。
            # road 行 i=x前 -> 应配纵轴 x；列 j=y左 -> 应配横轴 y。故 X=y、Y=x：
            Xy = np.tile(axis, (S, 1))          # (S,S) 横轴 y，沿列 j 变化
            Yx = np.tile(axis[:, None], (1, S)) # (S,S) 纵轴 x，沿行 i 变化
            ax.pcolormesh(Xy, Yx, road, cmap="Greys", alpha=0.35, vmin=0, vmax=1,
                          shading="nearest", zorder=0)

        # 自车矩形（原点，朝 +x）。俯视图横轴=y、纵轴=x，故角点取 (y, x) 顺序。
        ego_corners = _oriented_box_corners(0.0, 0.0, 0.0, _EGO_W, _EGO_L)
        ego_poly = np.stack([ego_corners[:, 1], ego_corners[:, 0]], axis=-1)
        ax.add_patch(Polygon(ego_poly, closed=True, fill=True, facecolor="tab:blue",
                             edgecolor="navy", alpha=0.5, linewidth=1.5,
                             label="ego", zorder=6))

        # 自车轨迹：预测(红) vs 真值(绿)。俯视图横轴=y左、纵轴=x前，故 plot(y, x)。
        gmask_xy, pmask_xy = gt_m[tmask], pred_m[tmask]
        ax.plot(gmask_xy[:, 1], gmask_xy[:, 0], "g-o", ms=3, lw=2,
                label="ego GT", zorder=5)
        ax.plot(pmask_xy[:, 1], pmask_xy[:, 0], "r-x", ms=4, lw=2,
                label="ego Pred", zorder=5)

        # 障碍：GT 框（虚线）
        seen_gt = set()
        for k in range(len(gt_mask)):
            if not gt_mask[k]:
                continue
            cls = int(gt_cls[k])
            lab = f"GT {_CLS_NAMES.get(cls, cls)}" if cls not in seen_gt else None
            seen_gt.add(cls)
            _draw_box(ax, gt_box[k], _CLS_COLORS.get(cls, "gray"), "--", label=lab, lw=1.2)

        # 障碍：高置信度预测框(实线) + 预测未来(细实线) + 匹配 GT 未来(细虚线)
        keep = np.where(pred_score > score_thr)[0]
        seen_pred, seen_fut = set(), set()
        for k in keep:
            cls = int(pred_cls[k]); color = _CLS_COLORS.get(cls, "gray")
            lab = f"Pred {_CLS_NAMES.get(cls, cls)}" if cls not in seen_pred else None
            seen_pred.add(cls)
            _draw_box(ax, pred_box[k], color, "-", label=lab, lw=1.7)
            _draw_future(ax, pred_box[k, 0:2], pred_fut[k], color, "-", lw=1.0)
            if k in gt_for_pred:
                g = gt_for_pred[k]; vstep = gt_fut_mask[g]
                if vstep.any():
                    lab_gt = "obstacle GT future" if "f" not in seen_fut else None
                    seen_fut.add("f")
                    _draw_future(ax, gt_box[g, 0:2], gt_fut[g][vstep], "black", "--",
                                 lw=1.0, label=lab_gt)

        # 矢量地图叠加（可选，向后兼容）：模型含矢量地图头时 out["vmap"] 不为 None，叠加
        # 高置信度**预测折线**（实线按类着色）；纯旧 ckpt 无该头则 out.get("vmap") 为 None，
        # 整段跳过不报错（需求 8.5）。统一俯视图里只画预测折线（不画 GT 折线）以免过挤，
        # GT 折线留给专用的 run_vectormap_viz 对照（需求 8.1/8.2/8.3）。
        vmap = out.get("vmap")
        n_vmap = _overlay_vmap_pred(ax, vmap, score_thr) if vmap is not None else 0

        ax.set_xlim(BEV_RANGE, -BEV_RANGE)   # 左(+y)在左侧
        ax.set_ylim(-BEV_RANGE, BEV_RANGE)   # 前(+x)在上
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("y left (m)"); ax.set_ylabel("x forward (m)")
        _vmap_title = f"  vmap={n_vmap}" if vmap is not None else ""
        ax.set_title(f"Top-down: ego box+traj (pred/GT) + obstacles box+future  "
                     f"idx={idx}  ADE={ade:.2f} m  GT={n_gt}  pred={len(keep)}{_vmap_title}")
        ax.legend(loc="upper right", fontsize=7)

        png = os.path.join(out_dir, f"topdown_{n:02d}_idx{idx}.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[topdown] idx={idx}  ADE={ade:.2f} m  GT={n_gt}  pred={len(keep)}"
              f"{f'  vmap={n_vmap}' if vmap is not None else ''}  -> {png}")
        saved.append(png)

    print(f"[topdown] 共保存 {len(saved)} 张到 {out_dir}/")
    return saved


# ---------------------------------------------------------------------------
# 矢量地图可视化（任务 5.2）：预测折线（实线按类）+ GT 折线（虚线）叠到 BEV 俯视
# ---------------------------------------------------------------------------

# 3 类地图元素配色/名称（与 design.md 组件 6 一致）：
#   0=车道分隔线 lane divider、1=道路边界 road boundary、2=人行横道 ped crossing
_MAP_CLS_COLORS = {0: "tab:green", 1: "tab:red", 2: "tab:purple"}
_MAP_CLS_NAMES = {0: "lane divider", 1: "road boundary", 2: "ped crossing"}


def _draw_polyline(ax, pts_m, color, linestyle, lw=1.4, label=None):
    """在 BEV 俯视子图画一条折线。pts_m (P,2) 为 ego 系米坐标（列0=x前,列1=y左）。

    俯视图统一约定（与 _draw_box/_draw_future 一致）：横轴=y 左、纵轴=x 前，
    故画到画布时取 (y, x) 顺序，即 plot(pts_m[:,1], pts_m[:,0])。
    """
    ax.plot(pts_m[:, 1], pts_m[:, 0], linestyle, c=color, lw=lw, label=label, zorder=3)


def _overlay_vmap_pred(ax, vmap, score_thr=0.3):
    """把矢量地图头的**预测折线**（实线按类着色）叠加到已有的 BEV 俯视轴上。

    供综合图/统一俯视图复用：只画高置信度预测折线（sigmoid(score) > score_thr），pts 去
    归一化乘回 SCENE_SCALE 到米，遵循 ego 系 x前/y左 约定（横轴=y左、纵轴=x前，见
    _draw_polyline）。保持轻量——不画 GT 折线以免与检测框/轨迹过度重叠；GT 对照留给专用
    的 run_vectormap_viz。每类只给一个图例条目。

    vmap：模型 forward 返回的 dict['vmap']（含 pts/cls/score）；为 None 时调用方应跳过。
    返回画出的预测折线数量（供标题展示）。
    """
    pred_pts = vmap["pts"][0].cpu().numpy().copy()           # (M,P,2) 归一化
    pred_pts *= SCENE_SCALE                                  # -> 米
    pred_cls = vmap["cls"][0].argmax(-1).cpu().numpy()       # (M,)
    pred_score = torch.sigmoid(vmap["score"][0, :, 0]).cpu().numpy()  # (M,)

    keep = np.where(pred_score > score_thr)[0]
    seen = set()
    for k in keep:
        cls = int(pred_cls[k])
        lab = f"vmap {_MAP_CLS_NAMES.get(cls, cls)}" if cls not in seen else None
        seen.add(cls)
        _draw_polyline(ax, pred_pts[k], _MAP_CLS_COLORS.get(cls, "gray"), "-",
                       lw=1.6, label=lab)
    return len(keep)


def run_vectormap_viz(ckpt, meta="cache/meta_2000_vmap.pt", num=6, seed=0,
                      val_ratio=0.2, out_dir="outputs", score_thr=0.3):
    """加载矢量地图 ckpt，对**验证集**样本画预测折线(实线按类) + GT 折线(虚线)。

    孪生关系（镜像 run_detection_viz）：把「高置信度预测框 + GT 虚线框」换成「高置信度
    预测折线 + GT 虚线折线」，其余（验证划分复现、置信度过滤、去归一化、俯视约定）完全
    对齐检测头可视化。

    严谨性：用与训练相同的 seed/val_ratio 复现验证划分（见 val_indices），只可视化模型
    没训过的验证样本（需求 8.4）。预测折线按 sigmoid(score) > score_thr 过滤为「高置信度」
    并按类别着色（实线）；GT 折线用虚线对照（需求 8.1）。pts 去归一化乘回 SCENE_SCALE 到米
    （需求 8.3）；折线遵循 ego 系 x前/y左 约定（需求 8.2）。保存到 out_dir/step_vmap_*.png。

    模型重建：读 checkpoint 内存的 use_vector_map 开关（与 use_planner_v2 同款处理），
    保证按同版本重建；矢量地图 ckpt 恒开此头（否则 forward 无 'vmap' 键无从可视化）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = NuScenesBEVDataset(meta)
    vidx = val_indices(len(ds), val_ratio, seed)
    print(f"[vmap-viz] 验证集 {len(vidx)} 个（seed={seed}）；取前 {num} 个可视化")

    state = torch.load(ckpt, map_location=dev, weights_only=True)
    # 矢量地图头必须在，才能产出 'vmap'；按 checkpoint 存的开关重建（默认 True）
    use_vector_map = bool(state.get("use_vector_map", True))
    model = NanoFSDv2(bev_size=BEV_SIZE, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, use_vector_map=use_vector_map,
                      pretrained=True, freeze_backbone=True).to(dev)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[vmap-viz] 已加载 {ckpt}  (use_vector_map={use_vector_map})")

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for n, idx in enumerate(vidx[:num]):
        sample = ds[idx]
        images = sample["images"].unsqueeze(0).to(dev)
        K = sample["K"].unsqueeze(0); R = sample["R"].unsqueeze(0); t = sample["t"].unsqueeze(0)
        uv, valid = geometry_from_batch(K, R, t, IMG_HW, dev)
        with torch.no_grad():
            vmap = model(images, uv, valid, command=None)["vmap"]

        # 预测折线/类别/置信度（去归一化到米）：pts 与 GT 同为 SCENE_SCALE 归一化 -> 乘回米
        pred_pts = vmap["pts"][0].cpu().numpy().copy()           # (M,P,2) 归一化
        pred_pts *= SCENE_SCALE                                  # -> 米
        pred_cls = vmap["cls"][0].argmax(-1).cpu().numpy()       # (M,)
        pred_score = torch.sigmoid(vmap["score"][0, :, 0]).cpu().numpy()  # (M,)

        # GT 折线/类别/mask（去归一化到米）
        gt_pts = sample["map_pts"].numpy().copy()                # (MAP_MAX,P,2) 归一化
        gt_pts *= SCENE_SCALE                                    # -> 米
        gt_cls = sample["map_cls"].numpy()                       # (MAP_MAX,)
        gt_mask = sample["map_poly_mask"].numpy()                # (MAP_MAX,)
        n_gt = int(gt_mask.sum())

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        ax.plot(0, 0, "b^", ms=12, zorder=5, label="ego")

        # GT 折线（虚线对照，按类着色）
        seen_gt = set()
        for k in range(len(gt_mask)):
            if not gt_mask[k]:
                continue
            cls = int(gt_cls[k])
            lab = f"GT {_MAP_CLS_NAMES.get(cls, cls)}" if cls not in seen_gt else None
            seen_gt.add(cls)
            _draw_polyline(ax, gt_pts[k], _MAP_CLS_COLORS.get(cls, "gray"), "--",
                           lw=1.2, label=lab)

        # 高置信度预测折线（实线，按类别着色）
        keep = np.where(pred_score > score_thr)[0]
        seen_pred = set()
        for k in keep:
            cls = int(pred_cls[k])
            lab = f"Pred {_MAP_CLS_NAMES.get(cls, cls)}" if cls not in seen_pred else None
            seen_pred.add(cls)
            _draw_polyline(ax, pred_pts[k], _MAP_CLS_COLORS.get(cls, "gray"), "-",
                           lw=1.8, label=lab)

        ax.set_xlim(BEV_RANGE, -BEV_RANGE)   # 左(+y)在左侧
        ax.set_ylim(-BEV_RANGE, BEV_RANGE)   # 前(+x)在上
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("y left (m)"); ax.set_ylabel("x forward (m)")
        ax.set_title(f"Vector map (BEV): pred(solid, score>{score_thr}) vs GT(dashed)  "
                     f"idx={idx}  GT={n_gt}  pred={len(keep)}")
        ax.legend(loc="upper right", fontsize=7)

        png = os.path.join(out_dir, f"step_vmap_{n:02d}_idx{idx}.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[vmap-viz] idx={idx}  GT={n_gt}  high-score pred={len(keep)}  -> {png}")
        saved.append(png)

    print(f"[vmap-viz] 共保存 {len(saved)} 张到 {out_dir}/")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--meta", type=str, default="cache/meta_2000.pt")
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="outputs/viz_sub2000")
    parser.add_argument("--det", action="store_true",
                        help="检测框可视化模式（高置信度预测框 + GT 虚线框叠到 BEV）")
    parser.add_argument("--motion", action="store_true",
                        help="他车预测轨迹可视化模式（高置信度预测框的预测/GT 未来轨迹叠到 BEV）")
    parser.add_argument("--topdown", action="store_true",
                        help="统一俯视图模式（自车矩形+预测/真值轨迹+障碍框+障碍未来轨迹，单栏）")
    parser.add_argument("--vmap", action="store_true",
                        help="矢量地图可视化模式（高置信度预测折线 + GT 虚线折线叠到 BEV）")
    parser.add_argument("--score-thr", type=float, default=0.5,
                        help="检测/运动框可视化的置信度阈值（sigmoid(score) 高于此值才画）")
    args = parser.parse_args()
    if args.vmap:
        run_vectormap_viz(args.ckpt, args.meta, args.num, args.seed,
                          args.val_ratio, args.out_dir, args.score_thr)
    elif args.topdown:
        run_topdown_viz(args.ckpt, args.meta, args.num, args.seed,
                        args.val_ratio, args.out_dir, args.score_thr)
    elif args.motion:
        run_motion_viz(args.ckpt, args.meta, args.num, args.seed,
                       args.val_ratio, args.out_dir, args.score_thr)
    elif args.det:
        run_detection_viz(args.ckpt, args.meta, args.num, args.seed,
                          args.val_ratio, args.out_dir, args.score_thr)
    else:
        run_visualization(args.ckpt, args.meta, args.num, args.seed,
                          args.val_ratio, args.out_dir)


if __name__ == "__main__":
    main()
