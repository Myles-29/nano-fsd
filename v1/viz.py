"""viz.py — BEV 轨迹可视化（GT vs Pred）。

本模块实现任务 9.1 的 `visualize`：在同一张 BEV 图上画出周围 agent、GT 未来
轨迹（绿）、预测轨迹（红）与自车原点（蓝三角）。`--ckpt` 入口在任务 9.2 实现。

坐标系与单位约定（与 data.py 保持一致）：ego 坐标系，x 向前、y 向左，单位米。
"""

import os

import numpy as np
import torch
from torch import Tensor

from config import Config
from data import NanoFSDDataset, Sample, denormalize, split_train_val_indices
from model import NanoFSD
from train import _resolve_device


def _to_numpy(x):
    """把 Tensor / ndarray / 列表安全转成 numpy 数组（Tensor 先 detach 到 cpu）。"""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def visualize(sample: Sample, traj_pred_m, ax=None, title: str | None = None,
              out_png: str | None = None):
    """在 BEV 图上画 周围 agent + GT 轨迹(绿) + Pred 轨迹(红) + 自车原点(蓝三角)。

    单位契约（重要，供任务 9.2 调用方遵循）
    --------------------------------------
    - ``sample``：一个 **标准化单位** 的 `Sample`（即缓存/`dataset[i]` 的原样）。
      本函数内部对 ``sample.traj_gt`` 与 ``sample.agent_gt`` 调用 `denormalize`
      转成米后再绘制。
    - ``traj_pred_m``：预测轨迹，形状 ``(Tf, 2)``，**已是米制单位**。调用方（9.2）
      负责在前向得到标准化输出后 `denormalize(traj)` 再传入，避免双重反标准化。

    参数
    ----
    sample      : Sample，标准化单位。只画有效部分（agent_mask / traj_mask）。
    traj_pred_m : (Tf, 2) Tensor 或 ndarray，米制单位的预测轨迹。
    ax          : 可选 matplotlib 轴；为 None 时新建 fig/ax。
    title       : 可选图标题。
    out_png     : 可选保存路径；给定时自动创建父目录并保存。

    返回
    ----
    绘制所用的 matplotlib Axes。
    """
    # headless 环境下也能出图：在函数内设置 Agg 后端再导入 pyplot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- 取有效部分并反标准化到米 ---
    agent_mask = _to_numpy(sample.agent_mask).astype(bool)   # (N,)
    traj_mask = _to_numpy(sample.traj_mask).astype(bool)     # (Tf,)

    agents_xy = denormalize(_to_numpy(sample.agent_gt))[agent_mask]  # (k, 2) 米
    gt_xy = denormalize(_to_numpy(sample.traj_gt))[traj_mask]        # (tf, 2) 米

    # 预测轨迹已是米；只取与 GT 对齐的有效步，保证公平对比
    pred_xy = _to_numpy(traj_pred_m)                                 # (Tf, 2) 米
    assert pred_xy.ndim == 2 and pred_xy.shape[1] == 2, \
        f"traj_pred_m 形状应为 (Tf, 2), 实际 {pred_xy.shape}"
    pred_valid = traj_mask[:pred_xy.shape[0]]
    pred_xy = pred_xy[:len(pred_valid)][pred_valid]

    # --- 画图 ---
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # 周围 agent：散点标记其当前帧位置（最小可用版本，等价于 data.py demo 风格）
    if len(agents_xy) > 0:
        ax.scatter(agents_xy[:, 0], agents_xy[:, 1], s=18, c="gray",
                   marker="s", label=f"agents ({len(agents_xy)})")

    # GT（绿）与 Pred（红）
    if len(gt_xy) > 0:
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "g-o", label="GT")
    if len(pred_xy) > 0:
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "r-x", label="Pred")

    # 自车在原点
    ax.plot(0, 0, "b^", markersize=12, label="Ego")

    ax.set_aspect("equal")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")

    if out_png is not None:
        parent = os.path.dirname(out_png)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ax.figure.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)

    return ax


def visualize_perception(sample: Sample, agent_pred_m, ax=None,
                         title: str | None = None, out_png: str | None = None):
    """感知结果 BEV 可视化：周围 agent 的真值位置(绿) vs 感知头预测位置(红)。

    与 `visualize`（规划）对等，但画的是 **感知头输出**：每个有效 agent 的真值位置
    画绿点、预测位置画红叉，并用一条灰色细线把「预测→真值」连起来，直观展示偏差
    方向与大小；自车在原点（蓝三角）。

    单位契约（与 visualize 一致）
    ------------------------------
    - ``sample``：标准化单位的 `Sample`；内部对 ``sample.agent_gt`` 调 denormalize 转米。
    - ``agent_pred_m``：感知头预测，形状 ``(N, 2)``，**已是米制**。调用方负责在前向得到
      标准化输出后 denormalize 再传入，避免双重反标准化。只画 agent_mask 有效位。

    参数
    ----
    sample       : Sample，标准化单位。只画有效 agent（agent_mask）。
    agent_pred_m : (N, 2) Tensor 或 ndarray，米制单位的预测 agent 相对位置。
    ax           : 可选 matplotlib 轴；为 None 时新建 fig/ax。
    title        : 可选图标题。
    out_png      : 可选保存路径；给定时自动创建父目录并保存。

    返回
    ----
    绘制所用的 matplotlib Axes。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agent_mask = _to_numpy(sample.agent_mask).astype(bool)       # (N,)
    gt_xy = denormalize(_to_numpy(sample.agent_gt))[agent_mask]  # (k, 2) 米

    pred_all = _to_numpy(agent_pred_m)                           # (N, 2) 米
    assert pred_all.ndim == 2 and pred_all.shape[1] == 2, \
        f"agent_pred_m 形状应为 (N, 2), 实际 {pred_all.shape}"
    pred_xy = pred_all[agent_mask]                               # (k, 2) 米

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # 预测→真值 的连线（灰色细线），直观看偏差方向与大小
    for (px, py), (gx, gy) in zip(pred_xy, gt_xy):
        ax.plot([px, gx], [py, gy], "-", color="gray", lw=0.7, alpha=0.6, zorder=1)

    # 真值（绿点）与 预测（红叉）
    if len(gt_xy) > 0:
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1], s=40, c="green", marker="o",
                   label=f"GT agents ({len(gt_xy)})", zorder=3)
    if len(pred_xy) > 0:
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=40, c="red", marker="x",
                   label="Pred agents", zorder=3)

    # 自车在原点
    ax.plot(0, 0, "b^", markersize=12, label="Ego", zorder=4)

    ax.set_aspect("equal")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")

    if out_png is not None:
        parent = os.path.dirname(out_png)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ax.figure.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)

    return ax


# =============================================================================
# 任务 9.2 / 9.3：可视化入口 `--ckpt` + 多样本保存
# -----------------------------------------------------------------------------
# 加载 checkpoint_stage2.pt，对若干样本（含直行与转弯）no_grad 前向得到规划轨迹，
# denormalize 回米后调用 visualize，各存一张图到 outputs/step9_traj_*.png，
# 标题标注样本序号与该样本的平均位移误差（ADE，米）。
# =============================================================================


def _load_ckpt(path: str, device: str) -> dict:
    """安全加载 checkpoint（{"model": state_dict}）；优先 weights_only=True，老版本回退。

    与 train.py 的 _load_stage1_ckpt 思路一致：checkpoint 只含权重张量，用
    weights_only=True 更安全；若当前 torch 版本不支持该参数则回退普通加载。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 checkpoint：{path!r}。"
            f"请先运行 `python train.py --stage 2` 完成阶段二训练后再可视化。")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # 老版本 torch 无 weights_only 参数，回退到普通加载
        return torch.load(path, map_location=device)


def _ade_meters(traj_pred_m: Tensor, sample: Sample) -> float:
    """计算单样本的平均位移误差（ADE，米），仅在有效未来步（traj_mask）上。

    traj_pred_m 已是米制 (Tf, 2)；GT 从 sample.traj_gt（标准化）denormalize 回米。
    对每个有效步取预测与真值的欧氏距离，再求平均。无有效步时返回 nan。
    """
    traj_mask = sample.traj_mask.bool()                          # (Tf,)
    gt_m = denormalize(sample.traj_gt.float())                   # (Tf, 2) 米
    pred_m = traj_pred_m.float()                                 # (Tf, 2) 米
    if int(traj_mask.sum()) == 0:
        return float("nan")
    dist = torch.linalg.norm(pred_m - gt_m, dim=-1)              # (Tf,)
    return float(dist[traj_mask].mean())


def _perception_ade_meters(agent_pred_m: Tensor, sample: Sample) -> float:
    """感知头单样本 ADE（米）：仅在有效 agent（agent_mask）上算预测与真值的欧氏距离均值。

    agent_pred_m 已是米制 (N, 2)；GT 从 sample.agent_gt（标准化）denormalize 回米。
    无有效 agent 时返回 nan。
    """
    agent_mask = sample.agent_mask.bool()                        # (N,)
    gt_m = denormalize(sample.agent_gt.float())                  # (N, 2) 米
    pred_m = agent_pred_m.float()                                # (N, 2) 米
    if int(agent_mask.sum()) == 0:
        return float("nan")
    dist = torch.linalg.norm(pred_m - gt_m, dim=-1)              # (N,)
    return float(dist[agent_mask].mean())


def _final_lateral(sample: Sample) -> float:
    """样本「转弯程度」启发式：最后一个有效未来步的横向位移 |y|（米，越大越像转弯）。

    直行时自车沿 x 向前、y≈0；转弯时 y 偏离明显。用它区分直行 / 转弯场景。
    """
    traj_mask = sample.traj_mask.bool()
    if int(traj_mask.sum()) == 0:
        return 0.0
    gt_m = denormalize(sample.traj_gt.float())                   # (Tf, 2) 米
    last_valid = int(torch.nonzero(traj_mask).max())             # 最后一个有效步下标
    return float(abs(gt_m[last_valid, 1]))


def _auto_select_indices(dataset: NanoFSDDataset, k: int = 4,
                         candidates: list[int] | None = None) -> list[int]:
    """从候选集合里自动挑样：至少含最直行（|y_end| 最小）与最转弯（|y_end| 最大）各一个。

    candidates=None 时在全体样本里挑；给定时**只在候选索引内**挑（用于只从验证集选样）。
    先按横向位移给候选打分，取最直与最弯两端点，再按「与已选横向位移差异最大」贪心补足
    到 k 个，增加直行/转弯多样性，不会越界或崩溃。
    """
    pool = list(range(len(dataset))) if candidates is None else list(candidates)
    if not pool:
        return []
    lat = {i: _final_lateral(dataset[i]) for i in pool}
    pool_sorted = sorted(pool, key=lambda i: lat[i])
    straight = pool_sorted[0]            # 最直行
    turning = pool_sorted[-1]            # 最转弯
    selected = [straight]
    if turning != straight:
        selected.append(turning)
    # 贪心补足：每次选与已选集合横向位移差异最大的候选，增加多样性
    while len(selected) < min(k, len(pool)):
        best_i, best_gap = None, -1.0
        for i in pool:
            if i in selected:
                continue
            gap = min(abs(lat[i] - lat[j]) for j in selected)
            if gap > best_gap:
                best_gap, best_i = gap, i
        if best_i is None:
            break
        selected.append(best_i)
    return selected


def run_from_checkpoint(ckpt_path: str = "checkpoint_stage2.pt",
                        cache: str = "data/cache_mini.pt",
                        cfg: Config | None = None,
                        indices: list[int] | None = None,
                        out_dir: str = "outputs",
                        k: int = 10, val_ratio: float = 0.2,
                        seed: int = 0) -> list[dict]:
    """加载规划 checkpoint，**只从验证集**选若干样本前向并保存 BEV 对比图。

    严谨性关键：可视化样本仅取自训练时的验证集（同 seed/val_ratio 的划分），
    保证展示的是模型**未训练过**的样本，图上 ADE 与训练日志的 val ADE 可对齐。

    流程：建模 -> 加载权重 -> eval -> 复现 val 划分 -> 在 val 索引内自动选样
    （含直行+转弯）-> no_grad 前向 -> denormalize 回米 -> 计算 ADE（米）-> 存图。

    参数
    ----
    ckpt_path : 规划 checkpoint 路径（{"model": state_dict}，建议 *_best.pt）。
    cache     : 预处理缓存路径，构建 NanoFSDDataset。
    cfg       : Config；None 时用默认 Config()。
    indices   : 覆盖自动选样；给定时直接用这些索引（不限制在验证集内，供手动排查）。
    out_dir   : 图片输出目录，图名 step9_traj_{i}.png（i=样本序号）。
    k         : 自动选样数量（默认 10）。
    val_ratio : 验证集占比，须与训练时一致（默认 0.2）。
    seed      : 划分种子，须与训练时一致（默认 0）。

    返回
    ----
    每个样本一个 dict：{index, ade_m, kind, out_png}。
    """
    cfg = cfg or Config()
    device = _resolve_device(cfg.device)

    # --- 建模 + 加载权重 ---
    model = NanoFSD(cfg)
    ckpt = _load_ckpt(ckpt_path, device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    dataset = NanoFSDDataset(cache)
    # 复现训练时的 val 划分，确保只从模型未训过的验证样本里选
    val_idx, _ = split_train_val_indices(len(dataset), val_ratio=val_ratio, seed=seed)
    print(f"[viz] 已加载 checkpoint {ckpt_path}（device={device}）")
    print(f"[viz] 数据集 {cache}：共 {len(dataset)} 个样本；"
          f"验证集 {len(val_idx)} 个（seed={seed}, val_ratio={val_ratio}）")

    # --- 选样：默认只在验证集内自动选 k 个（含直行+转弯）；indices 给定时直接用 ---
    if indices is None:
        sel = _auto_select_indices(dataset, k=k, candidates=val_idx)
        print(f"[viz] 从验证集自动选取 {len(sel)} 个样本（含直行/转弯）")
    else:
        sel = [i for i in indices if 0 <= i < len(dataset)]
    if not sel:
        raise ValueError("没有可用于可视化的样本（数据集为空或 indices 越界）。")

    # 为「直行/转弯」分类设一个阈值：用选中样本横向位移的中位数，稳健且不依赖绝对尺度
    lat_vals = [_final_lateral(dataset[i]) for i in sel]
    lat_thresh = float(np.median(lat_vals)) if len(lat_vals) > 1 else 0.0

    os.makedirs(out_dir, exist_ok=True)
    results: list[dict] = []
    for i in sel:
        sample = dataset[i]
        with torch.no_grad():
            traj = model(sample.ego_hist[None].to(device),
                         sample.agents[None].to(device),
                         sample.agent_mask[None].to(device),
                         mode="planning")[0]            # (Tf, 2) 标准化
        traj_m = denormalize(traj.cpu())                # (Tf, 2) 米

        ade = _ade_meters(traj_m, sample)
        lat = _final_lateral(sample)
        kind = "turn" if lat > lat_thresh else "straight"
        title = f"sample {i}  ADE={ade:.2f} m"
        out_png = os.path.join(out_dir, f"step9_traj_{i}.png")
        visualize(sample, traj_m, title=title, out_png=out_png)

        results.append({"index": i, "ade_m": ade, "kind": kind, "out_png": out_png})
        print(f"[viz] sample {i:>3}  [{kind:^8}]  ADE={ade:6.2f} m  -> {out_png}")

    print(f"[viz] 共保存 {len(results)} 张对比图到 {out_dir}/step9_traj_*.png")
    return results


def _auto_select_perception_indices(dataset: NanoFSDDataset, k: int = 4,
                                    candidates: list[int] | None = None) -> list[int]:
    """感知可视化选样：从候选里挑有效 agent 最多的样本（场景越丰富越能看出质量）。

    candidates=None 时在全体挑；给定时只在候选索引内挑（用于只从验证集选样）。
    """
    pool = list(range(len(dataset))) if candidates is None else list(candidates)
    if not pool:
        return []
    order = sorted(pool, key=lambda i: int(dataset[i].agent_mask.sum()), reverse=True)
    return order[:min(k, len(pool))]


def run_perception_from_checkpoint(ckpt_path: str = "checkpoint_stage1_best.pt",
                                   cache: str = "data/cache_mini.pt",
                                   cfg: Config | None = None,
                                   indices: list[int] | None = None,
                                   out_dir: str = "outputs",
                                   k: int = 10, val_ratio: float = 0.2,
                                   seed: int = 0) -> list[dict]:
    """加载感知 checkpoint，**只从验证集**选若干样本前向感知头并保存 BEV 对比图。

    与规划可视化一致：样本仅取自训练时的验证集（同 seed/val_ratio），保证是模型
    未训过的样本。流程：建模 -> 加载权重 -> eval -> 复现 val 划分 -> 在 val 内挑
    agent 最多的 k 个 -> no_grad 前向 -> denormalize -> 感知 ADE -> 存图。

    参数
    ----
    ckpt_path : 感知 checkpoint 路径（建议 *_best.pt）。
    cache     : 预处理缓存路径。
    cfg       : Config；None 用默认。
    indices   : 覆盖自动选样；给定时直接用（不限验证集，供手动排查）。
    out_dir   : 图片输出目录，图名 step9_perc_{i}.png。
    k         : 自动选样数量（默认 10）。
    val_ratio / seed : 验证集划分参数，须与训练一致（默认 0.2 / 0）。

    返回
    ----
    每个样本一个 dict：{index, ade_m, n_valid, out_png}。
    """
    cfg = cfg or Config()
    device = _resolve_device(cfg.device)

    model = NanoFSD(cfg)
    ckpt = _load_ckpt(ckpt_path, device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    dataset = NanoFSDDataset(cache)
    val_idx, _ = split_train_val_indices(len(dataset), val_ratio=val_ratio, seed=seed)
    print(f"[viz] 已加载感知 checkpoint {ckpt_path}（device={device}）")
    print(f"[viz] 数据集 {cache}：共 {len(dataset)} 个样本；"
          f"验证集 {len(val_idx)} 个（seed={seed}, val_ratio={val_ratio}）")

    if indices is None:
        sel = _auto_select_perception_indices(dataset, k=k, candidates=val_idx)
        print(f"[viz] 从验证集自动选取 {len(sel)} 个样本（agent 最多）")
    else:
        sel = [i for i in indices if 0 <= i < len(dataset)]
    if not sel:
        raise ValueError("没有可用于可视化的样本（数据集为空或 indices 越界）。")

    os.makedirs(out_dir, exist_ok=True)
    results: list[dict] = []
    for i in sel:
        sample = dataset[i]
        with torch.no_grad():
            pred = model(sample.ego_hist[None].to(device),
                         sample.agents[None].to(device),
                         sample.agent_mask[None].to(device),
                         mode="perception")[0]           # (N, 2) 标准化
        pred_m = denormalize(pred.cpu())                 # (N, 2) 米

        ade = _perception_ade_meters(pred_m, sample)
        n_valid = int(sample.agent_mask.sum())
        title = f"sample {i}  perception ADE={ade:.2f} m  (agents={n_valid})"
        out_png = os.path.join(out_dir, f"step9_perc_{i}.png")
        visualize_perception(sample, pred_m, title=title, out_png=out_png)

        results.append({"index": i, "ade_m": ade, "n_valid": n_valid,
                        "out_png": out_png})
        print(f"[viz] sample {i:>3}  agents={n_valid:>2}  "
              f"perception ADE={ade:6.2f} m  -> {out_png}")

    print(f"[viz] 共保存 {len(results)} 张感知对比图到 {out_dir}/step9_perc_*.png")
    return results


if __name__ == "__main__":
    # 规划（默认）: python viz.py --ckpt checkpoint_stage2.pt
    # 感知:         python viz.py --task perception --ckpt checkpoint_stage1_best.pt
    import argparse

    parser = argparse.ArgumentParser(
        description="nano-fsd 可视化入口：感知 / 规划 的 GT vs Pred BEV 对比图")
    parser.add_argument("--task", type=str, choices=["planning", "perception"],
                        default="planning",
                        help="可视化任务：planning=规划轨迹（默认，阶段二）；"
                             "perception=感知 agent 位置（阶段一）")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint 路径；默认规划用 checkpoint_stage2.pt、"
                             "感知用 checkpoint_stage1_best.pt")
    parser.add_argument("--cache", type=str, default="/home/miaozijian/life/mzj_planning/data/cache_trainval.pt",
                        help="预处理缓存路径")
    parser.add_argument("--out-dir", type=str, default="outputs",
                        help="图片输出目录")
    parser.add_argument("--indices", type=str, default=None,
                        help="逗号分隔的样本序号，覆盖自动选样（如 0,3,7；不限验证集）")
    parser.add_argument("--k", type=int, default=10,
                        help="自动选样数量（默认 10，仅从验证集选）")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="验证集占比，须与训练一致（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0,
                        help="验证集划分种子，须与训练一致（默认 0）")
    args = parser.parse_args()

    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]

    if args.task == "perception":
        ckpt = args.ckpt or "checkpoint_stage1_best.pt"
        run_perception_from_checkpoint(ckpt_path=ckpt, cache=args.cache,
                                       indices=indices, out_dir=args.out_dir,
                                       k=args.k, val_ratio=args.val_ratio,
                                       seed=args.seed)
    else:
        ckpt = args.ckpt or "checkpoint_stage2.pt"
        run_from_checkpoint(ckpt_path=ckpt, cache=args.cache,
                            indices=indices, out_dir=args.out_dir,
                            k=args.k, val_ratio=args.val_ratio, seed=args.seed)
