"""台阶1-第1步演示入口：可视化结构化 agent 真值（GT）。

从已构建的 meta 缓存里取若干样本，在 BEV 俯视图上画出：
  - 自车（原点，蓝色三角）
  - 每个有效 GT agent 的朝向框（按类别着色：车=橙、人=青）
  - 每个 agent 的 GT 未来轨迹（细线，相对其当前中心累加回绝对位置）

目的：人眼确认框位置/朝向/未来轨迹方向是否合理（数据层正确性的直观验证）。

坐标约定：ego 系，x 前、y 左；BEV 俯视图里让 x 向上、y 向左。
Dataset 已把 agent_box 的 (x,y) 与 agent_fut 按 SCENE_SCALE 归一化，这里乘回米。

用法:
  python viz_agent_gt.py --meta cache/meta_2000_agent.pt --num 6
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境后端
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from dataset import BEV_RANGE, SCENE_SCALE, NuScenesBEVDataset

# 2 类配色（0=车 vehicle，1=人 human）
_CLS_COLORS = {0: "tab:orange", 1: "tab:cyan"}
_CLS_NAMES = {0: "vehicle", 1: "human"}


def _oriented_box_corners(x, y, yaw, w, l):
    """由中心 (x,y)、朝向 yaw、尺寸 (w,l) 算 ego 系下框的 4 个角 (4,2)。

    nuScenes size=[w,l,h]：w 为宽（沿车体 y），l 为长（沿车体前进方向 x）。
    yaw=0 时车头朝 ego 的 +x（前）。角点顺序沿矩形一圈。
    """
    dx, dy = l / 2.0, w / 2.0  # 车体前后半长、左右半宽
    # 车体坐标系 4 角：(前右, 前左, 后左, 后右)
    local = np.array([[+dx, -dy], [+dx, +dy], [-dx, +dy], [-dx, -dy]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return (local @ rot.T) + np.array([x, y])


def _plot_sample(ax, sample):
    """在一个子图上画单个样本的 agent 真值（已反归一化到米）。"""
    agent_box = sample["agent_box"].numpy().copy()        # (MAX_GT,5) [x,y,yaw,w,l]
    agent_mask = sample["agent_mask"].numpy()             # (MAX_GT,)
    agent_cls = sample["agent_cls"].numpy()               # (MAX_GT,)
    agent_fut = sample["agent_fut"].numpy().copy()        # (MAX_GT,Tf,2)
    agent_fut_mask = sample["agent_fut_mask"].numpy()     # (MAX_GT,Tf)

    # 反归一化：Dataset 把 xy 列与未来轨迹按 SCENE_SCALE 缩小，这里乘回米
    agent_box[:, 0:2] *= SCENE_SCALE
    agent_fut *= SCENE_SCALE

    # ego 原点
    ax.plot(0, 0, "b^", ms=12, zorder=5)

    n_valid = int(agent_mask.sum())
    seen_cls = set()
    for k in range(len(agent_mask)):
        if not agent_mask[k]:
            continue
        x, y, yaw, w, l = agent_box[k]
        cls = int(agent_cls[k])
        color = _CLS_COLORS.get(cls, "gray")

        # 朝向框
        corners = _oriented_box_corners(x, y, yaw, w, l)
        label = _CLS_NAMES.get(cls) if cls not in seen_cls else None
        seen_cls.add(cls)
        poly = Polygon(corners, closed=True, fill=False, edgecolor=color,
                       linewidth=1.5, label=label, zorder=3)
        ax.add_patch(poly)

        # 朝向指示：从中心指向车头（+x 车体方向）
        head = np.array([x, y]) + (l / 2.0) * np.array([np.cos(yaw), np.sin(yaw)])
        ax.plot([x, head[0]], [y, head[1]], "-", c=color, lw=1.0, zorder=4)

        # GT 未来轨迹（相对当前中心 -> 加回中心成绝对），只画有效步
        step_mask = agent_fut_mask[k]
        if step_mask.any():
            fut_abs = agent_fut[k] + np.array([x, y])      # (Tf,2) 绝对位置
            fut_abs = fut_abs[step_mask]
            track = np.vstack([[x, y], fut_abs])           # 从当前中心起画
            ax.plot(track[:, 0], track[:, 1], "-", c=color, lw=0.8, alpha=0.9, zorder=2)

    # BEV 俯视：x 前（向上）、y 左（向左）。matplotlib 默认右为 +x，
    # 这里让横轴=y(左为正，故左侧为正)、纵轴=x(上为正)：
    #   横轴用 ego y，但要"左"在左侧 => 反转横轴方向
    ax.set_xlim(BEV_RANGE, -BEV_RANGE)   # 左(+y)在左侧
    ax.set_ylim(-BEV_RANGE, BEV_RANGE)   # 前(+x)在上
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("y left (m)")
    ax.set_ylabel("x forward (m)")
    return n_valid


def run_viz_agent(meta="cache/meta_2000_agent.pt", num=6,
                  out_dir="outputs", seed=0):
    """从缓存取 num 个样本，各出一张 agent GT 的 BEV 图，存到 out_dir。"""
    ds = NuScenesBEVDataset(meta)
    n = len(ds)
    # 固定 seed 取一批样本（优先取含 agent 的，避免全是空场景）
    g = np.random.default_rng(seed)
    order = g.permutation(n).tolist()

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    picked = 0
    for idx in order:
        if picked >= num:
            break
        sample = ds[idx]
        n_valid = int(sample["agent_mask"].sum())
        # 尽量挑有 agent 的样本展示（空场景也允许，但优先非空）
        if n_valid == 0 and picked < num - 1:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        n_valid = _plot_sample(ax, sample)
        ax.set_title(f"Agent GT (BEV top-down)  idx={idx}  agents={n_valid}")
        if n_valid > 0:
            ax.legend(loc="upper right", fontsize=8)

        out = os.path.join(out_dir, f"step1_agent_gt_{picked:02d}_idx{idx}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz-agent] idx={idx}  valid_agents={n_valid}  -> {out}")
        saved.append(out)
        picked += 1

    print(f"[viz-agent] 共保存 {len(saved)} 张到 {out_dir}/")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="cache/meta_2000_agent.pt")
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_viz_agent(args.meta, args.num, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
