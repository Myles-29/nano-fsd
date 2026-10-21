"""任务 5.4：显存与端到端前向自测。

目的（对齐 design.md「性能考量（单卡 24G）」与需求 5.4、4.4、9.4）：
  1) 全头开启（规划 v2 + 占据 + 可行驶区 + 检测 + 运动 + 矢量地图），batch_size=4 跑一次
     完整的前向 + 反向（多任务总损失），打印峰值显存并断言 ≤ 24G（单卡预算）。
  2) 需求 4.4「规划退化」轻量校验：把检测头的存在置信度 score 全部压成很负（等价
     「agent 全背景」），确认 PlanningHeadV2 的 traj 仍为有限值（无 NaN/Inf），即优雅退化
     为「仅 BEV 规划」而不报错。

运行（在 v2/ 目录下）：
  conda run -n mcts_debug python selftest_mem.py
可选用随机 dummy 输入（无需读缓存）：
  conda run -n mcts_debug python selftest_mem.py --dummy

说明：优先用真实 batch（cache/meta_2000_vmap.pt via make_loaders，含矢量地图 GT），让各
任务 loss 量级真实；若 --dummy 或缓存不可用则回退到形状匹配的随机输入 + 合成 GT。backbone 冻结
（与训练一致），不改动模型/训练逻辑，仅做自测。
"""

import argparse

import torch

from dataset import (HEIGHTS, MAP_MAX, MAP_POINTS, SCENE_SCALE,
                     geometry_from_batch, make_loaders)
from model import NanoFSDv2
from train import (detection_loss, motion_loss, occupancy_loss, planning_loss,
                   vector_map_loss)

IMG_HW = (224, 400)
FUT_STEPS = 12
BUDGET_GIB = 24.0                 # 单卡显存预算


def _synthetic_batch(B, dev):
    """构造形状匹配的随机 dummy 输入 + 合成 GT（无缓存时的回退）。

    形状对齐 dataset / model：images (B,6,3,224,400)，相机参数 K/R/t，GT 轨迹/占据/
    可行驶区/结构化 agent 列表（MAX_GT=50）。uv/valid 由 geometry_from_batch 用随机标定
    算出（几何可见占比随机，不影响显存/反传链路的自测目的）。
    """
    V, H, W = 6, 224, 400
    S, Z, Tf, N = 100, len(HEIGHTS), FUT_STEPS, 50

    images = torch.rand(B, V, 3, H, W, device=dev)
    # 随机但量级合理的相机内外参（只为跑通 geometry_from_batch；可见性不是本测重点）
    K = torch.eye(3, device=dev).view(1, 1, 3, 3).repeat(B, V, 1, 1).clone()
    K[..., 0, 0] = 500.0; K[..., 1, 1] = 500.0
    K[..., 0, 2] = W / 2; K[..., 1, 2] = H / 2
    R = torch.eye(3, device=dev).view(1, 1, 3, 3).repeat(B, V, 1, 1).contiguous()
    t = torch.zeros(B, V, 3, device=dev)
    uv, valid = geometry_from_batch(K, R, t, IMG_HW, dev)

    batch = {
        "images": images,
        "traj_gt": torch.randn(B, Tf, 2, device=dev) * 0.1,
        "traj_mask": torch.ones(B, Tf, dtype=torch.bool, device=dev),
        "occ": (torch.rand(B, S, S, device=dev) > 0.98).float(),
        "drivable": (torch.rand(B, S, S, device=dev) > 0.5).float(),
        "command": torch.randint(0, 3, (B,), device=dev),
        "agent_box": torch.randn(B, N, 5, device=dev) * 0.2,
        "agent_cls": torch.randint(0, 2, (B, N), device=dev),
        "agent_mask": torch.rand(B, N, device=dev) > 0.6,
        "agent_fut": torch.randn(B, N, Tf, 2, device=dev) * 0.1,
        "agent_fut_mask": torch.rand(B, N, Tf, device=dev) > 0.3,
        # 矢量地图 GT（VectorMapHead 监督）：折线点 (B,MAP_MAX,P,2) 归一化坐标 ~0.2 尺度、
        # 类别 (B,MAP_MAX) 取 0/1/2、有效折线 mask (B,MAP_MAX) 随机 bool
        "map_pts": torch.randn(B, MAP_MAX, MAP_POINTS, 2, device=dev) * 0.2,
        "map_cls": torch.randint(0, 3, (B, MAP_MAX), device=dev),
        "map_poly_mask": torch.rand(B, MAP_MAX, device=dev) > 0.5,
    }
    return batch, uv, valid


def _real_batch(meta, B, dev):
    """从缓存取一个真实 batch（batch_size=B），返回 batch 字典 + uv/valid。"""
    train, _val, (ntr, _nva) = make_loaders(meta, batch_size=B, val_ratio=0.2,
                                            seed=0, num_workers=0)
    batch = next(iter(train))
    batch = {k: v.to(dev) for k, v in batch.items()}
    uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
    return batch, uv, valid


def multitask_loss(out, batch):
    """全头多任务总损失（与 train.train_multitask 保守默认权重一致）：
       L = 1.0·L_plan + 1.0·L_occ + 1.0·L_map + 0.5·L_det + 1.0·L_mot + 0.5·L_vmap。
    """
    l_plan = planning_loss(out["traj"], batch["traj_gt"], batch["traj_mask"])
    l_occ = occupancy_loss(out["occ"], batch["occ"])
    l_map = occupancy_loss(out["map"], batch["drivable"])
    det = out["det"]
    l_det = detection_loss(det, batch["agent_box"], batch["agent_cls"], batch["agent_mask"])
    l_mot = motion_loss(det["agent_fut"], det, batch["agent_fut"],
                        batch["agent_fut_mask"], batch["agent_box"], batch["agent_mask"])
    # 矢量地图头：匹配后 折线回归 + 类别 CE + 存在性 BCE（权重 0.5，对齐 --vmap-weight 默认）
    l_vmap = vector_map_loss(out["vmap"], batch["map_pts"], batch["map_cls"],
                             batch["map_poly_mask"])
    total = l_plan + l_occ + l_map + 0.5 * l_det + 1.0 * l_mot + 0.5 * l_vmap
    return total, {"plan": l_plan.item(), "occ": l_occ.item(), "map": l_map.item(),
                   "det": l_det.item(), "mot": l_mot.item(), "vmap": l_vmap.item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="cache/meta_2000_vmap.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dummy", action="store_true",
                        help="用随机 dummy 输入而非真实缓存 batch")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "本自测需要 CUDA（显存测量）"
    dev = "cuda"
    torch.manual_seed(0)
    B = args.batch_size

    # ---- 取输入（优先真实 batch）----------------------------------------------------
    if args.dummy:
        print(f"[selftest] 使用随机 dummy 输入 (B={B})", flush=True)
        batch, uv, valid = _synthetic_batch(B, dev)
    else:
        try:
            print(f"[selftest] 从缓存取真实 batch: {args.meta} (B={B})", flush=True)
            batch, uv, valid = _real_batch(args.meta, B, dev)
        except Exception as e:  # noqa: BLE001
            print(f"[selftest] 读缓存失败（{e}），回退随机 dummy 输入", flush=True)
            batch, uv, valid = _synthetic_batch(B, dev)

    # ---- 构建全头模型（与训练一致：backbone 冻结、规划 v2）----------------------------
    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      use_planner_v2=True, use_vector_map=True,
                      pretrained=True, freeze_backbone=True).to(dev)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

    # ---- 全头前向 + 反向，测峰值显存 ------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    out = model(batch["images"], uv, valid, command=batch["command"])
    total, parts = multitask_loss(out, batch)
    opt.zero_grad()
    total.backward()
    opt.step()
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gib = peak_bytes / 1024 ** 3
    peak_mib = peak_bytes / 1024 ** 2

    print(f"[selftest] 全头前向+反向完成 (bs={B})", flush=True)
    print(f"[selftest]   total_loss={total.item():.6f}  "
          f"plan={parts['plan']:.4f} occ={parts['occ']:.4f} map={parts['map']:.4f} "
          f"det={parts['det']:.4f} mot={parts['mot']:.6f} vmap={parts['vmap']:.4f}", flush=True)
    print(f"[selftest]   峰值显存: {peak_mib:.1f} MiB = {peak_gib:.3f} GiB "
          f"(预算 {BUDGET_GIB:.0f}G)", flush=True)

    assert torch.isfinite(total), "总损失非有限（NaN/Inf），反传链路异常"
    assert peak_gib <= BUDGET_GIB, (
        f"峰值显存 {peak_gib:.3f} GiB 超出单卡预算 {BUDGET_GIB:.0f}G")
    print(f"[selftest]   ✓ 峰值显存 ≤ {BUDGET_GIB:.0f}G", flush=True)

    # ---- 需求 4.4：规划退化（agent 全背景）轻量校验 ----------------------------------
    # 复用已算好的 BEV / 检测 query_emb，但把 score 全压成很负（sigmoid→0，等价全背景），
    # 直接调 planner_v2，确认退化为仅 BEV 规划、输出有限（无 NaN/Inf），不报错。
    with torch.no_grad():
        bev = model.bev(model.backbone(batch["images"]), uv, valid)   # (B,D,S,S)
        det = model.detection(bev)
        cmd_emb = model.cmd_emb(batch["command"])

        # 正常 score 下的规划（作对照）
        traj_normal = model.planner_v2(bev, det["query_emb"], det["score"], cmd_emb)

        # agent 全背景：score 全设 -50（sigmoid≈0 -> gate≈0 -> 退化为仅 BEV）
        bg_score = torch.full_like(det["score"], -50.0)
        traj_bg = model.planner_v2(bev, det["query_emb"], bg_score, cmd_emb)

    assert torch.isfinite(traj_bg).all(), "agent 全背景下 traj 含 NaN/Inf，退化不安全"
    assert torch.isfinite(traj_normal).all(), "正常 score 下 traj 含 NaN/Inf"
    max_abs = traj_bg.abs().max().item()
    print(f"[selftest]   规划退化检查(需求4.4): agent 全背景 traj 有限 ✓ "
          f"(max|traj|={max_abs:.3f}, 形状={tuple(traj_bg.shape)})", flush=True)

    print("[selftest] 全部通过 ✓", flush=True)


if __name__ == "__main__":
    main()
