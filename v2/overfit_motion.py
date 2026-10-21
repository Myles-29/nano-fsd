"""检测 + 运动头联合过拟合 sanity check（任务 3.4 / 正确性属性 7：过拟合可行性）。

从带 agent 真值的缓存（cache/meta_2000_agent.pt）取**一个固定样本**，反复**联合训练检测
loss + 运动 loss**，确认两者都下降、尤其**运动 loss 显著下降** => 「图像->BEV->检测头->
运动头」整条梯度链路通畅、可训练。

为什么要联合（而非只开运动 loss）：运动头从检测头的 query_emb 回归轨迹，而运动监督只在
「预测框匹配到 GT」的 query 上计算（motion_loss 内部用 det['box'] 重算最近距离匹配）。
若不同时训练检测头，预测框会乱飘、匹配不稳定，运动监督信号跟着抖动。先让检测把框压到
GT 附近，匹配固定下来，运动 loss 才能稳定下压——这正是 UniAD「检测 query 承接运动」的
链路：检测头告诉 query「我在哪」，运动头让同一 query 产出「我要去哪」。

运动输出接入方式（任务 3.4）：模型 forward 返回 dict，其 dict['det'] 现在额外带
det['agent_fut'] (B,N,Tf,2)，由顶层 self.motion(det['query_emb']) 算得。本脚本直接从 det
取 agent_fut，配合 train.py 的 motion_loss（内部用同一套匹配）计算运动损失。

坐标/单位沿用全局约定：ego 系、SCENE_SCALE=50 归一化、MAX_GT=N=50、n_cls=2、Tf=12。
"""

import argparse

import torch

from dataset import NuScenesBEVDataset, HEIGHTS, collate, geometry_from_batch
from model import NanoFSDv2
from train import detection_loss, motion_loss

IMG_HW = (224, 400)
FUT_STEPS = 12


def _pick_sample_with_agents(ds, want_idx):
    """选一个有有效 agent 且有有效未来步的样本：指定 --sample-idx 则直接用；否则扫描找
    第一个 agent_mask 有有效位、且 agent_fut_mask 有有效步的样本（否则运动 loss 没有正
    样本，联合过拟合不出运动信号）。"""
    if want_idx is not None:
        return want_idx
    for i in range(len(ds)):
        s = ds[i]
        if bool(s["agent_mask"].any()) and bool(s["agent_fut_mask"].any()):
            return i
    # 退而求其次：只要有有效 agent
    for i in range(len(ds)):
        if bool(ds[i]["agent_mask"].any()):
            return i
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="cache/meta_2000_agent.pt")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="固定过拟合的样本下标；默认自动找第一个含有效 agent + 未来步的样本")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[overfit-motion] device={dev}；加载缓存 {args.meta} ...", flush=True)

    ds = NuScenesBEVDataset(args.meta)
    idx = _pick_sample_with_agents(ds, args.sample_idx)
    sample = ds[idx]
    n_valid_gt = int(sample["agent_mask"].sum().item())
    n_valid_step = int(sample["agent_fut_mask"].sum().item())
    print(f"[overfit-motion] 固定样本 idx={idx}，有效 GT agent 数 M={n_valid_gt}，"
          f"有效未来步总数={n_valid_step}", flush=True)
    if n_valid_gt == 0:
        print("[overfit-motion] 警告：该样本无有效 agent，检测/运动 loss 无正样本，过拟合无意义。")
    if n_valid_step == 0:
        print("[overfit-motion] 警告：该样本无有效未来步，运动 loss 无信号。")

    # 组装成单样本 batch（B=1），device 上预算几何采样网格（固定输入，训练中不变）
    batch = collate([sample])
    images = batch["images"].to(dev)                                   # (1,6,3,H,W)
    uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
    command = batch["command"].to(dev) if "command" in batch else None
    gt_box = batch["agent_box"].to(dev)                                # (1,MAX_GT,5)
    gt_cls = batch["agent_cls"].to(dev)                                # (1,MAX_GT)
    gt_mask = batch["agent_mask"].to(dev)                              # (1,MAX_GT)
    gt_fut = batch["agent_fut"].to(dev)                                # (1,MAX_GT,Tf,2)
    gt_fut_mask = batch["agent_fut_mask"].to(dev)                      # (1,MAX_GT,Tf)

    # 模型 + 优化器（backbone 冻结；训 BEV 编码器 + 检测头 + 运动头等可训练参数）
    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6,
                      fut_steps=FUT_STEPS, n_query=50, n_cls=2,
                      pretrained=True, freeze_backbone=True).to(dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    # 余弦退火：后期降 lr 让贪心匹配稳定（预测框不再大跳、query 与 GT 配对固定），
    # 检测/运动的回归目标才不反复切换，loss 能压到底。
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(f"[overfit-motion] 开始单样本联合过拟合 {args.steps} 步（检测 + 运动 loss）...",
          flush=True)
    first_det = first_mot = None
    best_det = best_mot = float("inf")
    last_det = last_mot = None
    for step in range(args.steps):
        # 只取 det 输出（含 agent_fut），忽略 traj/occ/drivable（只开检测 + 运动 loss）
        det = model(images, uv, valid, command=command)["det"]
        l_det = detection_loss(det, gt_box, gt_cls, gt_mask)
        l_mot = motion_loss(det["agent_fut"], det, gt_fut, gt_fut_mask, gt_box, gt_mask)
        loss = l_det + l_mot                       # 联合（等权；运动量级小不盖过检测）
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        dv, mv = l_det.item(), l_mot.item()
        if first_det is None:
            first_det, first_mot = dv, mv
        last_det, last_mot = dv, mv
        best_det = min(best_det, dv)
        best_mot = min(best_mot, mv)
        if (step + 1) % 50 == 0 or step == 0:
            print(f"  step {step+1:>4}/{args.steps}  det_loss={dv:.6f} (best {best_det:.6f})  "
                  f"mot_loss={mv:.6f} (best {best_mot:.6f})", flush=True)

    det_drop = (first_det - best_det) / max(abs(first_det), 1e-8) * 100.0
    mot_drop = (first_mot - best_mot) / max(abs(first_mot), 1e-8) * 100.0
    print(f"[overfit-motion] 检测 loss：初始 {first_det:.6f} -> 最终 {last_det:.6f} "
          f"（最优 {best_det:.6f}，相对初始下降 {det_drop:.1f}%）", flush=True)
    print(f"[overfit-motion] 运动 loss：初始 {first_mot:.6f} -> 最终 {last_mot:.6f} "
          f"（最优 {best_mot:.6f}，相对初始下降 {mot_drop:.1f}%）", flush=True)

    # 属性 7：固定 batch 反复训练，运动 loss 显著下降（相对初始下降 ≥50% 视为通过）。
    # 用 best 判定——贪心匹配带来的 loss 抖动不影响「梯度链路是否通畅」的结论。
    mot_ok = best_mot < first_mot * 0.5
    det_ok = best_det < first_det * 0.5
    print(f"[overfit-motion] 判定：检测 loss {'显著下降' if det_ok else '下降不足'}；"
          f"运动 loss {'显著下降' if mot_ok else '下降不足'}", flush=True)
    verdict = ("运动头梯度链路通畅、可过拟合（属性 7 通过）" if mot_ok
               else "运动 loss 下降不足，需检查梯度链路")
    print(f"[overfit-motion] 属性 7（运动）：{verdict}", flush=True)


if __name__ == "__main__":
    main()
