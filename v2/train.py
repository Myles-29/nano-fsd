"""v2 端到端训练：图像 -> BEV -> 轨迹，用 planning loss 全量/子集训练。

沿用 v1 风格：固定 seed、train/val 划分、每 epoch 评估规划 ADE(米)、按 val ADE 存
最优权重、带时间戳实验目录、训练后自动可视化。backbone 默认冻结（省显存/时间）。

记号：ADE = 有效未来步上预测与真值轨迹的平均欧氏距离（米）。
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from dataset import SCENE_SCALE, geometry_from_batch, make_loaders
from model import NanoFSDv2

IMG_HW = (224, 400)
FUT_STEPS = 12


import sys

import torch.nn.functional as F


class _Tee:
    """stdout 同时写终端 + 文件，用于把训练日志落盘到 run_dir/train.log。"""

    def __init__(self, *streams):
        self._s = streams

    def write(self, data):
        for s in self._s:
            s.write(data); s.flush()
        return len(data)

    def flush(self):
        for s in self._s:
            s.flush()


def _build_optimizer(model, lr):
    """构造 AdamW。解冻 backbone 时 backbone 用更小 lr（lr/10）微调，避免大 lr 破坏
    ImageNet 预训练特征；其余头/编码器用主 lr。冻结时 backbone 不进优化器（无梯度）。"""
    bb_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if name.startswith("backbone.") else head_params).append(p)
    groups = [{"params": head_params, "lr": lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": lr * 0.1})
        print(f"[opt] backbone 解冻微调：{len(bb_params)} 组参数用 lr={lr*0.1:g}", flush=True)
    return torch.optim.AdamW(groups)


def planning_loss(pred, gt, mask):
    """masked MSE：只在有效未来步上算。pred/gt (B,Tf,2), mask (B,Tf)。"""
    err = ((pred - gt) ** 2).sum(-1)                 # (B,Tf)
    m = mask.float()
    return (err * m).sum() / m.sum().clamp(min=1)


def occupancy_loss(occ_logit, occ_gt):
    """占据头 BCE。occ_logit (B,1,S,S), occ_gt (B,S,S) in {0,1}。

    占据很稀疏(~1%)，给正类加权（pos_weight）平衡，否则模型会全预测 0。
    """
    occ_gt = occ_gt.unsqueeze(1)                     # (B,1,S,S)
    pos = occ_gt.mean().clamp(min=1e-4)
    pos_weight = ((1 - pos) / pos).clamp(max=100.0)  # 正类稀疏 -> 加权
    return F.binary_cross_entropy_with_logits(
        occ_logit, occ_gt, pos_weight=pos_weight)


@torch.no_grad()
def match_pred_to_gt(pred_box, gt_box, gt_mask, anchor=None):
    """最近距离贪心匹配（单帧 / per-sample）。

    把每个**有效** GT 贪心指派给一个最近且尚未被占用的 query。距离基准二选一：
      - anchor 不为 None（**推荐，破死锁**）：用每个 query 的**固定锚点** (N,2) 到 GT 的
        距离。锚点不动 → 匹配从第一轮就稳定 → query 学出空间分工，框回归可收敛。
      - anchor 为 None（向后兼容）：退回用**预测框** pred_box 的 (x,y) 到 GT 的距离
        （DETR 式，训练初期预测乱会导致匹配抖动——anchor-based 检测头应传 anchor）。
    未被指派的 query 即背景负样本（由上层 detection_loss 处理，本函数只返回正匹配对）。

    契约（signature contract）：**per-sample，一帧一调用**。
      - pred_box: (N,5) 单帧预测框 `[x,y,yaw,w,l]`（本函数只用前 2 列 x,y 算距离）。
      - gt_box:   (MAX_GT,5) 单帧 GT 框，同样只用前 2 列。
      - gt_mask:  (MAX_GT,) bool，有效 GT 位。
    上层 detection_loss / motion_loss 对 batch 维逐帧循环调用本函数。选择 per-sample 而
    非 batch 维，是因为每帧有效 GT 数 M 不同、匹配结果长度不定，per-sample 契约最干净。

    返回：
      (matched_pred_idx, matched_gt_idx)：两个等长 1D long 张量，长度 = 匹配上的有效 GT
      数（≤ min(M, N)）。matched_gt_idx[k] 这个 GT 被指派给了 matched_pred_idx[k] 这个
      预测 query。M=0（无有效 GT）时返回两个空张量。

    贪心策略：按「GT 到自车(原点)距离近者优先」的顺序逐个指派——近处 agent 更重要，先
    为其抢占最近的 query，缓解贪心的次优性（远处 agent 退而求其次）。
    """
    device = pred_box.device
    empty = torch.empty(0, dtype=torch.long, device=device)

    valid_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()  # (M,) 有效 GT 的原始下标
    M = int(valid_idx.numel())
    N = int(pred_box.shape[0])
    if M == 0 or N == 0:
        return empty, empty

    # 距离基准：优先用固定锚点（稳定），否则退回预测框中心（兼容旧 DETR 式检测头）
    ref_xy = anchor[:, :2] if anchor is not None else pred_box[:, :2]  # (N,2)
    gt_xy = gt_box[valid_idx, :2]                 # (M,2) 只取有效 GT

    # 距离矩阵 (M,N)：每个有效 GT 到每个 query 参考点(锚点/预测框)的中心欧氏距离
    dist = torch.cdist(gt_xy, ref_xy)             # (M,N)

    # GT 指派顺序：到自车(ego 原点)近者优先
    gt_to_ego = gt_xy.pow(2).sum(-1).sqrt()       # (M,)
    order = torch.argsort(gt_to_ego)              # (M,) 处理顺序

    occupied = torch.zeros(N, dtype=torch.bool, device=device)
    matched_pred, matched_gt = [], []
    for m in order.tolist():
        d = dist[m].clone()
        d[occupied] = float("inf")                # 已占用的 query 不可再选
        p = int(torch.argmin(d).item())
        if not torch.isfinite(d[p]):              # 所有 query 都被占用（M>N 时才可能）
            break
        occupied[p] = True
        matched_pred.append(p)
        matched_gt.append(int(valid_idx[m].item()))

    if not matched_pred:
        return empty, empty
    return (torch.tensor(matched_pred, dtype=torch.long, device=device),
            torch.tensor(matched_gt, dtype=torch.long, device=device))


def detection_loss(det, gt_box, gt_cls, gt_mask):
    """检测损失：匹配后 框回归(smooth L1) + 类别 CE + 存在性 BCE。

    对齐 design.md 组件 6 与需求 2.5/2.7。先用 match_pred_to_gt 把每帧 N 个预测 query 与
    有效 GT 配对，再在匹配上的 query 算框回归 + 类别 CE；存在性 BCE 对**全部** query 算
    （匹配上的 query 目标=1 表示"真有物体"，未匹配的目标=0 表示背景）。

    参数：
      det:      检测头输出 dict，含 box (B,N,5)、cls (B,N,n_cls)、score (B,N,1)。
      gt_box:   (B,MAX_GT,5) GT 框 `[x,y,yaw,w,l]`（ego 系，SCENE_SCALE 归一化）。
      gt_cls:   (B,MAX_GT) GT 类别 id（long）。
      gt_mask:  (B,MAX_GT) bool，有效 GT 位。

    空场景安全（需求 2.7）：当整个 batch 匹配总数 M=0 时，框/类别 loss 贡献 0（不产生
    NaN），存在性 BCE 仍对全部负样本正常计算；分母一律 clamp(min=1) 防除零。

    返回：可微标量（float tensor），= 框回归 + 类别 CE + 存在性 BCE。
    """
    pred_box = det["box"]                                 # (B,N,5)
    pred_cls = det["cls"]                                 # (B,N,n_cls)
    pred_score = det["score"].squeeze(-1)                 # (B,N)
    B, N = pred_score.shape
    device = pred_score.device

    # 框回归的「量纲归一化」（任务 2.5 KNOWN ISSUE 修正）：box 的 5 维量纲差异极大——
    # x,y 已被 SCENE_SCALE 归一化到约 ±1，yaw 是弧度 ±π，w/l 是米（约 0.5~7.5）。直接
    # 对原始误差做 smooth L1，会让 w/l 这种大数值维度主导梯度、x,y 这种小数值维度几乎
    # 学不动，拖慢收敛。做法：把每一维的误差除以一个「典型尺度」向量，使各维误差落到
    # 可比区间后再算 smooth L1。保持零点性质（pred==GT 时误差仍为 0）与函数签名不变。
    _BOX_SCALE = pred_box.new_tensor([1.0, 1.0, 3.14159265, 10.0, 10.0])  # [x,y,yaw,w,l]

    # 存在性 BCE 的目标：默认全 0（背景），匹配上的 query 置 1（前景）
    exist_target = torch.zeros(B, N, device=device)

    box_err_sum = pred_box.new_zeros(())                  # 框回归累加（smooth L1，求和后再平均）
    cls_err_sum = pred_box.new_zeros(())                  # 类别 CE 累加
    n_matched = 0                                         # batch 内匹配总数 M

    anchor = det.get("anchor_xy")                         # (N,2) 固定锚点；无则退回预测框匹配
    for b in range(B):
        pred_idx, gt_idx = match_pred_to_gt(pred_box[b], gt_box[b], gt_mask[b],
                                            anchor=anchor)
        m = int(pred_idx.numel())
        if m == 0:
            continue
        n_matched += m
        exist_target[b, pred_idx] = 1.0                   # 这些 query 是前景

        # 框回归：匹配 query 的 5 维框 vs 其配到的 GT 框。先按 _BOX_SCALE 把各维误差
        # 归一到可比区间（见上方说明），再算 smooth L1（按元素求和，末尾统一平均）。
        # 等价于对 (pred-gt)/scale 做 smooth_l1_loss(·, 0)：pred==gt 时仍精确为 0。
        p_box = pred_box[b, pred_idx]                     # (m,5)
        g_box = gt_box[b, gt_idx]                         # (m,5)
        box_diff = (p_box - g_box) / _BOX_SCALE           # 量纲归一化后的误差
        box_err_sum = box_err_sum + F.smooth_l1_loss(
            box_diff, torch.zeros_like(box_diff), reduction="sum")

        # 类别 CE：匹配 query 的类别 logit vs GT 类别 id
        p_cls = pred_cls[b, pred_idx]                     # (m,n_cls)
        g_cls = gt_cls[b, gt_idx].long()                  # (m,)
        cls_err_sum = cls_err_sum + F.cross_entropy(p_cls, g_cls, reduction="sum")

    # 匹配上的部分：按匹配总数平均（M=0 时贡献 0，分母 clamp 防除零/NaN）
    denom = max(n_matched, 1)
    l_box = box_err_sum / denom
    l_cls = cls_err_sum / denom

    # 存在性 BCE：对全部 B*N 个 query 算（前景目标=1、背景目标=0），始终有效
    l_exist = F.binary_cross_entropy_with_logits(pred_score, exist_target)

    return l_box + l_cls + l_exist


def polyline_distance(pred_pts, gt_pts):
    """两条折线的「顺序鲁棒逐点 L2 均值」（对齐 design.md 组件 5、决策 4，需求 6.1/6.5）。

    道路线元素（车道分隔线 / 道路边界 / 人行横道边界）**无固有方向**：同一条线正着存还是
    倒着存都对应同一几何，强制点序会制造虚假大误差。故本距离对 GT 取「正序对齐」与「GT 点序
    翻转对齐」两种逐点 L2 均值的**较小者**，让距离对折线整体朝向不敏感，但仍惩罚形状偏差。

    契约：
      - pred_pts / gt_pts: (P,2) 两条等长折线（P 个有序点，SCENE_SCALE 归一化 ego 系 xy）。
        （也可批量传入 (...,P,2)：对最后两维算，返回 (...) 的距离，供向量化场景复用；
         但 loss 与匹配只用 (P,2)→标量这一契约。）
    返回：
      标量（或 batch 形状）距离 = min( mean‖pred-gt‖, mean‖pred-reverse(gt)‖ )。
      pred==gt 时精确为 0（翻转取小不改变零点）；对 pred 可微（匹配时在 no_grad 下调用）。
    """
    # 逐点欧氏距离的数值安全写法：sqrt 在 0 处导数无穷，当 pred==gt（每点平方和=0）时会产生
    # NaN 梯度。用一个极小 eps 做下界 clamp 后再 sqrt——eps=1e-12 使 sqrt(eps)=1e-6 量级，
    # 对距离**值**的影响在容差内（零点仍"精确为 0"到 1e-6），却让梯度处处有限（反传安全）。
    def _pointwise_mean_l2(p, g):
        sq = (p - g).pow(2).sum(-1)                                    # (...,P) 每点平方距离
        # sqrt(sq+eps)：eps 极小（1e-16），零点处值≈1e-8（容差内可视作 0），梯度 x/√(x²+eps)
        # 处处有限（x=0 时为 0），既保零点又保反传安全。
        return (sq + 1e-16).sqrt().mean(-1)                            # (...) 点维均值
    # 正序对齐
    d_fwd = _pointwise_mean_l2(pred_pts, gt_pts)                       # (...) 正序逐点 L2 均值
    # 反序对齐：翻转 GT 的点序（倒数第二维 P）
    d_rev = _pointwise_mean_l2(pred_pts, gt_pts.flip(-2))              # (...) 反序逐点 L2 均值
    return torch.minimum(d_fwd, d_rev)


@torch.no_grad()
def match_polyline_to_gt(pred_pts, gt_pts, gt_mask, gt_cls=None, pred_cls=None,
                         anchor=None):
    """折线最近距离贪心匹配（单帧 / per-sample），与 match_pred_to_gt 刻意同构。

    把每个**有效** GT 折线贪心指派给一个最近且尚未被占用的 map query。距离基准二选一：
      - anchor 不为 None（**推荐，破死锁**，与检测头一致）：用每个 query 的**固定锚点**
        (M,2) 到每条有效 GT 折线**质心**（P 个点的均值）的距离。锚点不动 → 匹配从第一轮
        就稳定 → query 学出空间分工，折线回归可收敛。
      - anchor 为 None（更精细但初期抖动）：用 polyline_distance 逐 (gt,query) 对算顺序
        鲁棒折线距离。
    未被指派的 query 即背景负样本（由上层 vector_map_loss 的存在性 BCE 处理，本函数只返回
    正匹配对）。

    契约（signature contract）：**per-sample，一帧一调用**，镜像 match_pred_to_gt。
      - pred_pts: (M,P,2) 单帧预测折线。
      - gt_pts:   (MAX_MAP,P,2) 单帧 GT 折线。
      - gt_mask:  (MAX_MAP,) bool，有效 GT 位。
      - gt_cls / pred_cls（可选，开启类内匹配）：gt_cls (MAX_MAP,) GT 类别 id；
        pred_cls (M,n_cls) query 类别 logit（取 argmax 作预测类）。两者都给定时，把
        不同类的 (gt,query) 距离置为 inf，仅同类可配。
      - anchor（可选）：(M,2) 固定锚点坐标；给定则用锚点-质心距离，否则用 polyline_distance。
    上层 vector_map_loss 对 batch 维逐帧循环调用本函数。

    返回：
      (matched_pred_idx, matched_gt_idx)：两个等长 1D long 张量，长度 = 匹配上的有效 GT 数
      （≤ min(M_gt, M)）。matched_gt_idx[k] 这个 GT 被指派给了 matched_pred_idx[k] 这个
      query。M_gt=0（无有效 GT）时返回两个空张量。

    贪心策略：按「GT 到自车(原点)距离近者优先」的顺序逐个指派——近处地图元素更重要，先为其
    抢占最近的 query，缓解贪心的次优性（远处元素退而求其次）。
    """
    device = pred_pts.device
    empty = torch.empty(0, dtype=torch.long, device=device)

    valid_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()   # (M_gt,) 有效 GT 的原始下标
    M_gt = int(valid_idx.numel())
    M = int(pred_pts.shape[0])
    if M_gt == 0 or M == 0:
        return empty, empty

    gt_valid = gt_pts[valid_idx]                   # (M_gt,P,2) 只取有效 GT
    gt_centroid = gt_valid.mean(dim=1)             # (M_gt,2) 每条 GT 折线质心

    # 距离矩阵 (M_gt, M)：每个有效 GT 到每个 query 的距离，基准二选一
    if anchor is not None:
        # 锚点(固定) 到 GT 质心的欧氏距离——稳定，匹配从第一轮就定下来
        dist = torch.cdist(gt_centroid, anchor[:, :2])             # (M_gt,M)
    else:
        # 顺序鲁棒折线距离：对每个 (gt,query) 对算 polyline_distance（P 小，开销可忽略）
        # 广播成 (M_gt,M,P,2) 后在点维上算正/反序取小
        pg = gt_valid.unsqueeze(1)                                 # (M_gt,1,P,2)
        pp = pred_pts.unsqueeze(0)                                 # (1,M,P,2)
        dist = polyline_distance(pp, pg)                           # (M_gt,M)

    # 可选类内匹配：不同类的 (gt,query) 距离置 inf，仅同类可配
    if gt_cls is not None and pred_cls is not None:
        gt_c = gt_cls[valid_idx].long()                            # (M_gt,)
        pred_c = pred_cls.argmax(dim=-1).long()                    # (M,) 预测类别
        cross = gt_c.unsqueeze(1) != pred_c.unsqueeze(0)           # (M_gt,M) 异类位
        dist = dist.masked_fill(cross, float("inf"))

    # GT 指派顺序：到自车(ego 原点)近者优先
    gt_to_ego = gt_centroid.pow(2).sum(-1).sqrt()  # (M_gt,)
    order = torch.argsort(gt_to_ego)               # (M_gt,) 处理顺序

    occupied = torch.zeros(M, dtype=torch.bool, device=device)
    matched_pred, matched_gt = [], []
    for m in order.tolist():
        d = dist[m].clone()
        d[occupied] = float("inf")                 # 已占用的 query 不可再选
        p = int(torch.argmin(d).item())
        if not torch.isfinite(d[p]):               # 无可配 query（M_gt>M 或类内无同类）
            continue
        occupied[p] = True
        matched_pred.append(p)
        matched_gt.append(int(valid_idx[m].item()))

    if not matched_pred:
        return empty, empty
    return (torch.tensor(matched_pred, dtype=torch.long, device=device),
            torch.tensor(matched_gt, dtype=torch.long, device=device))


def vector_map_loss(vmap, gt_pts, gt_cls, gt_mask):
    """矢量地图损失：匹配后 折线回归(顺序鲁棒 L2) + 类别 CE + 存在性 BCE。

    对齐 design.md 组件 5 与需求 6.1~6.4/6.7，与 detection_loss **刻意对称**：先用
    match_polyline_to_gt 把每帧 M 个 map query 与有效 GT 折线配对，再在匹配上的 query 算
    折线回归 + 类别 CE；存在性 BCE 对**全部** query 算（匹配上的 query 目标=1 表示"真有一条
    地图元素"，未匹配的目标=0 表示背景）。

    与 detection_loss 的唯一实质差异：**回归项不做量纲归一化**——折线点与 agent xy 同为
    SCENE_SCALE 归一化坐标、量纲一致（不像 box 的 x,y/yaw/w,l 五维量纲悬殊），直接用
    polyline_distance（已含正/反序取小的顺序鲁棒）求和、按匹配总数平均即可。

    参数：
      vmap:    地图头输出 dict，含 pts (B,M,P,2)、cls (B,M,n_cls)、score (B,M,1)[,anchor_xy (M,2)]。
      gt_pts:  (B,MAX_MAP,P,2) GT 折线（ego 系，SCENE_SCALE 归一化）。
      gt_cls:  (B,MAX_MAP) GT 类别 id（long）。
      gt_mask: (B,MAX_MAP) bool，有效 GT 折线位。

    空场景安全（需求 6.7）：当整个 batch 匹配总数=0 时，折线/类别 loss 贡献 0（不产生 NaN），
    存在性 BCE 仍对全部负样本正常计算；分母一律 clamp(min=1) 防除零。

    返回：可微标量（float tensor），= 折线回归 + 类别 CE + 存在性 BCE。
    """
    pred_pts = vmap["pts"]                                 # (B,M,P,2)
    pred_cls = vmap["cls"]                                 # (B,M,n_cls)
    pred_score = vmap["score"].squeeze(-1)                 # (B,M)
    B, M = pred_score.shape
    device = pred_score.device

    # 存在性 BCE 的目标：默认全 0（背景），匹配上的 query 置 1（前景）
    exist_target = torch.zeros(B, M, device=device)

    reg_err_sum = pred_pts.new_zeros(())                   # 折线回归累加（polyline_distance 求和后再平均）
    cls_err_sum = pred_pts.new_zeros(())                   # 类别 CE 累加
    n_matched = 0                                          # batch 内匹配总数

    anchor = vmap.get("anchor_xy")                         # (M,2) 固定锚点；无则退回 polyline_distance 匹配
    for b in range(B):
        pred_idx, gt_idx = match_polyline_to_gt(pred_pts[b], gt_pts[b], gt_mask[b],
                                                anchor=anchor)
        m = int(pred_idx.numel())
        if m == 0:
            continue
        n_matched += m
        exist_target[b, pred_idx] = 1.0                    # 这些 query 是前景

        # 折线回归：匹配 query 的折线 vs 其配到的 GT 折线，逐对算顺序鲁棒 polyline_distance，
        # 按匹配对求和（末尾统一按匹配总数平均）。pred==gt 时精确为 0（需求 6.4）。
        p_pts = pred_pts[b, pred_idx]                      # (m,P,2)
        g_pts = gt_pts[b, gt_idx]                          # (m,P,2)
        reg_err_sum = reg_err_sum + polyline_distance(p_pts, g_pts).sum()

        # 类别 CE：匹配 query 的类别 logit vs GT 类别 id
        p_cls = pred_cls[b, pred_idx]                      # (m,n_cls)
        g_cls = gt_cls[b, gt_idx].long()                   # (m,)
        cls_err_sum = cls_err_sum + F.cross_entropy(p_cls, g_cls, reduction="sum")

    # 匹配上的部分：按匹配总数平均（匹配总数=0 时贡献 0，分母 clamp 防除零/NaN）
    denom = max(n_matched, 1)
    l_reg = reg_err_sum / denom
    l_cls = cls_err_sum / denom

    # 存在性 BCE：对全部 B*M 个 query 算（前景目标=1、背景目标=0），始终有效
    l_exist = F.binary_cross_entropy_with_logits(pred_score, exist_target)

    return l_reg + l_cls + l_exist


def motion_loss(agent_fut_pred, det, gt_fut, gt_fut_mask, gt_box, gt_mask):
    """运动损失：只在匹配上的 query、有效未来步算 masked MSE。

    对齐 design.md 组件 6 与需求 3.3。运动预测必须和检测用**同一套匹配**，否则第 i 个
    query 的轨迹会对错 GT。为此本函数内部用 det['box'] 与 gt_box/gt_mask 逐帧重算
    match_pred_to_gt（与 detection_loss 完全一致的 per-sample 贪心匹配），避免把一个
    预计算的 match 对象在调用链里来回传递——这是比 design 原型里 `match` 入参更干净的
    契约（signature contract）。

    契约（OPTION A，推荐签名）：
      agent_fut_pred: (B,N,Tf,2) 运动头输出，相对各 agent 当前中心（第 i 个 query 的轨迹
                      与其检测框顺序对齐）。
      det:            检测头输出 dict，本函数只用 det['box'] (B,N,5) 做匹配（与
                      detection_loss 读同一字段，保证两头匹配一致）。
      gt_fut:         (B,MAX_GT,Tf,2) GT 未来相对轨迹（SCENE_SCALE 归一化）。
      gt_fut_mask:    (B,MAX_GT,Tf) bool，未来有效步（instance 中途消失则对应步 False）。
      gt_box:         (B,MAX_GT,5) GT 框，用于匹配。
      gt_mask:        (B,MAX_GT) bool，有效 GT 位，用于匹配。

    空场景安全（需求 3.3 / 属性 6）：当整个 batch 没有任何匹配、或匹配上的 GT 没有任何
    有效未来步时，返回 0（可微标量，不产生 NaN）；分母一律 clamp(min=1) 防除零。

    返回：可微标量（float tensor），= 匹配 query 在有效未来步上的 masked MSE。
    """
    pred_box = det["box"]                                 # (B,N,5)，仅用于匹配
    anchor = det.get("anchor_xy")                         # (N,2) 固定锚点；与 detection_loss 一致
    B = pred_box.shape[0]

    err_sum = agent_fut_pred.new_zeros(())                # 平方误差累加（末尾按有效步平均）
    n_valid = agent_fut_pred.new_zeros(())                # 有效 (step) 计数（含 xy 两维前）

    for b in range(B):
        # 与 detection_loss 完全相同的匹配（同样传 anchor）：保证第 i 个 query 的运动对到同一 GT
        pred_idx, gt_idx = match_pred_to_gt(pred_box[b], gt_box[b], gt_mask[b],
                                            anchor=anchor)
        if int(pred_idx.numel()) == 0:
            continue

        p_fut = agent_fut_pred[b, pred_idx]               # (m,Tf,2) 预测轨迹
        g_fut = gt_fut[b, gt_idx]                         # (m,Tf,2) 对应 GT 轨迹
        step_mask = gt_fut_mask[b, gt_idx].float()        # (m,Tf) 有效未来步

        # masked MSE：每步 xy 的平方误差，只在有效步累加
        se = ((p_fut - g_fut) ** 2).sum(-1)               # (m,Tf) 每步欧氏平方距离
        err_sum = err_sum + (se * step_mask).sum()
        n_valid = n_valid + step_mask.sum()

    return err_sum / n_valid.clamp(min=1)


@torch.no_grad()
def evaluate_detection(model, loader, device):
    """验证集平均检测 loss（仅检测头，用于 --det-only 的分步验证）。

    对每个 batch 调一次 forward 取 dict['det']，算 detection_loss，按 batch
    数平均。只读检测相关监督字段 agent_box/agent_cls/agent_mask。
    """
    model.eval()
    loss_sum, nb = 0.0, 0
    for batch in loader:
        images = batch["images"].to(device)
        uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, device)
        gt_box = batch["agent_box"].to(device)
        gt_cls = batch["agent_cls"].to(device)
        gt_mask = batch["agent_mask"].to(device)
        det = model(images, uv, valid, command=None)["det"]
        loss_sum += detection_loss(det, gt_box, gt_cls, gt_mask).item()
        nb += 1
    model.train()
    return loss_sum / max(nb, 1)


def train_detection_only(args, dev):
    """仅用检测 loss 在小子集上训练 BEV 编码器 + 检测头若干 epoch（演示入口，任务 2.6）。

    对齐 design.md 决策 6「先只开检测 loss 验证检测头能收敛」与需求 2.8（固定小子集反复
    训练检测 loss 显著下降）。backbone 默认冻结，优化除 backbone 外的所有可训练参数
    （BEV cross-attention + 检测头）。只读检测监督字段，不碰规划/占据/可行驶区 loss。
    产物：run_dir 下 checkpoint_best.pt（按 val 检测 loss 最优）、checkpoint_last.pt、
    history.json，便于 visualize.py --det 复现验证划分后叠加预测/GT 框。
    """
    from dataset import HEIGHTS

    train_loader, val_loader, (ntr, nva) = make_loaders(
        args.meta, batch_size=args.batch_size, val_ratio=args.val_ratio,
        seed=args.seed, num_workers=args.num_workers)
    print(f"[det] device={dev}  train={ntr} val={nva}  bs={args.batch_size} "
          f"lr={args.lr} epochs={args.epochs}  run_dir={args.run_dir}", flush=True)

    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      use_vector_map=args.use_vector_map,
                      pretrained=True, freeze_backbone=not args.unfreeze_backbone).to(dev)
    opt = _build_optimizer(model, args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_epoch = float("inf"), -1
    best_path = os.path.join(args.run_dir, "checkpoint_best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, nb = 0.0, 0
        for batch in train_loader:
            images = batch["images"].to(dev)
            uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
            gt_box = batch["agent_box"].to(dev)
            gt_cls = batch["agent_cls"].to(dev)
            gt_mask = batch["agent_mask"].to(dev)
            det = model(images, uv, valid, command=None)["det"]
            loss = detection_loss(det, gt_box, gt_cls, gt_mask)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); nb += 1
        train_loss = loss_sum / max(nb, 1)

        val_loss = evaluate_detection(model, val_loader, dev)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch + 1
            torch.save({"model": model.state_dict(),
                        "use_vector_map": args.use_vector_map}, best_path)
        print(f"[det] epoch {epoch+1:>3}/{args.epochs}  train_det_loss={train_loss:.6f}  "
              f"val_det_loss={val_loss:.6f}{' *' if improved else ''}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    torch.save({"model": model.state_dict(),
                "use_vector_map": args.use_vector_map},
               os.path.join(args.run_dir, "checkpoint_last.pt"))
    with open(os.path.join(args.run_dir, "history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch, "best_val_loss": best_loss,
                   "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"[det] 完成。最优 val_det_loss={best_loss:.6f} (epoch {best_epoch}) -> {best_path}")

    # 训练完自动可视化：用 best 权重对验证集叠加高置信度预测框 + GT 框（虚线）
    try:
        from visualize import run_detection_viz
        viz_dir = os.path.join(args.run_dir, "viz")
        print(f"[det] 自动可视化检测框（用最优权重）-> {viz_dir}", flush=True)
        run_detection_viz(best_path, meta=args.meta, num=6, seed=args.seed,
                          val_ratio=args.val_ratio, out_dir=viz_dir, score_thr=0.3)
    except Exception as e:  # noqa: BLE001
        print(f"[det] 自动可视化失败（不影响训练产物）: {e}", flush=True)


@torch.no_grad()
def evaluate_vectormap(model, loader, device):
    """验证集平均矢量地图 loss（仅矢量地图头，用于 --vmap-only 的分步验证）。

    与 evaluate_detection 刻意同构：对每个 batch 调一次 forward 取 dict['vmap']，算
    vector_map_loss，按 batch 数平均。只读矢量地图监督字段 map_pts/map_cls/
    map_poly_mask（分别对应 GT 折线点、类别 id、有效折线位）。
    """
    model.eval()
    loss_sum, nb = 0.0, 0
    for batch in loader:
        images = batch["images"].to(device)
        uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, device)
        gt_pts = batch["map_pts"].to(device)
        gt_cls = batch["map_cls"].to(device)
        gt_mask = batch["map_poly_mask"].to(device)
        vmap = model(images, uv, valid, command=None)["vmap"]
        loss_sum += vector_map_loss(vmap, gt_pts, gt_cls, gt_mask).item()
        nb += 1
    model.train()
    return loss_sum / max(nb, 1)


def train_vectormap_only(args, dev):
    """仅用矢量地图 loss 在小子集上训练 BEV 编码器 + 矢量地图头若干 epoch（演示入口，任务 5.1）。

    与 train_detection_only 刻意对称（检测头的孪生）：对齐 design.md 决策 6「先只开单头
    loss 验证该头能收敛」与需求 7.5/9.1。backbone 默认冻结，优化除 backbone 外的所有可
    训练参数（BEV cross-attention + 矢量地图头）。只读矢量地图监督字段，不碰规划/占据/
    可行驶区/检测/运动 loss。本路径**强制开启**矢量地图头（use_vector_map=True），因此即便
    命令行传了 --no-vector-map 也会被此入口覆盖为 True（否则 forward 不含 'vmap' 键无从训练）。
    产物：run_dir 下 checkpoint_best.pt（按 val 矢量地图 loss 最优）、checkpoint_last.pt、
    history.json，便于 visualize.py --vmap 复现验证划分后叠加预测/GT 折线。
    """
    from dataset import HEIGHTS

    train_loader, val_loader, (ntr, nva) = make_loaders(
        args.meta, batch_size=args.batch_size, val_ratio=args.val_ratio,
        seed=args.seed, num_workers=args.num_workers)
    print(f"[vmap] device={dev}  train={ntr} val={nva}  bs={args.batch_size} "
          f"lr={args.lr} epochs={args.epochs}  run_dir={args.run_dir}", flush=True)

    # 本入口强制开启矢量地图头（否则 forward 无 'vmap' 键无从监督），忽略 --no-vector-map
    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      use_vector_map=True,
                      pretrained=True, freeze_backbone=not args.unfreeze_backbone).to(dev)
    opt = _build_optimizer(model, args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_epoch = float("inf"), -1
    best_path = os.path.join(args.run_dir, "checkpoint_best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, nb = 0.0, 0
        for batch in train_loader:
            images = batch["images"].to(dev)
            uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
            gt_pts = batch["map_pts"].to(dev)
            gt_cls = batch["map_cls"].to(dev)
            gt_mask = batch["map_poly_mask"].to(dev)
            vmap = model(images, uv, valid, command=None)["vmap"]
            loss = vector_map_loss(vmap, gt_pts, gt_cls, gt_mask)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); nb += 1
        train_loss = loss_sum / max(nb, 1)

        val_loss = evaluate_vectormap(model, val_loader, dev)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch + 1
            torch.save({"model": model.state_dict(),
                        "use_vector_map": True}, best_path)
        print(f"[vmap] epoch {epoch+1:>3}/{args.epochs}  train_vmap_loss={train_loss:.6f}  "
              f"val_vmap_loss={val_loss:.6f}{' *' if improved else ''}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    torch.save({"model": model.state_dict(),
                "use_vector_map": True},
               os.path.join(args.run_dir, "checkpoint_last.pt"))
    with open(os.path.join(args.run_dir, "history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch, "best_val_loss": best_loss,
                   "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"[vmap] 完成。最优 val_vmap_loss={best_loss:.6f} (epoch {best_epoch}) -> {best_path}")

    # 训练完自动可视化：用 best 权重对验证集叠加高置信度预测折线 + GT 折线。
    # run_vectormap_viz 由兄弟任务 5.2 实现，这里 try/except 保护 import/调用——即便该函数
    # 尚未落地也只打印失败、不中断训练产物（与 train_detection_only 的保护写法一致）。
    try:
        from visualize import run_vectormap_viz
        viz_dir = os.path.join(args.run_dir, "viz")
        print(f"[vmap] 自动可视化矢量地图折线（用最优权重）-> {viz_dir}", flush=True)
        run_vectormap_viz(best_path, meta=args.meta, num=6, seed=args.seed,
                          val_ratio=args.val_ratio, out_dir=viz_dir, score_thr=0.3)
    except Exception as e:  # noqa: BLE001
        print(f"[vmap] 自动可视化失败（不影响训练产物）: {e}", flush=True)


@torch.no_grad()
def evaluate_detection_motion(model, loader, device):
    """验证集平均 (检测 loss, 运动 loss)（用于 --det-motion 的分步验证，任务 3.5）。

    对每个 batch 调一次 forward 取 dict['det']（含 agent_fut），分别算
    detection_loss 与 motion_loss，各按 batch 数平均。读检测监督字段 agent_box/
    agent_cls/agent_mask + 运动监督字段 agent_fut/agent_fut_mask。
    """
    model.eval()
    det_sum, mot_sum, nb = 0.0, 0.0, 0
    for batch in loader:
        images = batch["images"].to(device)
        uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, device)
        gt_box = batch["agent_box"].to(device)
        gt_cls = batch["agent_cls"].to(device)
        gt_mask = batch["agent_mask"].to(device)
        gt_fut = batch["agent_fut"].to(device)
        gt_fut_mask = batch["agent_fut_mask"].to(device)
        det = model(images, uv, valid, command=None)["det"]
        det_sum += detection_loss(det, gt_box, gt_cls, gt_mask).item()
        mot_sum += motion_loss(det["agent_fut"], det, gt_fut, gt_fut_mask,
                               gt_box, gt_mask).item()
        nb += 1
    model.train()
    nb = max(nb, 1)
    return det_sum / nb, mot_sum / nb


def train_detection_motion(args, dev):
    """检测 + 运动联合在小子集上训练 BEV 编码器 + 检测头 + 运动头（演示入口，任务 3.5）。

    在 --det-only 路径（任务 2.6）基础上再叠加运动 loss：运动头复用检测头的 query_emb
    回归每个 agent 的未来轨迹，与检测用同一套贪心匹配（见 motion_loss）。对齐 design.md
    决策 6「先检测、再叠加运动」的分步节奏与需求 3.5（固定小子集反复训练运动 loss 显著
    下降）。backbone 默认冻结，优化除 backbone 外的所有可训练参数（BEV cross-attention
    + 检测头 + 运动头）。不碰规划/占据/可行驶区 loss。

    总损失：L = L_det + mot_weight · L_mot（运动 loss 量级通常与检测不同，用 --mot-weight
    调平；默认 1.0）。按 val 检测+运动联合 loss 最优存 checkpoint_best.pt，便于
    visualize.py --motion 复现验证划分后叠加预测/GT 他车轨迹。
    """
    from dataset import HEIGHTS

    train_loader, val_loader, (ntr, nva) = make_loaders(
        args.meta, batch_size=args.batch_size, val_ratio=args.val_ratio,
        seed=args.seed, num_workers=args.num_workers)
    print(f"[det-mot] device={dev}  train={ntr} val={nva}  bs={args.batch_size} "
          f"lr={args.lr} epochs={args.epochs}  mot_weight={args.mot_weight}  "
          f"run_dir={args.run_dir}", flush=True)

    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      use_vector_map=args.use_vector_map,
                      pretrained=True, freeze_backbone=not args.unfreeze_backbone).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = {"train_det_loss": [], "train_mot_loss": [],
               "val_det_loss": [], "val_mot_loss": []}
    best_loss, best_epoch = float("inf"), -1
    best_path = os.path.join(args.run_dir, "checkpoint_best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        det_sum, mot_sum, nb = 0.0, 0.0, 0
        for batch in train_loader:
            images = batch["images"].to(dev)
            uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
            gt_box = batch["agent_box"].to(dev)
            gt_cls = batch["agent_cls"].to(dev)
            gt_mask = batch["agent_mask"].to(dev)
            gt_fut = batch["agent_fut"].to(dev)
            gt_fut_mask = batch["agent_fut_mask"].to(dev)
            det = model(images, uv, valid, command=None)["det"]
            l_det = detection_loss(det, gt_box, gt_cls, gt_mask)
            l_mot = motion_loss(det["agent_fut"], det, gt_fut, gt_fut_mask, gt_box, gt_mask)
            loss = l_det + args.mot_weight * l_mot
            opt.zero_grad(); loss.backward(); opt.step()
            det_sum += l_det.item(); mot_sum += l_mot.item(); nb += 1
        nb = max(nb, 1)
        train_det, train_mot = det_sum / nb, mot_sum / nb

        val_det, val_mot = evaluate_detection_motion(model, val_loader, dev)
        history["train_det_loss"].append(train_det)
        history["train_mot_loss"].append(train_mot)
        history["val_det_loss"].append(val_det)
        history["val_mot_loss"].append(val_mot)
        val_total = val_det + args.mot_weight * val_mot
        improved = val_total < best_loss
        if improved:
            best_loss, best_epoch = val_total, epoch + 1
            torch.save({"model": model.state_dict(),
                        "use_vector_map": args.use_vector_map}, best_path)
        print(f"[det-mot] epoch {epoch+1:>3}/{args.epochs}  "
              f"train_det={train_det:.6f} train_mot={train_mot:.6f}  "
              f"val_det={val_det:.6f} val_mot={val_mot:.6f}"
              f"{' *' if improved else ''}  ({time.time()-t0:.1f}s)", flush=True)

    torch.save({"model": model.state_dict(),
                "use_vector_map": args.use_vector_map},
               os.path.join(args.run_dir, "checkpoint_last.pt"))
    with open(os.path.join(args.run_dir, "history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch, "best_val_loss": best_loss,
                   "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"[det-mot] 完成。最优 val(det+mot)={best_loss:.6f} (epoch {best_epoch}) -> {best_path}")

    # 训练完自动可视化：用 best 权重对验证集叠加高置信度预测框的预测/GT 他车未来轨迹
    try:
        from visualize import run_motion_viz
        viz_dir = os.path.join(args.run_dir, "viz")
        print(f"[det-mot] 自动可视化他车预测轨迹（用最优权重）-> {viz_dir}", flush=True)
        run_motion_viz(best_path, meta=args.meta, num=6, seed=args.seed,
                       val_ratio=args.val_ratio, out_dir=viz_dir, score_thr=0.3)
    except Exception as e:  # noqa: BLE001
        print(f"[det-mot] 自动可视化失败（不影响训练产物）: {e}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device):
    """验证集平均规划 ADE(米) 与 loss。"""
    model.eval()
    dist_sum, n_valid, loss_sum, weight = 0.0, 0, 0.0, 0.0
    for batch in loader:
        images = batch["images"].to(device)
        traj_gt = batch["traj_gt"].to(device)
        mask = batch["traj_mask"].to(device)
        uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, device)
        command = batch["command"].to(device)
        pred = model(images, uv, valid, command=command)["traj"]
        loss_sum += planning_loss(pred, traj_gt, mask).item() * mask.sum().item()
        weight += mask.sum().item()
        err_m = (pred - traj_gt) * SCENE_SCALE
        d = err_m.pow(2).sum(-1).sqrt()              # (B,Tf) 米
        dist_sum += d[mask].sum().item()
        n_valid += int(mask.sum().item())
    model.train()
    if n_valid == 0:
        return float("nan"), float("nan")
    return loss_sum / weight, dist_sum / n_valid


def train_multitask(args, dev):
    """全头联合多任务训练：规划 + 占据 + 可行驶区 + 检测 + 运动（任务 4.4，决策 6）。

    总损失（design.md 决策 6）：
        L = w_plan·L_plan + w_occ·L_occ + w_map·L_map + w_det·L_det + w_mot·L_mot
    其中 w_plan 固定为 1（规划是主任务），其余权重由命令行 --occ-weight / --map-weight /
    --det-weight / --mot-weight 调节（保守默认见 main 的参数定义）。

    按需只开部分头（需求 5.3，分步验证用）：**把某个头的权重设为 0 即可干净地关掉该头
    的监督**——对应 loss 项不参与总损失（也不进反传），且在便于短路的地方跳过其计算以
    省算力。例如 `--det-weight 0 --mot-weight 0` 就只训练 规划+占据+可行驶区 三头；
    `--occ-weight 0 --map-weight 0 --mot-weight 0` 则退化成 规划+检测。权重为 0 时该项不
    产生 NaN、不浪费计算（见下方训练循环的短路逻辑）。
    注意：当 use_planner_v2=True 时，规划头内部要用检测头的 query_emb/score，因此即便
    --det-weight/--mot-weight 都为 0，检测前向仍会执行（只是不计入 loss）；只有在
    --no-planner-v2 且 det/mot 权重皆 0 时，才完全跳过检测头前向以省算力。

    规划头版本（决策 4）：由 args.use_planner_v2 选择：
      - True ：PlanningHeadV2（ego query 交互式 attend BEV + agent query_emb）。
      - False：旧 PlanningHead（BEV 全局池化），用于学习对照。
    模型统一返回 dict，一次拿到 traj/occ/map 与检测头输出 det（含运动 det['agent_fut']）；
    当 use_planner_v2=True 时 traj 已来自 planner_v2（内部用了检测信息），det 被复用不重复计算。

    评估：每 epoch 用 evaluate 算验证集规划 ADE(米)，按 val ADE 最优存 checkpoint_best.pt
    （checkpoint 额外写入 use_planner_v2 便于可视化按同版本重建模型）。本路径亦为任务 5.1
    的多任务总损失骨架。
    """
    from dataset import HEIGHTS

    train_loader, val_loader, (ntr, nva) = make_loaders(
        args.meta, batch_size=args.batch_size, val_ratio=args.val_ratio,
        seed=args.seed, num_workers=args.num_workers)
    print(f"[multi] device={dev}  train={ntr} val={nva}  bs={args.batch_size} "
          f"lr={args.lr} epochs={args.epochs}  use_planner_v2={args.use_planner_v2}", flush=True)
    print(f"[multi] weights: plan=1.0 occ={args.occ_weight} map={args.map_weight} "
          f"det={args.det_weight} mot={args.mot_weight} vmap={args.vmap_weight}  "
          f"run_dir={args.run_dir}", flush=True)

    # 按需只开部分头（需求 5.3）：权重为 0 的头干净关掉。检测头前向是否需要——
    # use_planner_v2=True 时规划头内部依赖检测 query，故必须跑检测前向；只有在
    # 回退旧规划头（v1 全局池化）且 det/mot 权重皆 0 时，才完全跳过检测头以省算力。
    need_det = args.use_planner_v2 or args.det_weight != 0.0 or args.mot_weight != 0.0
    if not need_det:
        print("[multi] det/mot 权重为 0 且用 v1 规划头 -> 跳过检测头前向以省算力", flush=True)

    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      use_planner_v2=args.use_planner_v2,
                      use_vector_map=args.use_vector_map,
                      pretrained=True, freeze_backbone=not args.unfreeze_backbone).to(dev)
    # 优化器参数分组：解冻 backbone 时用更小 lr（lr/10）微调，避免大 lr 破坏预训练特征；
    # 其余头/编码器用主 lr。冻结时 backbone 参数 requires_grad=False，自动不进优化器。
    opt = _build_optimizer(model, args.lr)
    # cosine lr 衰减（--lr-cosine 开启）：学习率从初始值按余弦曲线平滑降到 eta_min，
    # 让训练后段 lr 变小、精修更稳（对解冻 backbone 的长训尤其友好）。每个 epoch 末尾
    # scheduler.step() 推进一步，T_max=总 epoch 数。各参数组（含 backbone 的 lr/10）
    # 按各自初始 lr 同比例衰减。关闭时为 None，lr 保持恒定（向后兼容原行为）。
    sched = None
    if getattr(args, "lr_cosine", False):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.01)
        print(f"[multi] 启用 cosine lr 衰减：T_max={args.epochs} eta_min={args.lr*0.01:g}",
              flush=True)

    history = {"train_plan": [], "train_occ": [], "train_map": [],
               "train_det": [], "train_mot": [], "train_vmap": [],
               "val_loss": [], "val_ade": []}
    best_ade, best_epoch = float("inf"), -1
    best_path = os.path.join(args.run_dir, "checkpoint_best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        acc = {"plan": 0.0, "occ": 0.0, "map": 0.0, "det": 0.0, "mot": 0.0,
               "vmap": 0.0}
        nb = 0
        for batch in train_loader:
            images = batch["images"].to(dev)
            traj_gt = batch["traj_gt"].to(dev)
            mask = batch["traj_mask"].to(dev)
            uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
            command = batch["command"].to(dev)

            # 规划始终是主任务（w_plan=1），恒计算。其余头按权重是否为 0 短路：
            # 权重为 0 的头不计算 loss、不进反传、不进统计（干净关掉，不产生 NaN）。
            # 对应 GT 张量也只在该头开启时才 .to(dev)（权重为 0 的头连数据搬运都省掉）。
            # forward 统一返回 dict（任务 5.2）：det/agent_fut 始终可读。det 头是否参与
            # 反传仍由 need_det（权重）控制：不需要时读 dict['det'] 后不计 det/mot loss 即可。
            out = model(images, uv, valid, command=command)
            pred, occ_logit, map_logit = out["traj"], out["occ"], out["map"]
            det = out["det"] if need_det else None

            l_plan = planning_loss(pred, traj_gt, mask)
            loss = l_plan
            acc["plan"] += l_plan.item()

            if args.occ_weight != 0.0:
                occ_gt = batch["occ"].to(dev)
                l_occ = occupancy_loss(occ_logit, occ_gt)
                loss = loss + args.occ_weight * l_occ
                acc["occ"] += l_occ.item()
            if args.map_weight != 0.0:
                map_gt = batch["drivable"].to(dev)
                l_map = occupancy_loss(map_logit, map_gt)
                loss = loss + args.map_weight * l_map
                acc["map"] += l_map.item()
            if args.det_weight != 0.0:
                gt_box = batch["agent_box"].to(dev)
                gt_cls = batch["agent_cls"].to(dev)
                gt_mask = batch["agent_mask"].to(dev)
                l_det = detection_loss(det, gt_box, gt_cls, gt_mask)
                loss = loss + args.det_weight * l_det
                acc["det"] += l_det.item()
            if args.mot_weight != 0.0:
                gt_box = batch["agent_box"].to(dev)
                gt_mask = batch["agent_mask"].to(dev)
                gt_fut = batch["agent_fut"].to(dev)
                gt_fut_mask = batch["agent_fut_mask"].to(dev)
                l_mot = motion_loss(det["agent_fut"], det, gt_fut, gt_fut_mask,
                                    gt_box, gt_mask)
                loss = loss + args.mot_weight * l_mot
                acc["mot"] += l_mot.item()
            # 矢量地图头监督（与 det 项并列）：权重为 0 短路跳过（分步验证）。
            # 同时要求 dict 含 'vmap'——--no-vector-map 时不构造该头、forward 无 'vmap' 键，
            # 即便权重非 0 也干净跳过（out.get 容错，不触发 KeyError）。
            if args.vmap_weight != 0.0 and "vmap" in out:
                gt_pts = batch["map_pts"].to(dev)
                gt_mcls = batch["map_cls"].to(dev)
                gt_mmask = batch["map_poly_mask"].to(dev)
                l_vmap = vector_map_loss(out["vmap"], gt_pts, gt_mcls, gt_mmask)
                loss = loss + args.vmap_weight * l_vmap
                acc["vmap"] += l_vmap.item()

            opt.zero_grad(); loss.backward(); opt.step()
            nb += 1
        nb = max(nb, 1)
        for k in acc:
            acc[k] /= nb

        val_loss, val_ade = evaluate(model, val_loader, dev)
        history["train_plan"].append(acc["plan"]); history["train_occ"].append(acc["occ"])
        history["train_map"].append(acc["map"]); history["train_det"].append(acc["det"])
        history["train_mot"].append(acc["mot"]); history["train_vmap"].append(acc["vmap"])
        history["val_loss"].append(val_loss); history["val_ade"].append(val_ade)
        improved = val_ade < best_ade
        if improved:
            best_ade, best_epoch = val_ade, epoch + 1
            torch.save({"model": model.state_dict(),
                        "use_planner_v2": args.use_planner_v2,
                        "use_vector_map": args.use_vector_map}, best_path)
        # 每个 epoch 都存一份带编号的 checkpoint（便于回看任意 epoch 的权重）
        ckpt_dir = os.path.join(args.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "use_planner_v2": args.use_planner_v2,
                    "use_vector_map": args.use_vector_map,
                    "epoch": epoch + 1, "val_ade": val_ade},
                   os.path.join(ckpt_dir, f"checkpoint_ep{epoch+1:03d}.pt"))
        cur_lr = opt.param_groups[0]["lr"]               # 头/编码器主 lr（供日志观测衰减）
        print(f"[multi] epoch {epoch+1:>3}/{args.epochs}  "
              f"plan={acc['plan']:.6f} occ={acc['occ']:.4f} map={acc['map']:.4f} "
              f"det={acc['det']:.4f} mot={acc['mot']:.6f} vmap={acc['vmap']:.6f}  "
              f"val_ADE={val_ade:.3f} m{' *' if improved else ''}  "
              f"lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)", flush=True)
        # cosine 衰减每个 epoch 末推进一步（关闭时 sched 为 None，lr 恒定）
        if sched is not None:
            sched.step()

    torch.save({"model": model.state_dict(), "use_planner_v2": args.use_planner_v2,
                "use_vector_map": args.use_vector_map},
               os.path.join(args.run_dir, "checkpoint_last.pt"))
    with open(os.path.join(args.run_dir, "history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch, "best_val_ade": best_ade,
                   "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"[multi] 完成。最优 val_ADE={best_ade:.3f} m (epoch {best_epoch}) -> {best_path}")

    # 训练完自动可视化：用 best 权重出轨迹/占据/可行驶区三栏图（按同版本规划头重建）
    try:
        from visualize import run_topdown_viz, run_visualization
        viz_dir = os.path.join(args.run_dir, "viz")
        print(f"[multi] 自动可视化（用最优权重）-> {viz_dir}", flush=True)
        run_visualization(best_path, meta=args.meta, num=6, seed=args.seed,
                          val_ratio=args.val_ratio, out_dir=viz_dir,
                          use_planner_v2=args.use_planner_v2)
        # 额外产出统一俯视图（自车矩形+预测/真值轨迹+障碍框+障碍未来轨迹）
        run_topdown_viz(best_path, meta=args.meta, num=6, seed=args.seed,
                        val_ratio=args.val_ratio,
                        out_dir=os.path.join(viz_dir, "topdown"),
                        use_planner_v2=args.use_planner_v2)
    except Exception as e:  # noqa: BLE001
        print(f"[multi] 自动可视化失败（不影响训练产物）: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="cache/meta_2000.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="联合微调 ResNet（更强但更吃显存/更慢）")
    parser.add_argument("--occ-weight", type=float, default=1.0,
                        help="占据 loss 权重（多任务下设 0 可关闭占据头监督）")
    parser.add_argument("--map-weight", type=float, default=1.0,
                        help="可行驶区分割 loss 权重（多任务下设 0 可关闭可行驶区头监督）")
    parser.add_argument("--det-only", action="store_true",
                        help="仅检测 loss 训练 BEV 编码器 + 检测头（分步验证，任务 2.6）")
    parser.add_argument("--vmap-only", action="store_true",
                        help="仅矢量地图 loss 训练 BEV 编码器 + 矢量地图头（分步验证，任务 5.1）")
    parser.add_argument("--det-motion", action="store_true",
                        help="检测 + 运动联合训练 BEV 编码器 + 检测头 + 运动头（分步验证，任务 3.5）")
    parser.add_argument("--multitask", action="store_true",
                        help="全头联合多任务训练：规划+占据+可行驶区+检测+运动（任务 4.4）")
    parser.add_argument("--det-weight", type=float, default=0.5,
                        help="检测 loss 权重（多任务下与规划 loss 调平，保守默认 0.5；"
                             "设 0 可关闭检测头监督，用于分步验证）")
    parser.add_argument("--mot-weight", type=float, default=1.0,
                        help="运动 loss 权重（--det-motion / --multitask 下与其它 loss 调平；"
                             "多任务下设 0 可关闭运动头监督，用于分步验证）")
    parser.add_argument("--vmap-weight", type=float, default=0.5,
                        help="矢量地图 loss 权重；设 0 可关闭矢量地图头监督，用于分步验证")
    parser.add_argument("--lr-cosine", action="store_true",
                        help="多任务训练启用 cosine lr 衰减（lr 从初值按余弦降到 lr*0.01，"
                             "长训/解冻 backbone 精修更稳）；默认关闭保持恒定 lr")
    # 规划头版本开关（决策 4）：默认走 v2 交互式；--no-planner-v2 回退旧 v1 全局池化做对照
    parser.add_argument("--planner-v2", dest="use_planner_v2", action="store_true",
                        default=True, help="用交互式 PlanningHeadV2（默认）")
    parser.add_argument("--no-planner-v2", dest="use_planner_v2", action="store_false",
                        help="回退旧 PlanningHead（BEV 全局池化），用于学习对照")
    # 矢量地图头开关（vector-map-head 任务 2.3，需求 7.6）：默认启用；--no-vector-map 关闭
    parser.add_argument("--vector-map", dest="use_vector_map", action="store_true",
                        default=True, help="构造矢量地图头 VectorMapHead（默认）")
    parser.add_argument("--no-vector-map", dest="use_vector_map", action="store_false",
                        help="不构造矢量地图头，forward 返回 dict 不含 'vmap' 键")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = os.path.splitext(os.path.basename(args.meta))[0]
    if args.multitask:
        mode = "multi_" + ("pv2" if args.use_planner_v2 else "pv1")
    elif args.det_motion:
        mode = "detmot"
    elif args.det_only:
        mode = "det"
    elif args.vmap_only:
        mode = "vmap"
    else:
        mode = "v2"
    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join(
            "runs", f"{stamp}_{mode}_{tag}_ep{args.epochs}_lr{args.lr:g}_bs{args.batch_size}")
    os.makedirs(args.run_dir, exist_ok=True)

    # 训练日志同时落盘到 run_dir/train.log（行缓冲，实时可见）
    _logf = open(os.path.join(args.run_dir, "train.log"), "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _logf)
    sys.stderr = _Tee(sys.__stderr__, _logf)

    # 全头联合多任务训练路径（任务 4.4）：规划+占据+可行驶区+检测+运动，训练完自动可视化
    if args.multitask:
        train_multitask(args, dev)
        return

    # 检测 + 运动联合分步验证路径（任务 3.5）：检测 loss + 运动 loss，训练完自动叠加
    # 高置信度预测框的预测/GT 他车未来轨迹可视化
    if args.det_motion:
        train_detection_motion(args, dev)
        return

    # 检测专用分步验证路径（任务 2.6）：只开检测 loss，训练完自动叠加预测/GT 框可视化
    if args.det_only:
        train_detection_only(args, dev)
        return

    # 矢量地图专用分步验证路径（任务 5.1）：只开矢量地图 loss，训练完自动叠加预测/GT 折线可视化
    if args.vmap_only:
        train_vectormap_only(args, dev)
        return

    train_loader, val_loader, (ntr, nva) = make_loaders(
        args.meta, batch_size=args.batch_size, val_ratio=args.val_ratio,
        seed=args.seed, num_workers=args.num_workers)
    print(f"[train] device={dev}  train={ntr} val={nva}  bs={args.batch_size} "
          f"lr={args.lr} epochs={args.epochs}  run_dir={args.run_dir}", flush=True)

    from dataset import HEIGHTS
    model = NanoFSDv2(bev_size=100, n_heights=len(HEIGHTS), n_cams=6, fut_steps=FUT_STEPS,
                      use_planner_v2=args.use_planner_v2,
                      use_vector_map=args.use_vector_map,
                      pretrained=True, freeze_backbone=not args.unfreeze_backbone).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_ade": []}
    best_ade, best_epoch = float("inf"), -1
    best_path = os.path.join(args.run_dir, "checkpoint_best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, nb = 0.0, 0
        for batch in train_loader:
            images = batch["images"].to(dev)
            traj_gt = batch["traj_gt"].to(dev)
            mask = batch["traj_mask"].to(dev)
            uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"], IMG_HW, dev)
            occ_gt = batch["occ"].to(dev)
            map_gt = batch["drivable"].to(dev)
            command = batch["command"].to(dev)
            out = model(images, uv, valid, command=command)
            pred, occ_logit, map_logit = out["traj"], out["occ"], out["map"]
            l_plan = planning_loss(pred, traj_gt, mask)
            l_occ = occupancy_loss(occ_logit, occ_gt)
            l_map = occupancy_loss(map_logit, map_gt)        # 可行驶区分割，同为带权 BCE
            loss = l_plan + args.occ_weight * l_occ + args.map_weight * l_map
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += l_plan.item(); nb += 1               # 记录规划 loss 便于和之前对比
        train_loss = loss_sum / max(nb, 1)

        val_loss, val_ade = evaluate(model, val_loader, dev)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ade"].append(val_ade)
        improved = val_ade < best_ade
        if improved:
            best_ade, best_epoch = val_ade, epoch + 1
            torch.save({"model": model.state_dict(),
                        "use_planner_v2": args.use_planner_v2,
                        "use_vector_map": args.use_vector_map}, best_path)
        print(f"[train] epoch {epoch+1:>3}/{args.epochs}  train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  val_ADE={val_ade:.3f} m"
              f"{' *' if improved else ''}  ({time.time()-t0:.1f}s)", flush=True)

    torch.save({"model": model.state_dict(), "use_planner_v2": args.use_planner_v2,
                "use_vector_map": args.use_vector_map},
               os.path.join(args.run_dir, "checkpoint_last.pt"))
    with open(os.path.join(args.run_dir, "history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch, "best_val_ade": best_ade,
                   "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"[train] 完成。最优 val_ADE={best_ade:.3f} m (epoch {best_epoch}) -> {best_path}")

    # 训练完自动可视化：用 best 权重对验证集出三栏图，存到 run_dir/viz/
    try:
        from visualize import run_visualization
        viz_dir = os.path.join(args.run_dir, "viz")
        print(f"[train] 自动可视化（用最优权重）-> {viz_dir}", flush=True)
        run_visualization(best_path, meta=args.meta, num=6, seed=args.seed,
                          val_ratio=args.val_ratio, out_dir=viz_dir,
                          use_planner_v2=args.use_planner_v2)
    except Exception as e:  # noqa: BLE001
        print(f"[train] 自动可视化失败（不影响训练产物）: {e}", flush=True)


if __name__ == "__main__":
    main()
