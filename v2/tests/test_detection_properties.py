"""任务 2.4 检测头属性测试（hypothesis）。

针对 `model.py` 的 `DetectionHead` 与 `train.py` 的 `match_pred_to_gt` /
`detection_loss`，验证四条正确性属性：

- **属性 1 形状守恒**：随机合法 BEV 输入下，det 各输出形状恒为
  `box (B,N,5)`、`cls (B,N,n_cls)`、`score (B,N,1)`、`query_emb (B,N,D)`。
- **属性 3 匹配正确性**：构造已知 GT，令部分预测框精确等于 GT，`match_pred_to_gt`
  必须把这些 GT 指派到对应的预测 query（且匹配为单射、来自有效 GT）。
- **属性 4 损失零点**：预测框=GT、类别 logit 强烈命中 GT 类、匹配 query 存在性置高、
  背景置低时，`detection_loss` 在有效位上 ≈0。框回归部分**精确为 0**；类别 CE 与存在性
  BCE 有数值下界（logit 有限），构造足够大的 logit 间隔使之趋近 0——对二者仅断言「很小」。
- **属性 6 空场景安全**：M=0（无有效 GT）时 `detection_loss` 返回有限值、框/类别贡献为 0，
  且反向传播不产生 NaN。

记号：N=50（query 数）、MAX_GT=50、D=128、n_cls=2、BEV 边长 S=100。

验证：正确性属性 1、3、4、6；需求 2.6, 2.7。
"""

import os
import sys

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DetectionHead  # noqa: E402
from train import detection_loss, match_pred_to_gt  # noqa: E402

# 全局约定（与 spec 一致）
N = 50            # query 数
MAX_GT = 50       # 结构化 agent 定长上限
D = 128           # 特征维度
N_CLS = 2         # 类别数：0=车 1=人
S = 100           # BEV 边长
KV_STRIDE = 2     # BEV token 下采样步长


def _make_head(n_query: int = N, n_cls: int = N_CLS,
               n_layers: int = 2) -> DetectionHead:
    """构造一个确定性初始化的检测头（固定 seed，便于复现）。"""
    torch.manual_seed(0)
    return DetectionHead(d_model=D, n_query=n_query, n_layers=n_layers,
                         n_cls=n_cls, bev_size=S, kv_stride=KV_STRIDE)


# ---------------------------------------------------------------------------
# 属性 1：形状守恒
# ---------------------------------------------------------------------------

@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(1, 3), n_query=st.sampled_from([1, 8, N]),
       n_cls=st.sampled_from([2, 3]))
def test_shape_conservation(B, n_query, n_cls):
    """Feature: nano-fsd-v2, Property 1 (DetectionHead output shapes).

    **Validates: 正确性属性 1；需求 2.6**
    """
    head = _make_head(n_query=n_query, n_cls=n_cls).eval()
    bev = torch.randn(B, D, S, S)
    with torch.no_grad():
        out = head(bev)
    assert out["box"].shape == (B, n_query, 5)
    assert out["cls"].shape == (B, n_query, n_cls)
    assert out["score"].shape == (B, n_query, 1)
    assert out["query_emb"].shape == (B, n_query, D)
    # 输出均为有限浮点，无 NaN/Inf
    for k in ("box", "cls", "score", "query_emb"):
        assert torch.isfinite(out[k]).all(), f"{k} 含 NaN/Inf"


# ---------------------------------------------------------------------------
# 属性 3：匹配正确性
# ---------------------------------------------------------------------------

def _random_gt(M: int, gen: torch.Generator):
    """构造 M 个有效 GT（pad 到 MAX_GT）：中心互相远离，避免贪心匹配歧义。

    返回 (gt_box (MAX_GT,5), gt_mask (MAX_GT,))。中心用网格 + 抖动保证两两距离足够大，
    这样「某预测框精确等于某 GT」时最近的一定是它自己。
    """
    gt_box = torch.zeros(MAX_GT, 5)
    gt_mask = torch.zeros(MAX_GT, dtype=torch.bool)
    # 网格坐标（间隔 10，远大于抖动幅度）保证中心两两可区分
    xs = torch.arange(M, dtype=torch.float32) * 10.0 - (M - 1) * 5.0
    jitter = (torch.rand(M, generator=gen) - 0.5) * 2.0      # ±1 抖动
    gt_box[:M, 0] = xs + jitter
    gt_box[:M, 1] = (torch.rand(M, generator=gen) - 0.5) * 4.0
    gt_box[:M, 2] = (torch.rand(M, generator=gen) - 0.5) * 3.0   # yaw
    gt_box[:M, 3:5] = torch.rand(M, 2, generator=gen) * 4.0 + 1.0  # w,l
    gt_mask[:M] = True
    return gt_box, gt_mask


@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(M=st.integers(1, MAX_GT), seed=st.integers(0, 10_000))
def test_match_correctness(M, seed):
    """Feature: nano-fsd-v2, Property 3 (exact-GT predictions must match).

    令 N 个预测框中，前 M 个精确等于 M 个有效 GT（打乱顺序放置），匹配结果必须把每个
    GT 指派到那个与它坐标相同的预测 query。

    **Validates: 正确性属性 3；需求 2.6**
    """
    gen = torch.Generator().manual_seed(seed)
    gt_box, gt_mask = _random_gt(M, gen)

    # 预测框：默认放到远处（(1000,1000)）当背景噪声；再把 M 个有效 GT 的中心精确拷到
    # 随机 M 个 query 槽位，建立「预测 query -> GT」的真值对应。
    pred_box = torch.full((N, 5), 1000.0)
    perm = torch.randperm(N, generator=gen)[:M]              # 选 M 个 query 槽位
    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()  # (M,)
    for k in range(M):
        pred_box[perm[k]] = gt_box[valid_gt_idx[k]]          # 坐标精确相等
    expected = {int(valid_gt_idx[k]): int(perm[k]) for k in range(M)}  # gt->pred

    pred_idx, matched_gt_idx = match_pred_to_gt(pred_box, gt_box, gt_mask)

    # 每个有效 GT 都被匹配，且匹配到的正是坐标相同的那个 query
    assert pred_idx.numel() == M
    assert matched_gt_idx.numel() == M
    # 匹配为单射：预测/ GT 索引均不重复
    assert len(set(pred_idx.tolist())) == M
    assert len(set(matched_gt_idx.tolist())) == M
    got = {int(g): int(p) for g, p in zip(matched_gt_idx.tolist(), pred_idx.tolist())}
    assert got == expected, f"匹配不符：got={got} expected={expected}"
    # 匹配到的 GT 必须都是有效 GT
    assert set(matched_gt_idx.tolist()).issubset(set(valid_gt_idx.tolist()))


def test_match_empty_scene():
    """M=0（无有效 GT）时匹配返回两个空张量。

    **Validates: 正确性属性 3、6；需求 2.6, 2.7**
    """
    pred_box = torch.randn(N, 5)
    gt_box = torch.zeros(MAX_GT, 5)
    gt_mask = torch.zeros(MAX_GT, dtype=torch.bool)
    pred_idx, gt_idx = match_pred_to_gt(pred_box, gt_box, gt_mask)
    assert pred_idx.numel() == 0 and gt_idx.numel() == 0


# ---------------------------------------------------------------------------
# 属性 4：损失零点
# ---------------------------------------------------------------------------

@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(M=st.integers(1, MAX_GT), seed=st.integers(0, 10_000))
def test_loss_zero_point(M, seed):
    """Feature: nano-fsd-v2, Property 4 (loss near zero at the GT).

    构造 det：匹配 query 的框精确等于 GT，类别 logit 强烈命中 GT 类，存在性 score 对
    匹配 query 置高、背景置低。此时：
      - 框回归部分 **精确为 0**（smooth L1 于相等输入）；
      - 类别 CE、存在性 BCE 有数值下界（logit 有限），构造大 logit 间隔使之趋近 0。
    断言 **总 detection_loss 很小**，且框回归分量精确为 0。

    **Validates: 正确性属性 4；需求 2.6**
    """
    gen = torch.Generator().manual_seed(seed)
    gt_box, gt_mask = _random_gt(M, gen)
    gt_cls = torch.zeros(MAX_GT, dtype=torch.long)
    gt_cls[:M] = torch.randint(0, N_CLS, (M,), generator=gen)

    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()  # (M,)

    # 单 batch。预测框默认远处背景，M 个 query 精确拷贝 GT 框。
    pred_box = torch.full((1, N, 5), 1000.0)
    perm = torch.randperm(N, generator=gen)[:M]
    cls_logit = torch.zeros(1, N, N_CLS)
    score_logit = torch.full((1, N, 1), -20.0)              # 背景 query 存在性置低
    for k in range(M):
        q = int(perm[k])
        g = int(valid_gt_idx[k])
        pred_box[0, q] = gt_box[g]                           # 框精确相等 -> smooth L1 = 0
        cls_logit[0, q, int(gt_cls[g])] = 30.0               # 类别 logit 强烈命中 -> CE≈0
        score_logit[0, q, 0] = 20.0                          # 匹配 query 存在性置高

    det = {"box": pred_box, "cls": cls_logit, "score": score_logit}
    loss = detection_loss(det, gt_box.unsqueeze(0), gt_cls.unsqueeze(0),
                          gt_mask.unsqueeze(0))

    assert torch.isfinite(loss), "loss 含 NaN/Inf"
    # 总 loss 很小（框=0，类别/存在性因大 logit 间隔趋近 0）
    assert loss.item() < 1e-3, f"损失零点不满足：loss={loss.item():.3e}"


def test_loss_box_part_exactly_zero():
    """框回归分量在 pred_box==gt_box 时精确为 0（显式检查，不依赖随机）。

    直接复算 detection_loss 内部的框回归项（smooth L1，匹配 query 对相等 GT 框）为 0，
    与 design.md「损失零点」中「框回归部分≈0」的描述对齐。

    **Validates: 正确性属性 4；需求 2.6**
    """
    import torch.nn.functional as F

    M = 5
    gen = torch.Generator().manual_seed(123)
    gt_box, gt_mask = _random_gt(M, gen)
    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()

    pred_box = torch.full((1, N, 5), 1000.0)
    perm = torch.randperm(N, generator=gen)[:M]
    for k in range(M):
        pred_box[0, int(perm[k])] = gt_box[int(valid_gt_idx[k])]

    pred_idx, gt_idx = match_pred_to_gt(pred_box[0], gt_box, gt_mask)
    box_err = F.smooth_l1_loss(pred_box[0, pred_idx], gt_box[gt_idx], reduction="sum")
    assert box_err.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 属性 6：空场景安全
# ---------------------------------------------------------------------------

@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(1, 3))
def test_empty_scene_loss_safe(B):
    """Feature: nano-fsd-v2, Property 6 (M=0: finite loss, no NaN on backward).

    整个 batch 无有效 GT（M=0）时：
      - detection_loss 返回有限标量（框/类别贡献为 0，仅存在性 BCE 对全背景生效）；
      - 反向传播不产生 NaN（梯度链路安全）。

    **Validates: 正确性属性 6；需求 2.7**
    """
    head = _make_head()
    bev = torch.randn(B, D, S, S, requires_grad=True)
    det = head(bev)

    gt_box = torch.zeros(B, MAX_GT, 5)
    gt_cls = torch.zeros(B, MAX_GT, dtype=torch.long)
    gt_mask = torch.zeros(B, MAX_GT, dtype=torch.bool)       # 全无效 -> M=0

    loss = detection_loss(det, gt_box, gt_cls, gt_mask)
    assert torch.isfinite(loss), "M=0 时 loss 非有限"

    loss.backward()
    # 反传无 NaN：对可学习参数与输入梯度都检查
    for name, p in head.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"参数 {name} 梯度含 NaN/Inf"
    assert bev.grad is None or torch.isfinite(bev.grad).all()


def test_empty_scene_loss_value_matches_bce_only():
    """M=0 时 loss 应等于「纯背景存在性 BCE」（框/类别项为 0）。

    显式验证空场景下框回归与类别 CE 贡献为 0：手动构造 det，断言 loss 等于对全 0 目标的
    存在性 BCE。

    **Validates: 正确性属性 6；需求 2.7**
    """
    import torch.nn.functional as F

    B = 2
    torch.manual_seed(1)
    det = {
        "box": torch.randn(B, N, 5),
        "cls": torch.randn(B, N, N_CLS),
        "score": torch.randn(B, N, 1),
    }
    gt_box = torch.zeros(B, MAX_GT, 5)
    gt_cls = torch.zeros(B, MAX_GT, dtype=torch.long)
    gt_mask = torch.zeros(B, MAX_GT, dtype=torch.bool)

    loss = detection_loss(det, gt_box, gt_cls, gt_mask)
    # 空场景：exist_target 全 0，loss == BCE(score, 0)
    expected = F.binary_cross_entropy_with_logits(
        det["score"].squeeze(-1), torch.zeros(B, N))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)
