"""任务 3.3 运动头属性测试（hypothesis）。

针对 `model.py` 的 `MotionHead` 与 `train.py` 的 `motion_loss`，验证三条正确性属性：

- **属性 1 形状守恒**：随机合法 `query_emb (B,N,D)` 输入下，`MotionHead` 输出恒为
  `agent_fut_pred (B,N,Tf,2)`，且全为有限浮点（无 NaN/Inf）。
- **属性 4 损失零点**：令匹配 query 的预测轨迹精确等于对应 GT 的未来轨迹（`agent_fut`）时，
  `motion_loss` 在有效位（匹配 query × 有效未来步）上 **精确为 0**（masked MSE 于相等输入）。
- **属性 5 mask 单调性**：先对 M 个有效 agent 算一次 `motion_loss`；再往 pad 槽位里塞入
  若干「存在但无效」的 agent（`agent_mask=False`、`agent_fut_mask=False`），有效位上的
  loss 值必须**完全不变**。无效 pad agent 不参与匹配、不计入分子分母，故不影响 loss。

记号：N=50（query 数）、MAX_GT=50、D=128、Tf=12、BEV 边长 S=100。

验证：正确性属性 1、4、5；需求 3.4。
"""

import os
import sys

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MotionHead  # noqa: E402
from train import motion_loss  # noqa: E402

# 全局约定（与 spec 一致）
N = 50            # query 数
MAX_GT = 50       # 结构化 agent 定长上限
D = 128           # 特征维度
TF = 12           # 未来步数（fut_steps）


def _make_head(fut_steps: int = TF) -> MotionHead:
    """构造一个确定性初始化的运动头（固定 seed，便于复现）。"""
    torch.manual_seed(0)
    return MotionHead(d_model=D, fut_steps=fut_steps)


def _random_gt_box(M: int, gen: torch.Generator):
    """构造 M 个有效 GT 框（pad 到 MAX_GT）：中心互相远离，避免贪心匹配歧义。

    与 test_detection_properties 的 `_random_gt` 同风格：网格 + 抖动保证两两距离足够大，
    这样「某预测框精确等于某 GT」时最近的一定是它自己，匹配结果确定。

    返回 (gt_box (MAX_GT,5), gt_mask (MAX_GT,))。
    """
    gt_box = torch.zeros(MAX_GT, 5)
    gt_mask = torch.zeros(MAX_GT, dtype=torch.bool)
    xs = torch.arange(M, dtype=torch.float32) * 10.0 - (M - 1) * 5.0
    jitter = (torch.rand(M, generator=gen) - 0.5) * 2.0          # ±1 抖动
    gt_box[:M, 0] = xs + jitter
    gt_box[:M, 1] = (torch.rand(M, generator=gen) - 0.5) * 4.0
    gt_box[:M, 2] = (torch.rand(M, generator=gen) - 0.5) * 3.0   # yaw
    gt_box[:M, 3:5] = torch.rand(M, 2, generator=gen) * 4.0 + 1.0  # w,l
    gt_mask[:M] = True
    return gt_box, gt_mask


# ---------------------------------------------------------------------------
# 属性 1：形状守恒
# ---------------------------------------------------------------------------

@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(1, 3), n_query=st.sampled_from([1, 8, N]),
       fut_steps=st.sampled_from([1, 6, TF]))
def test_shape_conservation(B, n_query, fut_steps):
    """Feature: nano-fsd-v2, Property 1 (MotionHead output shape).

    随机 `query_emb (B,n_query,D)` 下，输出恒为 `(B,n_query,fut_steps,2)` 且全有限。

    **Validates: 正确性属性 1；需求 3.4**
    """
    head = _make_head(fut_steps=fut_steps).eval()
    query_emb = torch.randn(B, n_query, D)
    with torch.no_grad():
        out = head(query_emb)
    assert out.shape == (B, n_query, fut_steps, 2)
    assert torch.isfinite(out).all(), "agent_fut_pred 含 NaN/Inf"


# ---------------------------------------------------------------------------
# 属性 4：损失零点
# ---------------------------------------------------------------------------

@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(M=st.integers(1, MAX_GT), seed=st.integers(0, 10_000))
def test_loss_zero_point(M, seed):
    """Feature: nano-fsd-v2, Property 4 (motion_loss is zero at the GT).

    构造单 batch：
      - M 个有效 GT 框放到互相远离的网格上；对应 GT 未来轨迹 `gt_fut` 随机、未来 mask 随机；
      - 预测框把这 M 个 GT 框精确拷到 M 个随机 query 槽位（建立确定匹配 query->gt）；
      - 运动头输出 `agent_fut_pred` 在那些匹配 query 上精确等于对应 `gt_fut`。
    此时 masked MSE 的分子逐项为 0 → `motion_loss` **精确为 0**。

    注意：motion_loss 内部用 det['box'] 与 gt_box/gt_mask 重算匹配，故只需保证匹配 query
    的预测轨迹对上即可；其余 query 的预测轨迹任意（不参与分子）。

    **Validates: 正确性属性 4；需求 3.4**
    """
    gen = torch.Generator().manual_seed(seed)
    gt_box, gt_mask = _random_gt_box(M, gen)
    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()      # (M,)

    # GT 未来轨迹与未来有效步（随机前缀有效，模拟 instance 中途消失）
    gt_fut = torch.zeros(MAX_GT, TF, 2)
    gt_fut_mask = torch.zeros(MAX_GT, TF, dtype=torch.bool)
    for k in range(M):
        g = int(valid_gt_idx[k])
        gt_fut[g] = (torch.rand(TF, 2, generator=gen) - 0.5) * 2.0
        valid_steps = int(torch.randint(0, TF + 1, (1,), generator=gen))
        gt_fut_mask[g, :valid_steps] = True

    # 预测框：默认远处背景噪声；M 个 query 精确拷贝 GT 框 -> 确定匹配
    pred_box = torch.full((N, 5), 1000.0)
    perm = torch.randperm(N, generator=gen)[:M]
    for k in range(M):
        pred_box[int(perm[k])] = gt_box[int(valid_gt_idx[k])]

    # 预测轨迹：匹配 query 精确等于对应 GT 轨迹；其余 query 任意（放大噪声，确认不被计入）
    pred_fut = torch.randn(N, TF, 2) * 100.0
    for k in range(M):
        pred_fut[int(perm[k])] = gt_fut[int(valid_gt_idx[k])]

    det = {"box": pred_box.unsqueeze(0)}
    loss = motion_loss(pred_fut.unsqueeze(0), det,
                       gt_fut.unsqueeze(0), gt_fut_mask.unsqueeze(0),
                       gt_box.unsqueeze(0), gt_mask.unsqueeze(0))

    assert torch.isfinite(loss), "loss 含 NaN/Inf"
    assert loss.item() == pytest.approx(0.0, abs=1e-6), \
        f"损失零点不满足：loss={loss.item():.3e}"


# ---------------------------------------------------------------------------
# 属性 5：mask 单调性（增加无效 agent 不改变有效位 loss）
# ---------------------------------------------------------------------------

@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(M=st.integers(1, 25), extra=st.integers(1, 25), seed=st.integers(0, 10_000))
def test_mask_monotonicity(M, extra, seed):
    """Feature: nano-fsd-v2, Property 5 (invalid pad agents do not change loss).

    先对 M 个有效 agent 算一次 `motion_loss`（基线）；再往空 pad 槽位塞入 `extra` 个
    「存在但无效」的 GT agent（`agent_mask=False`、`agent_fut_mask` 全 False，框/轨迹填
    任意噪声），重算 loss。两次 loss 必须**完全相等**：无效 agent 不参与匹配（match 只看
    gt_mask），也就不进入 masked MSE 的分子/分母。

    关键：**模型侧输入（det['box']、agent_fut_pred）在两次调用中保持完全一致**，只在 GT
    侧的无效 pad 槽位（gt_box/gt_fut）注入噪声。这样「增加无效 agent」严格对应「在 GT 列表
    尾部追加 mask=False 的占位 agent」，而匹配只对有效 GT 进行，预测侧不受影响。

    **Validates: 正确性属性 5；需求 3.4**
    """
    # extra 不能超过剩余 pad 槽位
    extra = min(extra, MAX_GT - M)
    if extra <= 0:
        extra = 0  # M 已占满时退化为「不加无效 agent」，loss 自然相等

    gen = torch.Generator().manual_seed(seed)
    gt_box, gt_mask = _random_gt_box(M, gen)
    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()      # (M,)

    gt_fut = torch.zeros(MAX_GT, TF, 2)
    gt_fut_mask = torch.zeros(MAX_GT, TF, dtype=torch.bool)
    for k in range(M):
        g = int(valid_gt_idx[k])
        gt_fut[g] = (torch.rand(TF, 2, generator=gen) - 0.5) * 2.0
        # 至少 1 步有效，保证基线 loss 的分母 > 0（否则空场景 loss 恒 0，比较失去意义）
        valid_steps = int(torch.randint(1, TF + 1, (1,), generator=gen))
        gt_fut_mask[g, :valid_steps] = True

    # 预测：匹配 query 的轨迹 = GT + 固定偏移，使基线 loss 为非零确定值
    pred_box = torch.full((N, 5), 1000.0)
    perm = torch.randperm(N, generator=gen)[:M]
    pred_fut = torch.zeros(N, TF, 2)
    for k in range(M):
        q = int(perm[k])
        g = int(valid_gt_idx[k])
        pred_box[q] = gt_box[g]
        pred_fut[q] = gt_fut[g] + (torch.rand(TF, 2, generator=gen))  # 非零误差

    det = {"box": pred_box.unsqueeze(0)}

    # --- 基线：仅 M 个有效 agent ---
    loss_base = motion_loss(pred_fut.unsqueeze(0), det,
                            gt_fut.unsqueeze(0), gt_fut_mask.unsqueeze(0),
                            gt_box.unsqueeze(0), gt_mask.unsqueeze(0))

    assert loss_base.item() > 0, "构造失败：基线 loss 应为非零（便于验证不变性）"

    # --- 加入 extra 个「存在但无效」的 pad GT agent（仅改 GT 侧，模型侧输入不动）---
    gt_box2 = gt_box.clone()
    gt_mask2 = gt_mask.clone()           # 保持不变：新增槽位 mask 仍为 False
    gt_fut2 = gt_fut.clone()
    gt_fut_mask2 = gt_fut_mask.clone()   # 保持不变：新增槽位未来步 mask 仍为 False
    empty_slots = torch.nonzero(~gt_mask, as_tuple=False).flatten()[:extra]
    for s in empty_slots.tolist():
        # 无效 agent：mask 仍为 False，但框/轨迹填入任意噪声（存在但无效）
        gt_box2[s] = torch.randn(5, generator=gen)
        gt_fut2[s] = torch.randn(TF, 2, generator=gen) * 50.0
        # agent_mask / agent_fut_mask 保持 False（关键：无效 -> 不参与匹配）

    # 模型侧（det['box'] 与 agent_fut_pred）与基线完全一致，确保差异只来自无效 GT。
    loss_more = motion_loss(pred_fut.unsqueeze(0), det,
                            gt_fut2.unsqueeze(0), gt_fut_mask2.unsqueeze(0),
                            gt_box2.unsqueeze(0), gt_mask2.unsqueeze(0))

    # 有效位 loss 完全不变（无效 pad agent 不参与匹配/不计入 masked MSE）
    assert loss_more.item() == pytest.approx(loss_base.item(), rel=1e-6, abs=1e-7), \
        f"mask 单调性被破坏：base={loss_base.item():.6e} more={loss_more.item():.6e}"


def test_mask_monotonicity_adding_to_empty_scene_stays_zero():
    """空场景加入无效 agent 后 loss 仍为 0（M=0 基线与属性 6 的交叉验证）。

    无任何有效 GT 时 motion_loss=0；塞入若干无效 pad agent 后仍应为 0，不被噪声污染。

    **Validates: 正确性属性 5；需求 3.4**
    """
    gen = torch.Generator().manual_seed(7)

    gt_box = torch.zeros(MAX_GT, 5)
    gt_mask = torch.zeros(MAX_GT, dtype=torch.bool)       # 全无效 -> M=0
    gt_fut = torch.zeros(MAX_GT, TF, 2)
    gt_fut_mask = torch.zeros(MAX_GT, TF, dtype=torch.bool)

    pred_box = torch.randn(1, N, 5)
    pred_fut = torch.randn(1, N, TF, 2)
    det = {"box": pred_box}

    loss_empty = motion_loss(pred_fut, det, gt_fut.unsqueeze(0),
                             gt_fut_mask.unsqueeze(0), gt_box.unsqueeze(0),
                             gt_mask.unsqueeze(0))
    assert loss_empty.item() == pytest.approx(0.0, abs=1e-7)

    # 塞入噪声但 mask 全 False 的 pad agent
    gt_box2 = torch.randn(MAX_GT, 5, generator=gen)
    gt_fut2 = torch.randn(MAX_GT, TF, 2, generator=gen) * 50.0
    loss_still = motion_loss(pred_fut, det, gt_fut2.unsqueeze(0),
                             gt_fut_mask.unsqueeze(0), gt_box2.unsqueeze(0),
                             gt_mask.unsqueeze(0))
    assert loss_still.item() == pytest.approx(0.0, abs=1e-7)
