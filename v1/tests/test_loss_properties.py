"""任务 4.3 损失函数属性测试（hypothesis）。

属性 4 损失零点：随机 gt，令 pred=gt，断言 masked_mse / planning_loss ≈ 0。
属性 5 mask 单调性：在有效位内容不变的前提下，追加更多 padding（无效位）
  并在其中填入任意垃圾值，有效位上的 loss 值不变——证明 masked 的分母是
  有效元素计数，padding 不会稀释 loss。

额外：错误处理场景 2——mask 全 False（空场景）时 loss 精确为 0 且不报错。

说明：损失为纯向量化计算，开销很低，故 max_examples 取 100、deadline=None。
生成的浮点值限定有限、无 NaN、落在 [-5, 5]，避免数值溢出干扰断言。
"""

import os
import sys

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import masked_mse, planning_loss  # noqa: E402

TOL = 1e-6
VAL = dict(min_value=-5.0, max_value=5.0,
           allow_nan=False, allow_infinity=False, width=32)


def _rand(gen: torch.Generator, *shape) -> torch.Tensor:
    """[-5, 5] 均匀分布随机张量（有限、无 NaN）。"""
    return (torch.rand(*shape, generator=gen) * 10.0 - 5.0).float()


# ---------------------------------------------------------------------------
# 属性 4：损失零点
# ---------------------------------------------------------------------------

@st.composite
def masked_mse_inputs(draw):
    """随机 (B, Tf) 形状 + K 维输出 + 随机 bool mask + 种子。"""
    b = draw(st.integers(1, 4), label="B")
    t = draw(st.integers(1, 12), label="Tf")
    k = draw(st.integers(1, 3), label="K")
    seed = draw(st.integers(0, 2**31 - 1), label="seed")
    return b, t, k, seed


@settings(max_examples=100, deadline=None)
@given(inp=masked_mse_inputs())
def test_masked_mse_zero_when_pred_equals_gt(inp):
    """Feature: nano-fsd, Property 4 (loss zero point).

    **Validates: 正确性属性 4**

    随机 gt，令 pred=gt，masked_mse 对任意 mask 都应 ≈ 0。
    """
    b, t, k, seed = inp
    g = torch.Generator().manual_seed(seed)
    gt = _rand(g, b, t, k)
    pred = gt.clone()
    mask = torch.rand(b, t, generator=g) > 0.5

    loss = masked_mse(pred, gt, mask)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=TOL), \
        f"pred==gt 时 masked_mse 应为 0，实得 {loss.item()}"


@st.composite
def planning_inputs(draw):
    """随机 (B, Tf, 2) 轨迹输入 + 随机 traj_mask + 种子。"""
    b = draw(st.integers(1, 4), label="B")
    tf = draw(st.integers(1, 12), label="Tf")
    seed = draw(st.integers(0, 2**31 - 1), label="seed")
    return b, tf, seed


@settings(max_examples=100, deadline=None)
@given(inp=planning_inputs())
def test_planning_loss_zero_when_pred_equals_gt(inp):
    """Feature: nano-fsd, Property 4 (loss zero point).

    **Validates: 正确性属性 4**

    随机 traj_gt，令 traj_pred=traj_gt，planning_loss 对任意 traj_mask 都应 ≈ 0。
    """
    b, tf, seed = inp
    g = torch.Generator().manual_seed(seed)
    traj_gt = _rand(g, b, tf, 2)
    traj_pred = traj_gt.clone()
    traj_mask = torch.rand(b, tf, generator=g) > 0.5

    loss = planning_loss(traj_pred, traj_gt, traj_mask)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=TOL), \
        f"traj_pred==traj_gt 时 planning_loss 应为 0，实得 {loss.item()}"


# ---------------------------------------------------------------------------
# 属性 5：mask 单调性（追加 padding 不改变有效位 loss）
# ---------------------------------------------------------------------------

@st.composite
def extend_inputs(draw):
    """基础输入 + 追加的 padding 行数（全部无效，填任意垃圾值）。"""
    b = draw(st.integers(1, 4), label="B")
    t_valid = draw(st.integers(1, 10), label="Tf_valid")
    k = draw(st.integers(1, 3), label="K")
    extra = draw(st.integers(1, 8), label="extra_pad")
    seed = draw(st.integers(0, 2**31 - 1), label="seed")
    return b, t_valid, k, extra, seed


@settings(max_examples=100, deadline=None)
@given(inp=extend_inputs())
def test_masked_mse_padding_monotonicity(inp):
    """Feature: nano-fsd, Property 5 (mask monotonicity).

    **Validates: 正确性属性 5**

    基础输入在时间维上全部有效，计算 base loss；扩展版在尾部追加 extra 个
    无效位（mask=False）并填入任意垃圾 pred/gt。有效位内容逐元素不变，
    则两次 loss 必须精确相等——padding 不稀释 masked 分母。
    """
    b, t_valid, k, extra, seed = inp
    g = torch.Generator().manual_seed(seed)

    # 有效部分：随机 pred/gt，mask 全 True
    pred_v = _rand(g, b, t_valid, k)
    gt_v = _rand(g, b, t_valid, k)
    mask_v = torch.ones(b, t_valid, dtype=torch.bool)
    base = masked_mse(pred_v, gt_v, mask_v)

    # 扩展：尾部追加 extra 个无效位，内容为任意垃圾（放大量级以凸显若被计入会出错）
    pred_pad = _rand(g, b, extra, k) * 100.0
    gt_pad = _rand(g, b, extra, k) * 100.0
    pred_e = torch.cat([pred_v, pred_pad], dim=1)
    gt_e = torch.cat([gt_v, gt_pad], dim=1)
    mask_e = torch.cat([mask_v, torch.zeros(b, extra, dtype=torch.bool)], dim=1)
    extended = masked_mse(pred_e, gt_e, mask_e)

    assert torch.isclose(base, extended, atol=TOL), \
        f"追加 padding 改变了有效位 loss: base={base.item()} extended={extended.item()}"


@settings(max_examples=100, deadline=None)
@given(inp=extend_inputs())
def test_planning_loss_padding_monotonicity(inp):
    """Feature: nano-fsd, Property 5 (mask monotonicity).

    **Validates: 正确性属性 5**

    对 planning_loss 同样成立：在有效未来步之后追加无效步（traj_mask=False）
    并填任意垃圾轨迹值，有效步上的 loss 不变。
    """
    b, t_valid, _, extra, seed = inp
    g = torch.Generator().manual_seed(seed)

    pred_v = _rand(g, b, t_valid, 2)
    gt_v = _rand(g, b, t_valid, 2)
    mask_v = torch.ones(b, t_valid, dtype=torch.bool)
    base = planning_loss(pred_v, gt_v, mask_v)

    pred_pad = _rand(g, b, extra, 2) * 100.0
    gt_pad = _rand(g, b, extra, 2) * 100.0
    pred_e = torch.cat([pred_v, pred_pad], dim=1)
    gt_e = torch.cat([gt_v, gt_pad], dim=1)
    mask_e = torch.cat([mask_v, torch.zeros(b, extra, dtype=torch.bool)], dim=1)
    extended = planning_loss(pred_e, gt_e, mask_e)

    assert torch.isclose(base, extended, atol=TOL), \
        f"追加 padding 改变了有效步 loss: base={base.item()} extended={extended.item()}"


# ---------------------------------------------------------------------------
# 错误处理场景 2：全 False mask -> loss 精确为 0，不报错（除零保护）
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(inp=masked_mse_inputs())
def test_all_false_mask_yields_zero(inp):
    """Feature: nano-fsd, Property 5 / 错误处理场景 2 (empty scene).

    **Validates: 正确性属性 5**

    mask 全 False（空场景）时，masked_mse 借 clamp(min=1) 防除零，
    对任意 pred/gt 精确返回 0 且不抛异常。
    """
    b, t, k, seed = inp
    g = torch.Generator().manual_seed(seed)
    pred = _rand(g, b, t, k) * 100.0   # 任意量级，仍应被完全屏蔽
    gt = _rand(g, b, t, k) * 100.0
    mask = torch.zeros(b, t, dtype=torch.bool)

    loss = masked_mse(pred, gt, mask)
    assert loss.item() == 0.0, f"全 False mask 应返回精确 0，实得 {loss.item()}"


def test_zero_point_smoke():
    """最小冒烟测试：固定输入下 pred=gt 的 masked_mse / planning_loss 为 0。"""
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[True, False]])
    assert masked_mse(pred, pred.clone(), mask).item() == 0.0
    assert planning_loss(pred, pred.clone(), mask).item() == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
