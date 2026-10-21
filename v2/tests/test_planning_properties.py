"""任务 4.3 交互式规划头属性测试（hypothesis）。

针对 `model.py` 的 `PlanningHeadV2`，验证两条正确性属性：

- **属性 1 形状守恒**：随机合法输入（bev / agent_emb / agent_score / cmd_emb，含
  `cmd_emb=None` 情形）下，`PlanningHeadV2` 输出恒为自车轨迹 `traj (B,Tf,2)`，且全为
  有限浮点（无 NaN/Inf）。

- **属性 8 规划交互敏感性**：对**同一** bev、**同一** agent_emb，比较两种输入：
    (a) 高置信度 agent（agent_score 取较大正值，如 +5 → sigmoid≈1，agent 分支正常注入）；
    (b) agent 全背景（agent_score 取很负值，如 -50 → sigmoid≈0 → gate≈0，退化为仅 BEV）。
  两条轨迹必须**有意义地不同**（差异超过浮点噪声阈值 1e-4），证明规划确实用到了 agent
  信息，而非忽略。同时断言「全背景」情形 **NaN-safe / 有限**（优雅退化不报错）。

  鲁棒性说明：属性 8 依赖 agent 注意力分支在高置信度下产生非零 refinement。默认随机初始化
  即应如此；为复现稳定，这里**确定性 seed** 构造规划头，并对 agent_emb 做适度缩放，确保
  高置信度与全背景两种情形的输出存在可检测差异（gate 从 ~0 变到 ~1，refinement 被注入）。

  附加（退化等价性）：全背景输出应与「agent 信息缺席」（gate 强制为 0 的等价构造）一致，
  这里用「agent_emb 全零 + 全背景 score」交叉印证退化到仅 BEV 规划。

记号：B=batch、D=128、N=50（agent 数）、Tf=12（fut_steps）、BEV 边长 S=100、kv_stride=2。

验证：正确性属性 1、8；需求 4.3, 4.4。
"""

import os
import sys

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PlanningHeadV2  # noqa: E402

# 全局约定（与 spec 一致）
D = 128           # 特征维度
N = 50            # agent 数（query 数）
TF = 12           # 未来步数（fut_steps）
BEV = 100         # BEV 边长 S
KV_STRIDE = 2     # BEV token 下采样步长


def _make_head(fut_steps: int = TF, seed: int = 0) -> PlanningHeadV2:
    """构造一个确定性初始化的规划头（固定 seed，便于复现属性 8 的敏感性）。"""
    torch.manual_seed(seed)
    return PlanningHeadV2(d_model=D, fut_steps=fut_steps,
                          bev_size=BEV, kv_stride=KV_STRIDE)


# ---------------------------------------------------------------------------
# 属性 1：形状守恒
# ---------------------------------------------------------------------------

@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(1, 3), n_agent=st.sampled_from([1, 8, N]),
       fut_steps=st.sampled_from([1, 6, TF]), use_cmd=st.booleans(),
       seed=st.integers(0, 10_000))
def test_shape_conservation(B, n_agent, fut_steps, use_cmd, seed):
    """Feature: nano-fsd-v2, Property 1 (PlanningHeadV2 output shape).

    随机 bev (B,D,S,S)、agent_emb (B,n_agent,D)、agent_score (B,n_agent,1) logit、
    cmd_emb (B,D) 或 None 下，输出恒为 `traj (B,fut_steps,2)` 且全有限（无 NaN/Inf）。

    覆盖 `cmd_emb=None` 与非 None 两种分支（use_cmd 布尔切换）。

    **Validates: 正确性属性 1；需求 4.3, 4.4**
    """
    gen = torch.Generator().manual_seed(seed)
    head = _make_head(fut_steps=fut_steps, seed=seed).eval()

    bev = torch.randn(B, D, BEV, BEV, generator=gen)
    agent_emb = torch.randn(B, n_agent, D, generator=gen)
    # agent_score 为任意 logit（含正负），覆盖高/低置信度混合
    agent_score = (torch.randn(B, n_agent, 1, generator=gen)) * 5.0
    cmd_emb = torch.randn(B, D, generator=gen) if use_cmd else None

    with torch.no_grad():
        traj = head(bev, agent_emb, agent_score, cmd_emb)

    assert traj.shape == (B, fut_steps, 2)
    assert torch.isfinite(traj).all(), "traj 含 NaN/Inf"


# ---------------------------------------------------------------------------
# 属性 8：规划交互敏感性（有 agent vs agent 全背景）
# ---------------------------------------------------------------------------

@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(1, 3), use_cmd=st.booleans(), seed=st.integers(0, 10_000))
def test_planning_interaction_sensitivity(B, use_cmd, seed):
    """Feature: nano-fsd-v2, Property 8 (planning uses agent information).

    对同一 bev、同一 agent_emb：
      (a) 高置信度 agent：agent_score=+5（sigmoid≈1，gate≈1 → agent refinement 正常注入）；
      (b) agent 全背景：agent_score=-50（sigmoid≈0，gate≈0 → refinement 被清零，退化仅 BEV）。
    两条轨迹必须有意义地不同（max 绝对差 > 1e-4）；且全背景情形必须有限（NaN-safe 退化）。

    **Validates: 正确性属性 8；需求 4.3, 4.4**
    """
    gen = torch.Generator().manual_seed(seed)
    head = _make_head(seed=seed).eval()

    bev = torch.randn(B, D, BEV, BEV, generator=gen)
    # agent_emb 适度缩放，保证高置信度分支的 refinement 可检测（非零输入）
    agent_emb = torch.randn(B, N, D, generator=gen) * 2.0
    cmd_emb = torch.randn(B, D, generator=gen) if use_cmd else None

    score_high = torch.full((B, N, 1), 5.0)      # sigmoid(+5)≈0.993 → gate≈1
    score_bg = torch.full((B, N, 1), -50.0)      # sigmoid(-50)≈0   → gate≈0

    with torch.no_grad():
        traj_high = head(bev, agent_emb, score_high, cmd_emb)
        traj_bg = head(bev, agent_emb, score_bg, cmd_emb)

    # 全背景优雅退化：输出有限（不 NaN/Inf）
    assert torch.isfinite(traj_bg).all(), "agent 全背景输出含 NaN/Inf（退化不安全）"
    assert torch.isfinite(traj_high).all(), "高置信度输出含 NaN/Inf"

    # 两种输入输出必须有意义地不同（超过浮点噪声）
    max_diff = (traj_high - traj_bg).abs().max().item()
    assert max_diff > 1e-4, (
        f"规划未体现 agent 交互敏感性：max|Δtraj|={max_diff:.3e} 过小，"
        f"高置信度与全背景输出几乎一致")


@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(1, 2), seed=st.integers(0, 10_000))
def test_all_background_degrades_to_bev_only(B, seed):
    """Feature: nano-fsd-v2, Property 8 (all-background ≈ agent-absent / BEV-only).

    附加退化等价性检查：当 agent 全背景（score 很负 → gate≈0），agent refinement 被清零，
    规划应退化为「仅 BEV 规划」。用「agent_emb 全零 + 全背景 score」作为「agent 信息缺席」
    的等价构造，二者输出应几乎一致（差异在浮点容差内）——印证 gate 确实把 agent 分支抹掉，
    且该退化 NaN-safe。

    **Validates: 正确性属性 8；需求 4.3**
    """
    gen = torch.Generator().manual_seed(seed)
    head = _make_head(seed=seed).eval()

    bev = torch.randn(B, D, BEV, BEV, generator=gen)
    agent_emb = torch.randn(B, N, D, generator=gen) * 2.0
    score_bg = torch.full((B, N, 1), -50.0)       # gate≈0

    with torch.no_grad():
        traj_bg = head(bev, agent_emb, score_bg, None)
        # agent 信息缺席的等价构造：emb 全零 + 全背景 score（gate≈0）
        traj_absent = head(bev, torch.zeros(B, N, D), score_bg, None)

    assert torch.isfinite(traj_bg).all()
    assert torch.isfinite(traj_absent).all()

    # 两者都应退化为仅 BEV（gate≈0 抹掉 agent 分支），差异在容差内
    max_diff = (traj_bg - traj_absent).abs().max().item()
    assert max_diff < 1e-3, (
        f"全背景未完全退化为仅 BEV：max|Δtraj|={max_diff:.3e} 偏大")
