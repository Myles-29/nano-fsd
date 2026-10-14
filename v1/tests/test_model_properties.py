"""任务 3.6 模型层属性测试（hypothesis）。

属性 1 形状守恒：随机合法输入下
  - Backbone 输出恒 (B, N+1, D)
  - PlanningHead / NanoFSD(mode="planning") 输出恒 (B, Tf, 2)
  - PerceptionHead / NanoFSD(mode="perception") 输出恒 (B, N, 2)
属性 2 padding 不变性：仅在 padding agent（agent_mask=False）内容上不同的两组输入，
  有效位（ego token + 有效 agent token）输出完全一致，证明 key_padding_mask 生效。

说明：Transformer 前向较慢，故 max_examples 取中等规模、deadline=None，
并在生成器里把 B 控制得较小。模型置 eval() + torch.no_grad() 以获得确定性输出
（Backbone 默认 dropout=0.0，eval() 更稳妥）。
"""

import os
import sys

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from model import Backbone, NanoFSD, PerceptionHead, PlanningHead  # noqa: E402

CFG = Config()
N, TH, TF, C, D = (CFG.n_agents, CFG.hist_steps, CFG.fut_steps,
                   CFG.agent_feat_dim, CFG.d_model)
TOL = 1e-4  # transformer 前向多层累积浮点误差，取较宽松容差


def _make_model(seed: int) -> NanoFSD:
    """用固定种子构建 NanoFSD 并置 eval()，保证前向确定性。"""
    torch.manual_seed(seed)
    model = NanoFSD(CFG)
    model.eval()
    return model


@st.composite
def valid_inputs(draw):
    """随机合法模型输入 (ego_hist, agents, agent_mask) + 模型种子。

    - B ∈ [1, 3]（控制前向开销）
    - 每个样本至少 1 个有效 agent（保证 padding 不变性可对比有效位）
    - 坐标量级落在标准化后 [-1,1] 附近（契约要求）
    """
    B = draw(st.integers(1, 3), label="B")
    seed = draw(st.integers(0, 2**31 - 1), label="seed")
    # 每个 batch 样本的有效 agent 数 ∈ [1, N]
    n_valid = draw(st.lists(st.integers(1, N), min_size=B, max_size=B),
                   label="n_valid")

    g = torch.Generator().manual_seed(seed)
    ego_hist = torch.randn(B, TH, 4, generator=g) * 0.5
    agents = torch.randn(B, N, TH, C, generator=g) * 0.5
    agent_mask = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        agent_mask[b, :n_valid[b]] = True  # 前 n_valid 个有效，其余为 padding
    return ego_hist, agents, agent_mask, seed


@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(inp=valid_inputs())
def test_shape_conservation(inp):
    """Feature: nano-fsd, Property 1 (shape conservation).

    **Validates: 正确性属性 1**
    """
    ego_hist, agents, agent_mask, seed = inp
    B = ego_hist.shape[0]
    model = _make_model(seed)

    with torch.no_grad():
        scene_emb = model.backbone(ego_hist, agents, agent_mask)
        traj_pred = model.planning_head(scene_emb)
        agent_pred = model.perception_head(scene_emb)
        traj_top = model(ego_hist, agents, agent_mask, mode="planning")
        agent_top = model(ego_hist, agents, agent_mask, mode="perception")

    # Backbone 输出恒 (B, N+1, D)
    assert scene_emb.shape == (B, N + 1, D)
    # 规划头 / 顶层 planning 输出恒 (B, Tf, 2)
    assert traj_pred.shape == (B, TF, 2)
    assert traj_top.shape == (B, TF, 2)
    # 感知头 / 顶层 perception 输出恒 (B, N, 2)
    assert agent_pred.shape == (B, N, 2)
    assert agent_top.shape == (B, N, 2)
    # 顶层与单独调用头结果一致（同一 scene_emb 的确定性）
    assert torch.allclose(traj_top, traj_pred, atol=TOL)
    assert torch.allclose(agent_top, agent_pred, atol=TOL)


@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(inp=valid_inputs())
def test_padding_invariance(inp):
    """Feature: nano-fsd, Property 2 (padding invariance).

    **Validates: 正确性属性 2**

    构造两组输入：有效位内容完全相同，仅 padding agent（mask=False）内容不同。
    期望：规划输出完全一致（规划头读 ego token），感知输出在有效 agent 位上完全一致
    （key_padding_mask 使 padding token 不参与注意力，故不影响有效位）。
    """
    ego_hist, agents, agent_mask, seed = inp
    B = ego_hist.shape[0]

    # 若某样本没有 padding 槽位（全部有效），无法体现差异，跳过该例
    if bool(agent_mask.all()):
        return

    model = _make_model(seed)

    # 第二组输入：复制有效位，padding 位填入不同的随机垃圾值
    g = torch.Generator().manual_seed(seed + 1)
    pad_noise = torch.randn(B, N, TH, C, generator=g) * 3.0
    pad_slots = ~agent_mask  # (B, N) True=padding
    agents_b = agents.clone()
    mask_expand = pad_slots[:, :, None, None].expand_as(agents_b)
    agents_b[mask_expand] = pad_noise[mask_expand]

    # 有效位必须逐元素相同（前提校验，避免误伤有效内容）
    valid_expand = agent_mask[:, :, None, None].expand_as(agents)
    assert torch.equal(agents[valid_expand], agents_b[valid_expand])

    with torch.no_grad():
        traj_a = model(ego_hist, agents, agent_mask, mode="planning")
        traj_b = model(ego_hist, agents_b, agent_mask, mode="planning")
        perc_a = model(ego_hist, agents, agent_mask, mode="perception")
        perc_b = model(ego_hist, agents_b, agent_mask, mode="perception")

    # 规划输出完全一致
    assert torch.allclose(traj_a, traj_b, atol=TOL), \
        f"planning 输出受 padding 内容影响: max diff={torch.abs(traj_a - traj_b).max()}"

    # 感知输出在有效 agent 位上完全一致
    valid_pred = agent_mask[:, :, None].expand_as(perc_a)
    diff = torch.abs(perc_a - perc_b)[valid_pred]
    assert torch.all(diff < TOL), \
        f"有效 agent 感知输出受 padding 内容影响: max diff={diff.max()}"


def test_component_shapes_smoke():
    """最小冒烟测试：直接实例化各组件验证形状（不经 hypothesis）。"""
    torch.manual_seed(0)
    B = 2
    ego_hist = torch.randn(B, TH, 4)
    agents = torch.randn(B, N, TH, C)
    agent_mask = torch.ones(B, N, dtype=torch.bool)
    agent_mask[:, N // 2:] = False

    backbone = Backbone(CFG).eval()
    perception = PerceptionHead(CFG).eval()
    planning = PlanningHead(CFG).eval()
    with torch.no_grad():
        emb = backbone(ego_hist, agents, agent_mask)
        assert emb.shape == (B, N + 1, D)
        assert perception(emb).shape == (B, N, 2)
        assert planning(emb).shape == (B, TF, 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
