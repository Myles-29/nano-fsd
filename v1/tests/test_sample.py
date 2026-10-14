"""任务 2.1 单元测试：Sample 的形状与验证规则（ego 原点、padding 全零、相邻步距）。"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from data import NanoFSDDataset, Sample  # noqa: E402

CFG = Config()
N, TH, TF, C = CFG.n_agents, CFG.hist_steps, CFG.fut_steps, CFG.agent_feat_dim
CACHE = "data/cache_mini.pt"


def fields(k: int = 3) -> dict:
    """合法样本字段（标准化单位）：前 k 个 agent 有效，ego 当前帧在原点。"""
    ego = torch.zeros(TH, 4)
    ego[:, 0] = torch.arange(-(TH - 1), 1) * 0.05
    agents = torch.zeros(N, TH, C)
    agents[:k] = torch.rand(k, TH, C)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:k] = True
    agent_gt = torch.zeros(N, 2)
    agent_gt[:k] = agents[:k, -1, :2]
    traj = torch.stack([torch.arange(1, TF + 1) * 0.05, torch.zeros(TF)], -1)
    return dict(ego_hist=ego, agents=agents, agent_mask=mask, agent_gt=agent_gt,
                traj_gt=traj, traj_mask=torch.ones(TF, dtype=torch.bool))


def test_valid_sample_ok():
    s = Sample(**fields())
    assert s.agents.shape == (N, TH, C) and s.traj_gt.shape == (TF, 2)


def test_ego_not_at_origin_rejected():
    f = fields()
    f["ego_hist"][-1, :2] = torch.tensor([0.1, 0.0])
    with pytest.raises(AssertionError, match="原点"):
        Sample(**f)


def test_padding_row_nonzero_rejected():
    f = fields(k=3)
    f["agents"][5, 0, 0] = 0.3
    with pytest.raises(AssertionError, match="全零"):
        Sample(**f)


def test_large_step_rejected_but_masked_step_ignored():
    f = fields()
    f["traj_gt"][6:] += 1.0                     # 第 5->6 步跳 1.0（标准化）= 50m
    with pytest.raises(AssertionError, match="相邻轨迹点"):
        Sample(**f)
    f["traj_mask"][6:] = False                  # 跳变落在 padding 段 -> 不检查
    f["traj_gt"][6:] = 0.0
    Sample(**f)


def test_bad_shape_raises_assertion():
    f = fields()
    f["ego_hist"] = torch.zeros(TH)
    with pytest.raises(AssertionError):
        Sample(**f)


@pytest.mark.skipif(not os.path.exists(CACHE), reason="缓存不存在")
def test_all_cached_samples_validate():
    ds = NanoFSDDataset(CACHE)
    for i in range(len(ds)):
        ds[i]  # __getitem__ 构造 Sample 时执行 validate()
