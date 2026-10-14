"""任务 2.3 单元测试：用合成 RawSample 验证标准化 / padding / 截断 / 过滤 / 缓存。"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from data import (SCENE_SCALE, RawSample, convert_raw_samples, denormalize,  # noqa: E402
                  load_cache, normalize, raw_to_sample)

CFG = Config()
TH, C = CFG.hist_steps, CFG.agent_feat_dim


def make_raw(n: int, tf: int, seed: int = 0) -> RawSample:
    """合成 RawSample：ego 匀速前进（当前帧在原点），n 个 agent 距离各不相同。"""
    rng = np.random.default_rng(seed)
    t = np.arange(-(TH - 1), 1, dtype=np.float32)            # [-3..0]
    ego_hist = np.stack([5 * t * 0.5, 0 * t, np.full_like(t, 5), 0 * t], -1)
    agents = rng.uniform(-30, 30, (n, TH, C)).astype(np.float32)
    agents[:, :, 6:] = 0
    agents[:, :, 6] = 1                                       # one-hot vehicle
    agent_gt = agents[:, -1, :2].copy()
    traj_gt = np.stack([2.5 * np.arange(1, tf + 1), np.zeros(tf)], -1).astype(np.float32)
    return RawSample(ego_hist, agents, agent_gt, traj_gt, sample_token=f"tok{seed}")


def test_normalize_roundtrip():
    x = torch.randn(3, CFG.fut_steps, 2) * 40
    assert torch.allclose(denormalize(normalize(x)), x)
    assert torch.allclose(normalize(x), x / SCENE_SCALE)


def test_padding_to_N_with_mask():
    raw = make_raw(n=5, tf=CFG.fut_steps)
    s = raw_to_sample(raw, CFG)
    assert s.agents.shape == (CFG.n_agents, TH, C)
    assert s.agent_mask.sum() == 5 and s.agent_mask[:5].all()
    assert s.agents[5:].abs().max() == 0 and s.agent_gt[5:].abs().max() == 0
    # 位置/速度被缩放，cos/sin/one-hot 不变；排序后与原始数据集合一致
    order = np.argsort(np.linalg.norm(raw.agent_gt, axis=-1))
    np.testing.assert_allclose(s.agents[:5, :, :4].numpy() * SCENE_SCALE,
                               raw.agents[order][:, :, :4], rtol=1e-5)
    np.testing.assert_allclose(s.agents[:5, :, 4:].numpy(), raw.agents[order][:, :, 4:])
    np.testing.assert_allclose(denormalize(s.agent_gt[:5]).numpy(),
                               raw.agent_gt[order], rtol=1e-5)


def test_truncate_keeps_nearest_N():
    n = CFG.n_agents + 10
    raw = make_raw(n=n, tf=CFG.fut_steps, seed=1)
    s = raw_to_sample(raw, CFG)
    assert s.agent_mask.all()
    kept = torch.linalg.norm(s.agent_gt, dim=-1) * SCENE_SCALE
    all_d = np.sort(np.linalg.norm(raw.agent_gt, axis=-1))
    np.testing.assert_allclose(kept.numpy(), all_d[:CFG.n_agents], rtol=1e-5)


def test_traj_padding_with_mask():
    s = raw_to_sample(make_raw(n=3, tf=5), CFG)
    assert s.traj_gt.shape == (CFG.fut_steps, 2)
    assert s.traj_mask[:5].all() and not s.traj_mask[5:].any()
    assert s.traj_gt[5:].abs().max() == 0


def test_filter_no_agent_and_cache(tmp_path):
    raws = [make_raw(3, CFG.fut_steps, 0), make_raw(0, CFG.fut_steps, 1),
            make_raw(40, 7, 2)]
    assert raw_to_sample(raws[1], CFG) is None
    cache, stats = convert_raw_samples(raws, CFG)
    assert stats["kept"] == 2 and stats["no_agent"] == 1
    assert cache["agents"].shape == (2, CFG.n_agents, TH, C)
    assert cache["sample_tokens"] == ["tok0", "tok2"]
    p = tmp_path / "c.pt"
    torch.save(cache, p)
    loaded = load_cache(str(p))
    assert torch.equal(loaded["traj_mask"], cache["traj_mask"])
    assert loaded["scene_scale"] == SCENE_SCALE
