"""任务 2.4 单元测试：用合成内存缓存验证 Dataset / DataLoader 的长度与 batch 形状。"""

import os
import sys
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from data import Batch, NanoFSDDataset, Sample, convert_raw_samples, make_loader  # noqa: E402
from test_data_cache import make_raw  # noqa: E402

CFG = Config()
N, TH, C, TF = CFG.n_agents, CFG.hist_steps, CFG.agent_feat_dim, CFG.fut_steps
S = 10


def synthetic_cache() -> dict:
    raws = [make_raw(n=3 + i, tf=TF - (i % 3), seed=i) for i in range(S)]
    cache, _ = convert_raw_samples(raws, CFG)
    return cache


def test_dataset_len_and_item():
    cache = synthetic_cache()
    ds = NanoFSDDataset(cache)
    assert len(ds) == S
    s = ds[3]
    assert isinstance(s, Sample)
    assert s.ego_hist.shape == (TH, 4) and s.agents.shape == (N, TH, C)
    assert torch.equal(s.traj_gt, cache["traj_gt"][3])


def test_loader_batches_batch_dim_first(tmp_path):
    cache = synthetic_cache()
    p = tmp_path / "c.pt"
    torch.save(cache, p)
    cfg = replace(CFG, batch_size=4)
    loader = make_loader(cfg, str(p), shuffle=False)   # 也覆盖从 .pt 路径加载
    batches = list(loader)
    assert [b.ego_hist.shape[0] for b in batches] == [4, 4, 2]
    b = batches[0]
    assert isinstance(b, Batch)
    assert b.ego_hist.shape == (4, TH, 4)
    assert b.agents.shape == (4, N, TH, C)
    assert b.agent_mask.shape == (4, N) and b.agent_mask.dtype == torch.bool
    assert b.agent_gt.shape == (4, N, 2)
    assert b.traj_gt.shape == (4, TF, 2)
    assert b.traj_mask.shape == (4, TF) and b.traj_mask.dtype == torch.bool
    # shuffle=False 时第一个 batch 即缓存前 4 个样本
    assert torch.equal(b.agents, cache["agents"][:4])
    assert b.to("cpu").traj_gt.shape == (4, TF, 2)
