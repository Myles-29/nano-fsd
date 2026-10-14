"""Feature: nano-fsd, Property 7 (overfit feasibility - perception).

**Validates: 正确性属性 7**

单 batch 过拟合 sanity check（感知头 + backbone 联合）。

这是一个学习型集成测试（非 hypothesis 属性测试）：固定一个合成 batch，用
mode="perception" 反复训练若干步，断言感知 loss 从初始值被显著压低、接近 0——
证明模型容量足够、backbone→感知头的梯度链路通畅（算法 A 的最小验证）。

全程 CPU、确定性（manual_seed），构造自包含的合成 batch，不依赖
data/cache_mini.pt，因此可在任何环境下运行。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from model import NanoFSD  # noqa: E402
from train import masked_mse  # noqa: E402


def _make_fixed_batch(cfg: Config, B: int = 4):
    """构造一个固定的合成感知 batch（ego 坐标系、标准化量级）。

    - ego_hist (B, Th, 4)       随机自车历史
    - agents   (B, N, Th, C)    随机 agent 历史，padding 行置零（镜像真实数据约定）
    - agent_mask (B, N) bool     每个样本保留前若干 agent 为有效，其余 padding
    - agent_gt (B, N, 2)         随机目标相对位置（感知头监督信号）
    """
    N, Th, C = cfg.n_agents, cfg.hist_steps, cfg.agent_feat_dim

    ego_hist = torch.randn(B, Th, 4) * 0.1
    agents = torch.randn(B, N, Th, C) * 0.5

    # 每个样本有效 agent 数不同，制造一些 padding
    agent_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_counts = [5, 8, 12, 3][:B]
    for b, k in enumerate(valid_counts):
        agent_mask[b, :k] = True

    # padding 行内容置零，镜像真实数据（损失按 mask 忽略，但保持一致更干净）
    agents = agents * agent_mask[:, :, None, None].float()

    agent_gt = torch.randn(B, N, 2) * 0.5
    return ego_hist, agents, agent_mask, agent_gt


def test_single_batch_overfit_perception():
    """Property 7：单 batch 反复训练，感知 loss 显著下降并接近 0。

    **Validates: 正确性属性 7**
    """
    torch.manual_seed(0)
    cfg = Config()  # 默认约 670k 参数，CPU 上几百步小 batch 可接受

    B = 4
    ego_hist, agents, agent_mask, agent_gt = _make_fixed_batch(cfg, B=B)

    model = NanoFSD(cfg)  # CPU
    model.train()
    # lr 略高以加速过拟合（单 batch 场景不担心泛化）
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 初始 loss（无梯度，仅记录）
    with torch.no_grad():
        pred0 = model(ego_hist, agents, agent_mask, mode="perception")
        initial_loss = masked_mse(pred0, agent_gt, agent_mask).item()

    # 固定 batch 反复训练
    steps = 300
    final_loss = initial_loss
    for _ in range(steps):
        pred = model(ego_hist, agents, agent_mask, mode="perception")
        loss = masked_mse(pred, agent_gt, agent_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    # 断言：显著下降（< 初始的 10%）且接近 0
    assert final_loss < initial_loss * 0.1, (
        f"感知 loss 未显著下降: initial={initial_loss:.6f} final={final_loss:.6f}")
    assert final_loss < 0.05, (
        f"感知 loss 未接近 0: final={final_loss:.6f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
