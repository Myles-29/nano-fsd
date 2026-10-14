"""Feature: nano-fsd, Property 3 (freeze correctness - dynamic).
**Validates: 正确性属性 3**

冻结端到端验证（动态部分）：跑若干个阶段二训练步，前后对比
backbone 与 perception_head 的全部参数逐元素相等；planning_head
至少有一个参数发生变化。

本测试自包含、CPU-only、确定性，不依赖任何磁盘缓存（合成一个小 batch）。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from model import NanoFSD  # noqa: E402
from train import planning_loss  # noqa: E402

CFG = Config()
N, TH, TF, C = CFG.n_agents, CFG.hist_steps, CFG.fut_steps, CFG.agent_feat_dim


def _snapshot(module: torch.nn.Module) -> dict:
    """按参数名快照（detach+clone）某模块下的全部参数张量。"""
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def _make_batch(B: int = 4):
    """合成一个小的阶段二训练 batch（标准化量级、部分 agent 有效）。"""
    ego_hist = torch.randn(B, TH, 4)
    agents = torch.randn(B, N, TH, C)

    # agent_mask: 每个样本保留前若干个有效，其余 padding；padding 行内容置零
    agent_mask = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        k = 3 + b  # 样本间有效数不同，但都 > 0
        agent_mask[b, :k] = True
    agents = agents * agent_mask[:, :, None, None].float()

    traj_gt = torch.randn(B, TF, 2)
    traj_mask = torch.ones(B, TF, dtype=torch.bool)
    return ego_hist, agents, agent_mask, traj_gt, traj_mask


def test_stage2_step_freezes_perception_stack_and_moves_planning_head():
    """阶段二训练步前后：感知栈参数逐元素不变，规划头至少一个参数变化。"""
    torch.manual_seed(0)

    model = NanoFSD(CFG)  # 默认 CPU
    model.freeze_perception_stack()

    # 训练前快照：backbone / perception_head（应保持不变）+ planning_head（应变化）
    bb_before = _snapshot(model.backbone)
    ph_before = _snapshot(model.perception_head)
    pl_before = _snapshot(model.planning_head)

    # optimizer 只收集可训练参数（即 planning_head）
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "冻结后应仍有可训练参数（planning_head）"
    opt = torch.optim.AdamW(trainable, lr=CFG.lr)

    ego_hist, agents, agent_mask, traj_gt, traj_mask = _make_batch(B=4)

    # 跑几步，确保 planning_head 确实发生移动
    for _ in range(3):
        traj_pred = model(ego_hist, agents, agent_mask, mode="planning")
        loss = planning_loss(traj_pred, traj_gt, traj_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---------- 断言 1：感知栈所有参数逐元素相等（冻结生效）----------
    bb_after = dict(model.backbone.named_parameters())
    ph_after = dict(model.perception_head.named_parameters())

    for name, before in bb_before.items():
        assert torch.equal(before, bb_after[name].detach()), \
            f"backbone 参数 {name} 在阶段二训练步后发生了变化（冻结失败）"
    for name, before in ph_before.items():
        assert torch.equal(before, ph_after[name].detach()), \
            f"perception_head 参数 {name} 在阶段二训练步后发生了变化（冻结失败）"

    # ---------- 断言 2：planning_head 至少一个参数发生变化 ----------
    pl_after = dict(model.planning_head.named_parameters())
    changed = [name for name, before in pl_before.items()
               if not torch.equal(before, pl_after[name].detach())]
    assert changed, "planning_head 没有任何参数发生变化（梯度链路可能未打通）"
