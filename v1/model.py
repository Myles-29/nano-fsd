"""model.py — Backbone + 感知头 + 规划头 + 顶层 NanoFSD。

按 design.md 的接口契约逐步实现（任务 3.1~3.5）。
记号: B=batch, N=agent 上限, Th=历史帧数, C=每 agent 特征维, D=embedding 维。
"""

import torch
import torch.nn as nn
from torch import Tensor

from config import Config


class Backbone(nn.Module):
    """场景编码器：自车 token 0 + agent token 1..N，过 Transformer encoder。

    - 每个 token 把自己的历史展平 (Th*4 或 Th*C) 后线性投影到 D
    - 加一个可学习的类型 embedding（ego / agent），不加依赖 agent 序号的位置编码，
      保证 token 与输入 agent 一一对应、且对 padding 内容不敏感
    - padding agent 通过 key_padding_mask 屏蔽（True = 忽略）
    """

    def __init__(self, cfg: Config, dropout: float = 0.0):
        super().__init__()
        D = cfg.d_model
        self.ego_proj = nn.Sequential(
            nn.Linear(cfg.hist_steps * 4, D), nn.GELU(), nn.Linear(D, D))
        self.agent_proj = nn.Sequential(
            nn.Linear(cfg.hist_steps * cfg.agent_feat_dim, D), nn.GELU(), nn.Linear(D, D))
        self.type_emb = nn.Embedding(2, D)  # 0 = ego, 1 = agent
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.n_heads, dim_feedforward=4 * D,
            dropout=dropout, batch_first=True, norm_first=True)
        # enable_nested_tensor=False: 关闭 fast-path 的 nested tensor 转换，保证输出恒为 N+1 个 token
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)

    def forward(self, ego_hist: Tensor, agents: Tensor, agent_mask: Tensor) -> Tensor:
        """
        输入:
          ego_hist   (B, Th, 4)
          agents     (B, N, Th, C)
          agent_mask (B, N) bool, True=有效
        输出:
          scene_emb  (B, N+1, D)   # token 0 = ego, 1..N = agents（顺序保持）
        """
        B, N = agents.shape[:2]
        ego_tok = self.ego_proj(ego_hist.reshape(B, 1, -1)) + self.type_emb.weight[0]    # (B,1,D)
        agent_tok = self.agent_proj(agents.reshape(B, N, -1)) + self.type_emb.weight[1]  # (B,N,D)
        tokens = torch.cat([ego_tok, agent_tok], dim=1)                                 # (B,N+1,D)

        # key_padding_mask: True=忽略；ego token 永远有效
        ego_valid = torch.zeros(B, 1, dtype=torch.bool, device=agent_mask.device)
        pad_mask = torch.cat([ego_valid, ~agent_mask.bool()], dim=1)                    # (B,N+1)

        scene_emb = self.norm(self.encoder(tokens, src_key_padding_mask=pad_mask))
        assert scene_emb.shape == (B, N + 1, tokens.shape[-1])  # 不变式: token 数恒为 N+1
        return scene_emb


class PerceptionHead(nn.Module):
    """感知头（方案 A：相对位置回归）。

    取场景 embedding 的 agent token（1..N）过 MLP，回归每个 agent 的相对位置 (x, y)。
    token 顺序与输入 agent 一一对应，故输出第 i 行对应输入第 i 个 agent。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))

    def forward(self, scene_emb: Tensor) -> Tensor:
        """
        输入 scene_emb (B, N+1, D)
        输出 agent_pred (B, N, 2)   # 取 token 1..N 过 MLP -> (x, y)
        """
        agent_tokens = scene_emb[:, 1:]           # (B, N, D) 丢弃 ego token
        agent_pred = self.mlp(agent_tokens)       # (B, N, 2)
        return agent_pred


class PlanningHead(nn.Module):
    """规划头（最简版：直接回归自车未来轨迹）。

    取场景 embedding 的 ego token（token 0）过 MLP，一次性回归自车未来
    Tf 个时间步的相对位置 (x, y)，再 reshape 成 (B, Tf, 2)。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.fut_steps = cfg.fut_steps
        self.mlp = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, cfg.fut_steps * 2))

    def forward(self, scene_emb: Tensor) -> Tensor:
        """
        输入 scene_emb (B, N+1, D)
        输出 traj_pred (B, Tf, 2)   # 取 ego token(0) 过 MLP -> 展平回归 Tf*2 再 reshape
        """
        B = scene_emb.shape[0]
        ego_token = scene_emb[:, 0]               # (B, D) 取 token 0 = ego
        traj_pred = self.mlp(ego_token)           # (B, Tf*2)
        traj_pred = traj_pred.view(B, self.fut_steps, 2)  # (B, Tf, 2)
        return traj_pred


class NanoFSD(nn.Module):
    """顶层模型：整合 backbone + 感知头 + 规划头，按 mode 分派到对应头。

    - backbone 把场景编码成共享 embedding (B, N+1, D)
    - mode="perception" 走感知头，返回 (B, N, 2)
    - mode="planning"   走规划头，返回 (B, Tf, 2)
    - freeze_perception_stack 冻结 backbone + 感知头（阶段二只训规划头）
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = Backbone(cfg)
        self.perception_head = PerceptionHead(cfg)
        self.planning_head = PlanningHead(cfg)

    def forward(self, ego_hist: Tensor, agents: Tensor,
                agent_mask: Tensor, mode: str) -> Tensor:
        """
        输入:
          ego_hist   (B, Th, 4)
          agents     (B, N, Th, C)
          agent_mask (B, N) bool, True=有效
          mode       str, ∈ {"perception", "planning"}
        输出:
          mode="perception" -> agent_pred (B, N, 2)
          mode="planning"   -> traj_pred  (B, Tf, 2)
        """
        scene_emb = self.backbone(ego_hist, agents, agent_mask)  # (B, N+1, D)
        if mode == "perception":
            return self.perception_head(scene_emb)               # (B, N, 2)
        if mode == "planning":
            return self.planning_head(scene_emb)                 # (B, Tf, 2)
        raise ValueError(f"未知 mode: {mode!r}，应为 'perception' 或 'planning'")

    def freeze_perception_stack(self) -> None:
        """冻结感知栈（核心学习点：分阶段冻结训练的阶段二入口）。

        前置条件: 已从 checkpoint_stage1.pt 加载好 backbone + 感知头权重。
        后置条件:
          - backbone 与 perception_head 下所有参数 requires_grad == False（不再更新）
          - planning_head 下所有参数 requires_grad == True（继续训练）
          - backbone / perception_head 置为 eval() 模式，关闭 dropout、固定 BN 统计，
            避免冻结后仍受训练态影响
        """
        for module in (self.backbone, self.perception_head):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

        for param in self.planning_head.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # 演示入口：用随机输入跑一次前向，打印形状 / 参数量 / 冻结前后可训练参数个数。
    # 为保证到处可跑，演示固定在 CPU 上（不读取 cfg.device），batch 小且含部分 padding agent。
    torch.manual_seed(0)
    cfg = Config()
    B = 2
    N, Th, C, Tf, D = cfg.n_agents, cfg.hist_steps, cfg.agent_feat_dim, cfg.fut_steps, cfg.d_model

    model = NanoFSD(cfg)
    model.eval()  # 演示用 eval，关闭 dropout 等训练态

    # 构造随机输入
    ego_hist = torch.randn(B, Th, 4)
    agents = torch.randn(B, N, Th, C)
    # agent_mask: 含部分 padding —— 每个样本只保留前若干个 agent 为有效
    agent_mask = torch.zeros(B, N, dtype=torch.bool)
    agent_mask[0, :5] = True   # 样本 0: 5 个有效 agent，其余 padding
    agent_mask[1, :12] = True  # 样本 1: 12 个有效 agent，其余 padding
    # padding 行内容置零（与数据层约定一致，顺带演示 padding 不变性的前提）
    agents = agents * agent_mask[:, :, None, None].float()

    print("=" * 60)
    print(f"演示配置: B={B}, N={N}, Th={Th}, C={C}, Tf={Tf}, D={D}（CPU）")
    print(f"有效 agent 数: 样本0={int(agent_mask[0].sum())}, 样本1={int(agent_mask[1].sum())}")

    # 1) 前向：打印 scene_emb / agent_pred / traj_pred 形状
    print("\n--- 前向输出形状 ---")
    with torch.no_grad():
        scene_emb = model.backbone(ego_hist, agents, agent_mask)             # (B, N+1, D)
        agent_pred = model(ego_hist, agents, agent_mask, mode="perception")  # (B, N, 2)
        traj_pred = model(ego_hist, agents, agent_mask, mode="planning")     # (B, Tf, 2)
    print(f"  scene_emb  (backbone)       : {tuple(scene_emb.shape)}   期望 ({B}, {N + 1}, {D})")
    print(f"  agent_pred (perception_head): {tuple(agent_pred.shape)}        期望 ({B}, {N}, 2)")
    print(f"  traj_pred  (planning_head)  : {tuple(traj_pred.shape)}         期望 ({B}, {Tf}, 2)")

    # 2) 各模块参数量（numel 合计）
    def numel(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    n_bb = numel(model.backbone)
    n_ph = numel(model.perception_head)
    n_pl = numel(model.planning_head)
    n_total = numel(model)
    print("\n--- 参数量（numel 合计）---")
    print(f"  backbone        : {n_bb:>10,}")
    print(f"  perception_head : {n_ph:>10,}")
    print(f"  planning_head   : {n_pl:>10,}")
    print(f"  合计            : {n_total:>10,}")
    assert n_total == n_bb + n_ph + n_pl  # 三模块之和应等于总参数量

    # 3) 冻结演示：打印 freeze_perception_stack 前后各模块 requires_grad=True 的参数张量个数
    def n_trainable(module: nn.Module) -> int:
        return sum(1 for p in module.parameters() if p.requires_grad)

    print("\n--- 冻结前后可训练参数张量个数（requires_grad=True）---")
    before = (n_trainable(model.backbone), n_trainable(model.perception_head),
              n_trainable(model.planning_head))
    print(f"  冻结前: backbone={before[0]}, perception_head={before[1]}, planning_head={before[2]}")

    model.freeze_perception_stack()

    after = (n_trainable(model.backbone), n_trainable(model.perception_head),
             n_trainable(model.planning_head))
    print(f"  冻结后: backbone={after[0]}, perception_head={after[1]}, planning_head={after[2]}")
    # 冻结后 backbone / 感知头应全部关闭，规划头应全部保持可训练
    assert after[0] == 0 and after[1] == 0
    assert after[2] == before[2]
    print("\n冻结正确性自检通过：backbone/perception_head 已全部冻结，planning_head 保持可训练。")
    print("=" * 60)
