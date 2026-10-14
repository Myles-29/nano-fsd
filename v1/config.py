from dataclasses import dataclass


@dataclass
class Config:
    """nanoGPT 式集中管理超参的配置对象。

    记号: N=agent 上限, Th=历史帧数, Tf=未来帧数, C=每 agent 特征维, D=embedding 维。
    """

    # 场景维度
    n_agents: int = 32        # N: padding 后的 agent 数上限
    hist_steps: int = 4       # Th: 历史帧数
    fut_steps: int = 12       # Tf: 预测未来帧数 (nuScenes 2Hz -> 6s)
    # C = 4 + 2 + 3 = 9: [x, y, vx, vy, cos_yaw, sin_yaw, cls_onehot×3(vehicle/human/cyclist)]
    # （原设计写 8，与任务 2.2 实际产出的 3 类 one-hot 不符，按数据层实际维度改为 9）
    agent_feat_dim: int = 9
    # 模型维度
    d_model: int = 128        # D
    n_heads: int = 4
    n_layers: int = 3
    # 训练
    lr: float = 5e-4
    batch_size: int = 64
    device: str = "cuda"


if __name__ == "__main__":
    from dataclasses import fields

    cfg = Config()
    print("Config 默认值:")
    for f in fields(cfg):
        print(f"  {f.name:<15} = {getattr(cfg, f.name)!r}")

    # agent_feat_dim 各维含义
    feat_names = ["x", "y", "vx", "vy", "cos_yaw", "sin_yaw",
                  "cls_vehicle", "cls_human", "cls_cyclist"]
    assert len(feat_names) == cfg.agent_feat_dim
    print(f"\nagent_feat_dim={cfg.agent_feat_dim} 各维含义:")
    for i, name in enumerate(feat_names):
        print(f"  [{i}] {name}")
