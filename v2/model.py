"""v2 完整模型：图像 -> BEV -> 规划轨迹（端到端）。

把前面验证过的三块串成一个模型：
  ImageBackbone(6 路图提特征) -> BEVCrossAttention(投影+注意力融合成 BEV 特征图)
  -> PlanningHead(BEV 特征图回归自车未来轨迹)。

采样网格（build_sampling_grid 的结果）由数据侧按相机内外参预先算好传入，避免每次
前向重复算几何（几何只依赖标定，和图像内容无关）。
"""

import torch
import torch.nn as nn
from torch import Tensor

from bev_encoder import BEVCrossAttention
from image_backbone import ImageBackbone


class PlanningHead(nn.Module):
    """从 BEV 特征图回归自车未来轨迹。

    (B,D,S,S) --卷积下采样--> 全局池化 --> MLP --> (B, Tf, 2)
    """

    def __init__(self, d_model: int = 128, fut_steps: int = 12):
        super().__init__()
        self.fut_steps = fut_steps
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, fut_steps * 2))

    def forward(self, bev: Tensor, cmd_emb: Tensor | None = None) -> Tensor:
        """输出轨迹。规划头预测每步**位移增量**，累加成轨迹，保证连续平滑。

        bev: (B,D,S,S)；cmd_emb: 可选 (B,D) 导航指令 embedding，加到池化特征上
        （融入"该往哪个方向走"的意图）。head 输出 Tf 个 (dx,dy) 增量，cumsum 成轨迹。
        """
        x = self.conv(bev)                       # (B,128,S/8,S/8)
        x = x.mean(dim=(2, 3))                    # 全局平均池化 -> (B,128)
        if cmd_emb is not None:
            x = x + cmd_emb                      # 融合导航意图
        deltas = self.head(x).view(-1, self.fut_steps, 2)   # (B,Tf,2) 每步增量
        traj = torch.cumsum(deltas, dim=1)                  # 累加 -> 绝对轨迹
        return traj


class OccupancyHead(nn.Module):
    """占据头：BEV 特征 -> 每格"有没有物体"的占据 logit 图 (B,1,S,S)。

    提供密集的感知监督，逼 BEV 特征学出"哪里有车"的可读结构。几层保持空间分辨率
    的卷积（不下采样），最后 1x1 卷积出单通道 logit。
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_model, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 1, 1),                 # (B,1,S,S) logit
        )

    def forward(self, bev: Tensor) -> Tensor:
        return self.net(bev)                     # (B,1,S,S)


class NanoFSDv2(nn.Module):
    """图像 -> BEV -> 轨迹 + 占据 的端到端模型（多任务）。"""

    def __init__(self, d_model: int = 128, bev_size: int = 100,
                 n_heights: int = 4, n_cams: int = 6, fut_steps: int = 12,
                 pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = ImageBackbone(pretrained=pretrained, freeze=freeze_backbone)
        self.bev = BEVCrossAttention(
            d_model=d_model, feat_channels=self.backbone.out_channels,
            bev_size=bev_size, n_heights=n_heights, n_cams=n_cams)
        self.planner = PlanningHead(d_model=d_model, fut_steps=fut_steps)
        self.occ_head = OccupancyHead(d_model=d_model)      # 占据：有没有物体
        self.map_head = OccupancyHead(d_model=d_model)      # 地图：可行驶区（同结构，不同头）
        self.cmd_emb = nn.Embedding(3, d_model)             # 导航指令 0=左 1=直 2=右

    def forward(self, images: Tensor, sample_uv: Tensor, sample_valid: Tensor,
                command: Tensor | None = None, return_bev: bool = False):
        """
        输入:
          images       (B, V, 3, H, W)
          sample_uv    (B, S*S, V, Z, 2)
          sample_valid (B, S*S, V, Z)
          command      (B,) long，导航指令 0/1/2；None 时规划头不加意图（兼容旧调用）
        输出:
          traj (B,Tf,2), occ_logit (B,1,S,S), map_logit (B,1,S,S)；
          return_bev=True 额外返回 bev
        """
        feats = self.backbone(images)                       # (B,V,Cf,Hf,Wf)
        bev = self.bev(feats, sample_uv, sample_valid)      # (B,D,S,S)
        cmd_emb = self.cmd_emb(command) if command is not None else None
        traj = self.planner(bev, cmd_emb)                   # (B,Tf,2)
        occ = self.occ_head(bev)                            # (B,1,S,S) 占据
        drivable = self.map_head(bev)                       # (B,1,S,S) 可行驶区
        if return_bev:
            return traj, occ, drivable, bev
        return traj, occ, drivable


if __name__ == "__main__":
    # 形状 + 显存自测（随机输入，仅验证前向贯通）
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, V, H, W = 1, 6, 224, 400
    S, Z, Tf = 100, 4, 12

    model = NanoFSDv2(bev_size=S, n_heights=Z, n_cams=V, fut_steps=Tf,
                      pretrained=False).to(dev).eval()
    images = torch.rand(B, V, 3, H, W, device=dev)
    sample_uv = torch.rand(B, S * S, V, Z, 2, device=dev) * 2 - 1
    sample_valid = torch.rand(B, S * S, V, Z, device=dev) > 0.5

    with torch.no_grad():
        traj, occ, drivable, bev = model(images, sample_uv, sample_valid, return_bev=True)
    print(f"traj: {tuple(traj.shape)}  (期望 (1,{Tf},2))")
    print(f"occ : {tuple(occ.shape)}   (期望 (1,1,{S},{S}))")
    print(f"map : {tuple(drivable.shape)}   (期望 (1,1,{S},{S}))")
    print(f"bev : {tuple(bev.shape)}   (期望 (1,128,{S},{S}))")
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: 总 {n_tot:,}  可训练 {n_tr:,}（backbone 冻结则不计）")
    if dev == "cuda":
        torch.cuda.synchronize()
        print(f"峰值显存: {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")
