"""v2 第 3b 步：几何引导的 cross-attention —— 把 6 路图像特征搬到 BEV 俯视图。

承接 3a 验证好的投影几何：BEV 每个格子（query）带若干高度锚点，投影到 6 张图上
得到采样位置，用 grid_sample 从图像特征取出候选特征，再做 cross-attention 加权融合，
输出稠密 BEV 特征图 (B, D, S, S)。

记号: B=batch, V=相机数(6), D=模型维(128), Cf=图像特征通道(512),
      S=BEV 边长(100), Z=高度锚点数, Hf/Wf=图像特征图高宽(7/13)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BEVCrossAttention(nn.Module):
    """几何引导的 BEV cross-attention（BEVFormer 精简版，标准注意力代替 deformable）。

    流程（对应讲解的三个概念）:
      1) BEV query: S*S 个可学习格子向量（本模块只做"融合"，query 由外部传入或内建）。
      2) 几何确定"问谁": 外部传入每个格子 Z 个高度点投影到 V 路图的像素坐标 + 可见 mask。
      3) cross-attention: query 作 Q，命中的图像特征作 K/V，加权求和填回格子。
    """

    def __init__(self, d_model: int = 128, feat_channels: int = 512,
                 n_heads: int = 4, bev_size: int = 100,
                 n_heights: int = 4, n_cams: int = 6):
        super().__init__()
        self.d_model = d_model
        self.bev_size = bev_size
        self.n_heights = n_heights
        self.n_cams = n_cams

        # BEV query: 每个格子一个可学习向量 (S*S, D)
        self.bev_query = nn.Parameter(torch.randn(bev_size * bev_size, d_model) * 0.02)
        # 把图像特征通道 Cf 投到模型维 D，作为 K/V 的来源
        self.feat_proj = nn.Linear(feat_channels, d_model)
        # 注意力的 Q/K/V/O 线性层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img_feats: Tensor, sample_uv: Tensor,
                sample_valid: Tensor) -> Tensor:
        """
        输入:
          img_feats    (B, V, Cf, Hf, Wf)  6 路图像特征（第 2 步 CNN 输出）
          sample_uv    (B, S*S, V, Z, 2)   每格的 Z 个高度点投到 V 路图的归一化像素坐标
                                            （范围 [-1,1]，供 grid_sample 使用）
          sample_valid (B, S*S, V, Z)      bool，该采样点是否在对应相机视野内
        输出:
          bev          (B, D, S, S)         稠密 BEV 特征图
        """
        B, V, Cf, Hf, Wf = img_feats.shape
        Q = self.bev_size * self.bev_size   # 格子数
        Z = self.n_heights

        # --- 用 grid_sample 从每路图像特征里按投影像素取候选特征 ---
        # grid_sample 需要 grid 形状 (B*V, out_h, out_w, 2)；这里把 (Q,Z) 摊成 out_h*out_w
        feats = img_feats.reshape(B * V, Cf, Hf, Wf)
        # sample_uv: (B, Q, V, Z, 2) -> (B, V, Q, Z, 2) -> (B*V, Q, Z, 2)
        grid = sample_uv.permute(0, 2, 1, 3, 4).reshape(B * V, Q, Z, 2)
        sampled = F.grid_sample(feats, grid, mode="bilinear",
                                align_corners=False)       # (B*V, Cf, Q, Z)
        sampled = sampled.reshape(B, V, Cf, Q, Z).permute(0, 3, 1, 4, 2)  # (B,Q,V,Z,Cf)

        # 候选特征投到模型维，作为 K/V 源：每个格子有 V*Z 个候选
        cand = self.feat_proj(sampled).reshape(B, Q, V * Z, self.d_model)  # (B,Q,VZ,D)
        valid = sample_valid.permute(0, 1, 2, 3).reshape(B, Q, V * Z)      # (B,Q,VZ) bool

        # --- cross-attention: query 对候选做加权（逐格子，多头）---
        q = self.q_proj(self.bev_query).unsqueeze(0).expand(B, -1, -1)     # (B,Q,D)
        k = self.k_proj(cand)                                             # (B,Q,VZ,D)
        v = self.v_proj(cand)                                             # (B,Q,VZ,D)

        H, dh = self.n_heads, self.d_model // self.n_heads
        q = q.reshape(B, Q, H, dh)
        k = k.reshape(B, Q, V * Z, H, dh)
        v = v.reshape(B, Q, V * Z, H, dh)

        # 注意力分数: 每个格子 query 和它 VZ 个候选的点积 (B,Q,H,VZ)
        attn = torch.einsum("bqhd,bqnhd->bqhn", q, k) / (dh ** 0.5)
        # 不可见候选置 -inf（mask）；全不可见的格子（无相机覆盖）稍后置零
        mask = valid.unsqueeze(2)                                         # (B,Q,1,VZ)
        attn = attn.masked_fill(~mask, float("-inf"))
        no_cand = (~valid).all(dim=-1)                                    # (B,Q) 该格无任何可见候选
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)                            # 全 -inf 行 softmax 得 nan -> 0

        out = torch.einsum("bqhn,bqnhd->bqhd", attn, v).reshape(B, Q, self.d_model)
        out = self.out_proj(out)
        out = self.norm(out + self.bev_query.unsqueeze(0))                # 残差 + 归一化
        out = out.masked_fill(no_cand.unsqueeze(-1), 0.0)                 # 无覆盖格子置零

        bev = out.transpose(1, 2).reshape(B, self.d_model, self.bev_size, self.bev_size)
        return bev


def build_sampling_grid(bev_xy: Tensor, heights: Tensor,
                        K: Tensor, cam_R: Tensor, cam_t: Tensor,
                        feat_hw: tuple[int, int], img_hw: tuple[int, int]):
    """由相机内外参，算出每个 BEV 格子的 Z 个高度点投到 V 路图的采样坐标与可见性。

    这是 3a 投影几何的"可微张量版"：ego 3D 点 -> cam -> 像素 -> 归一化到 [-1,1]
    （grid_sample 约定）。像素按原图尺寸算，再归一化，所以无论特征图多大都对齐。

    输入:
      bev_xy  (Q, 2)       每个格子的 ego 系 (x, y)（米）
      heights (Z,)         高度锚点 z（米）
      K       (V, 3, 3)    各相机内参
      cam_R   (V, 3, 3)    各相机 cam->ego 旋转
      cam_t   (V, 3)       各相机 cam->ego 平移
      feat_hw (Hf, Wf)     图像特征图尺寸（grid_sample 的目标）
      img_hw  (H, W)       原图尺寸（内参 K 对应的像素尺度）
    输出:
      sample_uv    (Q, V, Z, 2)  归一化像素坐标 [-1,1]
      sample_valid (Q, V, Z)     bool，在视野内
    """
    Q = bev_xy.shape[0]
    Z = heights.shape[0]
    V = K.shape[0]
    H_img, W_img = img_hw

    # 组装每个格子的 Z 个 3D 点 (Q, Z, 3)
    xy = bev_xy[:, None, :].expand(Q, Z, 2)                       # (Q,Z,2)
    zz = heights[None, :, None].expand(Q, Z, 1)                   # (Q,Z,1)
    pts_ego = torch.cat([xy, zz], dim=-1)                        # (Q,Z,3)

    # ego -> cam: p_cam = R^T (p_ego - t)，对每个相机 V
    # 扩展维度做批量: (V,Q,Z,3)
    p = pts_ego[None] - cam_t[:, None, None, :]                  # (V,Q,Z,3)
    p_cam = torch.einsum("vij,vqzj->vqzi", cam_R.transpose(1, 2), p)  # R^T @ p
    z_cam = p_cam[..., 2]                                        # (V,Q,Z)
    in_front = z_cam > 1e-3

    # K @ p_cam，再除以深度
    uvw = torch.einsum("vij,vqzj->vqzi", K, p_cam)              # (V,Q,Z,3)
    eps = 1e-6
    u = uvw[..., 0] / (uvw[..., 2] + eps)
    v = uvw[..., 1] / (uvw[..., 2] + eps)
    in_img = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    valid = in_front & in_img                                   # (V,Q,Z)

    # 归一化到 [-1,1]（grid_sample 约定，按原图尺寸）
    un = (u / (W_img - 1)) * 2 - 1
    vn = (v / (H_img - 1)) * 2 - 1
    uv = torch.stack([un, vn], dim=-1)                          # (V,Q,Z,2)
    # 不可见点坐标置界外（2.0），grid_sample 会取零且不影响（反正被 mask）
    uv = torch.where(valid.unsqueeze(-1), uv, torch.full_like(uv, 2.0))

    # 转成模块需要的 (Q, V, Z, ...)
    sample_uv = uv.permute(1, 0, 2, 3).contiguous()             # (Q,V,Z,2)
    sample_valid = valid.permute(1, 0, 2).contiguous()          # (Q,V,Z)
    return sample_uv, sample_valid


if __name__ == "__main__":
    # 形状 + 显存自测：用随机但合理的相机参数，走通 几何 -> 采样 -> cross-attention。
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    B, V, Cf, Hf, Wf = 1, 6, 512, 7, 13
    S, Z, D = 100, 4, 128
    img_hw = (224, 400)

    # BEV 网格 (Q,2)：±50m, 100x100
    rng = 50.0
    xs = torch.linspace(-rng, rng, S)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    bev_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).to(dev)  # (Q,2)
    heights = torch.tensor([0.0, -0.5, -1.0, 0.5], device=dev)             # (Z,)

    # 随机相机参数（仅测形状/显存；真实参数后续从数据接入）
    K = torch.eye(3, device=dev).unsqueeze(0).repeat(V, 1, 1)
    K[:, 0, 0] = 400; K[:, 1, 1] = 400; K[:, 0, 2] = 200; K[:, 1, 2] = 112
    cam_R = torch.eye(3, device=dev).unsqueeze(0).repeat(V, 1, 1)
    cam_t = torch.randn(V, 3, device=dev) * 0.5

    sample_uv, sample_valid = build_sampling_grid(
        bev_xy, heights, K, cam_R, cam_t, (Hf, Wf), img_hw)
    sample_uv = sample_uv.unsqueeze(0)        # (1,Q,V,Z,2)
    sample_valid = sample_valid.unsqueeze(0)  # (1,Q,V,Z)

    img_feats = torch.randn(B, V, Cf, Hf, Wf, device=dev)

    enc = BEVCrossAttention(d_model=D, feat_channels=Cf, bev_size=S,
                            n_heights=Z, n_cams=V).to(dev)
    enc.eval()
    with torch.no_grad():
        bev = enc(img_feats, sample_uv, sample_valid)

    print(f"sample_uv    : {tuple(sample_uv.shape)}  (期望 (1,{S*S},{V},{Z},2))")
    print(f"sample_valid : {tuple(sample_valid.shape)}  可见采样点占比="
          f"{sample_valid.float().mean().item():.3f}")
    print(f"BEV 输出     : {tuple(bev.shape)}  (期望 (1,{D},{S},{S}))")
    n_param = sum(p.numel() for p in enc.parameters())
    print(f"模块参数量   : {n_param:,}")
    if dev == "cuda":
        torch.cuda.synchronize()
        print(f"峰值显存     : {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")
