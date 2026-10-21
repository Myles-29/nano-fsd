"""v2 完整模型：图像 -> BEV -> 规划轨迹（端到端）。

把前面验证过的三块串成一个模型：
  ImageBackbone(6 路图提特征) -> BEVCrossAttention(投影+注意力融合成 BEV 特征图)
  -> PlanningHead(BEV 特征图回归自车未来轨迹)。

采样网格（build_sampling_grid 的结果）由数据侧按相机内外参预先算好传入，避免每次
前向重复算几何（几何只依赖标定，和图像内容无关）。
"""

import numpy as np
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


class DetectionHead(nn.Module):
    """anchor-based query 检测头：N 个 query 各钉一个固定 BEV 锚点，attend BEV 出框/类别/置信度。

    为什么用锚点（破「匹配-回归死锁」）：原始 DETR 式检测里，匹配基于**预测框**与 GT 的距离，
    训练初期预测框是乱的 → 匹配乱 → 每个 GT 配到的 query 一直换 → 回归目标跳 → 框学不准。
    改法：给每个 query 钉一个**固定不动的锚点**（在 BEV ±range 内均匀铺 N 个点），让：
      1) 匹配基于「GT 离哪个锚点近」（锚点不动，匹配从第一轮就稳定，query 学出空间分工）；
      2) 框中心回归成「锚点 + 小偏移」（范围从 ±range 缩到一个格附近，收敛快）。
    锚点坐标以 buffer `anchor_xy` (N,2) 暴露，供 train.py 的匹配使用。

    输出（dict）：
      box   (B,N,5)：[x, y, yaw, w, l]，其中 x=anchor_x+dx、y=anchor_y+dy（标准化坐标系）
      cls   (B,N,n_cls)：类别 logit（2 类：0=车 1=人）
      score (B,N,1)：存在置信度 logit
      query_emb (B,N,D)：供下游运动/规划头复用
      anchor_xy (N,2)：固定锚点坐标（标准化，= 米/SCENE_SCALE），匹配用

    坐标约定：与数据侧一致——x,y 为 SCENE_SCALE 归一化后的 ego 系坐标（范围约 ±1）。
    显存控制：BEV token 作 K/V 支持 kv_stride 平均池化下采样。
    """

    def __init__(self, d_model: int = 128, n_query: int = 50, n_layers: int = 2,
                 n_cls: int = 2, bev_size: int = 100, kv_stride: int = 2,
                 n_heads: int = 4, bev_range: float = 50.0, scene_scale: float = 50.0):
        super().__init__()
        self.d_model = d_model
        self.n_query = n_query
        self.n_cls = n_cls
        self.bev_size = bev_size
        self.kv_stride = kv_stride

        # 固定锚点网格：在 BEV ±bev_range 内均匀铺 N 个点（标准化坐标 = 米/scene_scale）。
        # 取最接近 N 的近似方阵（g×g>=N），再截前 N 个，保证锚点覆盖整个 BEV 平面。
        g = int(np.ceil(np.sqrt(n_query)))
        lin = np.linspace(-bev_range, bev_range, g) / scene_scale   # 标准化坐标
        ax, ay = np.meshgrid(lin, lin, indexing="ij")
        anchors = np.stack([ax.reshape(-1), ay.reshape(-1)], axis=-1)[:n_query]  # (N,2)
        self.register_buffer("anchor_xy", torch.tensor(anchors, dtype=torch.float32))

        # 每个 query 的「内容」部分（可学习）；位置信息由锚点经 MLP 编码后注入（见 forward）。
        self.query_content = nn.Parameter(torch.randn(n_query, d_model) * 0.1)
        # 锚点位置编码 MLP：把 (x,y) 锚点坐标映射成 D 维位置 query，加到内容 query 上。
        self.anchor_pos_mlp = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # BEV token 下采样（平均池化），控显存；kv_stride=1 时为恒等
        self.kv_pool = (nn.AvgPool2d(kv_stride) if kv_stride > 1 else nn.Identity())

        # K/V 的位置编码：下采样后的 BEV 网格每个 token 一个可学习位置向量
        kv_size = bev_size // kv_stride if kv_stride > 1 else bev_size
        self.n_kv = kv_size * kv_size
        self.bev_pos = nn.Parameter(torch.randn(self.n_kv, d_model) * 0.02)

        # L_dec 层 decoder：每层 = cross-attn(query->BEV) + FFN，各带残差 + LN
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                          nn.Linear(d_model * 4, d_model))
            for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # 输出头：off_head 回归「相对锚点的中心偏移 (dx,dy)」+「yaw,w,l」共 5 维。
        # dx,dy 用 tanh 限幅到约 ±2 格范围（标准化坐标），避免偏移发散。
        self.off_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 5))
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_cls))
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        # 偏移限幅半径（标准化坐标）：约 2 个锚点间距，够覆盖锚点辖区又不发散。
        if n_query > 1:
            self.off_radius = float(2.0 * (lin[1] - lin[0]))
        else:
            self.off_radius = 1.0

    def forward(self, bev: Tensor) -> dict:
        """bev (B,D,S,S) -> {'box','cls','score','query_emb','anchor_xy'}。

        流程：
          1) BEV 下采样展平成 token (B,n_kv,D) 加位置编码作 K/V；
          2) query = 可学习内容 + 锚点位置编码（把「我负责哪块」注入 query）；逐层
             cross-attn(query->BEV)+FFN 更新；
          3) off_head 出 (dx,dy,yaw,w,l)：中心 = 锚点 + tanh(dx,dy)*off_radius（相对锚点
             小偏移），拼上 yaw/w/l 得 box；cls/score 照常。
        """
        B = bev.shape[0]

        # 1) BEV -> K/V token
        kv = self.kv_pool(bev)                               # (B,D,s,s)
        kv = kv.flatten(2).transpose(1, 2)                   # (B,n_kv,D)
        kv = kv + self.bev_pos.unsqueeze(0)

        # 2) query = 内容 + 锚点位置编码
        pos = self.anchor_pos_mlp(self.anchor_xy)            # (N,D) 锚点位置 query
        q = (self.query_content + pos).unsqueeze(0).expand(B, -1, -1)  # (B,N,D)

        for attn, n1, ffn, n2 in zip(self.cross_attn, self.norm1, self.ffn, self.norm2):
            attn_out, _ = attn(q, kv, kv)
            q = n1(q + attn_out)
            q = n2(q + ffn(q))

        # 3) 相对锚点的框回归
        off = self.off_head(q)                               # (B,N,5)
        dxy = torch.tanh(off[..., 0:2]) * self.off_radius    # 限幅小偏移
        center = self.anchor_xy.unsqueeze(0) + dxy           # (B,N,2) = 锚点 + 偏移
        box = torch.cat([center, off[..., 2:5]], dim=-1)     # (B,N,5) [x,y,yaw,w,l]

        return {
            "box": box,
            "cls": self.cls_head(q),
            "score": self.score_head(q),
            "query_emb": q,
            "anchor_xy": self.anchor_xy,                     # (N,2) 固定锚点，匹配用
        }


class VectorMapHead(nn.Module):
    """query-based 矢量地图头：M 个 map query 各钉一个固定 BEV 锚点，attend BEV 出一条折线/类别/置信度。

    孪生关系（与 DetectionHead 刻意对称）：本头在结构上几乎是检测头的「孪生兄弟」，
    最大化复用「query 作为信息载体 + 锚点稳定匹配 + 带限幅偏移的回归」范式。
    唯一的关键差异在**输出几何**：
      - DetectionHead：每个 query 出一个 5 维框 [x,y,yaw,w,l]（单点中心 + 尺寸朝向）；
      - VectorMapHead：每个 query 出一条**定长折线** (P,2)（P 个有序点，表示一条车道
        分隔线 / 道路边界 / 人行横道边界）。
    即把「框中心 = 锚点 + tanh 限幅偏移」推广成「折线 P 个点 = 锚点基准 + 逐点 tanh 限幅偏移」：
    每条折线的 P 个点都以同一个 BEV 锚点为基准，初始是锚点附近的一小簇点，训练中各点学会
    沿不同方向展开成一条线。这保留了锚点带来的「匹配从第一轮就稳定、回归范围受限收敛快」
    的好处（破解检测头里讲过的「匹配-回归死锁」），只是把标量框换成点序列。

    为什么用锚点（同 DetectionHead）：匹配基于「GT 离哪个锚点近」（锚点不动，空间分工稳定），
    折线点回归成「锚点 + 小偏移」（范围受限、收敛快）。锚点坐标以 buffer `anchor_xy` (M,2)
    暴露，供 train.py 的折线匹配使用。

    输出（dict）：
      pts   (B,M,P,2)：每条折线 P 个点 [x,y]，= anchor_xy 广播 + tanh(偏移)*off_radius（标准化坐标）
      cls   (B,M,n_cls)：类别 logit（3 类：0=车道分隔线 1=道路边界 2=人行横道）
      score (B,M,1)：存在置信度 logit
      query_emb (B,M,D)：最后一层 query embedding，供下游复用
      anchor_xy (M,2)：固定锚点坐标（标准化，= 米/SCENE_SCALE），匹配用

    坐标约定：与数据侧及检测头一致——x,y 为 SCENE_SCALE 归一化后的 ego 系坐标（x 前 y 左，范围约 ±1）。
    显存控制：BEV token 作 K/V 支持 kv_stride 平均池化下采样。
    """

    def __init__(self, d_model: int = 128, n_query: int = 48, n_points: int = 20,
                 n_layers: int = 2, n_cls: int = 3, bev_size: int = 100,
                 kv_stride: int = 2, n_heads: int = 4,
                 bev_range: float = 50.0, scene_scale: float = 50.0):
        super().__init__()
        self.d_model = d_model
        self.n_query = n_query          # M：map query 数（= MAX_MAP）
        self.n_points = n_points        # P：每条折线固定点数
        self.n_cls = n_cls
        self.bev_size = bev_size
        self.kv_stride = kv_stride

        # 固定锚点网格：在 BEV ±bev_range 内均匀铺 M 个点（标准化坐标 = 米/scene_scale）。
        # 与 DetectionHead 完全同款构造，保证两头锚点语义一致、匹配基建可复用。
        g = int(np.ceil(np.sqrt(n_query)))
        lin = np.linspace(-bev_range, bev_range, g) / scene_scale   # 标准化坐标
        ax, ay = np.meshgrid(lin, lin, indexing="ij")
        anchors = np.stack([ax.reshape(-1), ay.reshape(-1)], axis=-1)[:n_query]  # (M,2)
        self.register_buffer("anchor_xy", torch.tensor(anchors, dtype=torch.float32))

        # 每个 query 的「内容」部分（可学习）；位置信息由锚点经 MLP 编码后注入（见 forward）。
        self.query_content = nn.Parameter(torch.randn(n_query, d_model) * 0.1)
        # 锚点位置编码 MLP：把 (x,y) 锚点坐标映射成 D 维位置 query，加到内容 query 上。
        self.anchor_pos_mlp = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # BEV token 下采样（平均池化），控显存；kv_stride=1 时为恒等
        self.kv_pool = (nn.AvgPool2d(kv_stride) if kv_stride > 1 else nn.Identity())

        # K/V 的位置编码：下采样后的 BEV 网格每个 token 一个可学习位置向量
        kv_size = bev_size // kv_stride if kv_stride > 1 else bev_size
        self.n_kv = kv_size * kv_size
        self.bev_pos = nn.Parameter(torch.randn(self.n_kv, d_model) * 0.02)

        # L_dec 层 decoder：每层 = cross-attn(query->BEV) + FFN，各带残差 + LN（同 DetectionHead）
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                          nn.Linear(d_model * 4, d_model))
            for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # 输出头：pts_head 回归「每条折线 P 个点相对锚点的偏移」共 P*2 维。
        # 对应检测头的 off_head(5 维框)：这里换成 P*2 维折线点偏移，reshape 成 (P,2)。
        # 每个偏移用 tanh 限幅到约 ±off_radius（标准化坐标），避免折线点发散。
        self.pts_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_points * 2))
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_cls))
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        # 偏移限幅半径（标准化坐标）：约 2 个锚点间距，够让折线从锚点簇展开成跨越辖区的线段，
        # 又不至于发散。n_query<=1（无网格）时退化为固定 1.0。
        if n_query > 1 and g > 1:
            self.off_radius = float(2.0 * (lin[1] - lin[0]))
        else:
            self.off_radius = 1.0

    def forward(self, bev: Tensor) -> dict:
        """bev (B,D,S,S) -> {'pts','cls','score','query_emb','anchor_xy'}。

        流程（镜像 DetectionHead，仅输出几何由框换成折线）：
          1) BEV 下采样展平成 token (B,n_kv,D) 加位置编码作 K/V；
          2) query = 可学习内容 + 锚点位置编码（把「我负责哪块」注入 query）；逐层
             cross-attn(query->BEV)+FFN 更新，循环不变式：每层后 q 恒为 (B,M,D)；
          3) pts_head 出 (B,M,P*2) -> reshape (B,M,P,2) 偏移；折线点 = 锚点广播 +
             tanh(偏移)*off_radius（每点相对同一锚点的小偏移）；cls/score 照常。
        后置：所有输出 batch 维在前、M 维一致且顺序对齐；pts 对 bev 可微、无 NaN/Inf。
        """
        B = bev.shape[0]

        # 1) BEV -> K/V token
        kv = self.kv_pool(bev)                               # (B,D,s,s)
        kv = kv.flatten(2).transpose(1, 2)                   # (B,n_kv,D)
        kv = kv + self.bev_pos.unsqueeze(0)

        # 2) query = 内容 + 锚点位置编码
        pos = self.anchor_pos_mlp(self.anchor_xy)            # (M,D) 锚点位置 query
        q = (self.query_content + pos).unsqueeze(0).expand(B, -1, -1)  # (B,M,D)

        for attn, n1, ffn, n2 in zip(self.cross_attn, self.norm1, self.ffn, self.norm2):
            attn_out, _ = attn(q, kv, kv)
            q = n1(q + attn_out)                             # 循环不变式：q 恒为 (B,M,D)
            q = n2(q + ffn(q))

        # 3) 折线点回归：每条折线 P 个点 = 同一锚点基准 + 逐点 tanh 限幅偏移
        off = self.pts_head(q)                               # (B,M,P*2)
        off = off.view(B, self.n_query, self.n_points, 2)    # (B,M,P,2)
        dxy = torch.tanh(off) * self.off_radius              # 逐点限幅小偏移
        # anchor_xy (M,2) -> (1,M,1,2) 广播到每条折线的每个点
        anchor = self.anchor_xy.view(1, self.n_query, 1, 2)
        pts = anchor + dxy                                   # (B,M,P,2) 标准化 ego 系折线

        return {
            "pts": pts,                                      # (B,M,P,2)
            "cls": self.cls_head(q),                         # (B,M,n_cls)
            "score": self.score_head(q),                     # (B,M,1)
            "query_emb": q,                                  # (B,M,D)
            "anchor_xy": self.anchor_xy,                     # (M,2) 固定锚点，匹配用
        }


class MotionHead(nn.Module):
    """运动预测头：从每个 agent query embedding 回归该 agent 的未来相对轨迹（单模态）。

    对齐 UniAD 的 MotionFormer（本版简化：直接从检测头输出的 query embedding 回归，不做
    多模态、不做 agent-agent 的二次 attention）。这是 UniAD「query 贯穿检测→运动→规划」
    主线的第二棒：检测头告诉 query「我是谁、在哪」，运动头让同一个 query 产出「我要去哪」。

    结构：一个小 MLP 把每个 query 的 D 维 embedding 映射到 Tf*2，再 reshape 成 (Tf,2)。
    MLP 对每个 query 独立作用（只在最后一维上做线性变换），因此 query 顺序天然保持——
    第 i 个 query 的运动预测与其在检测头的框一一对应（正确性属性 2：query 顺序对齐）。

    输出语义：预测的是**相对各 agent 当前中心**的未来轨迹（与 GT 的 agent_fut 对齐），
    不是绝对坐标，也不是增量——与数据侧 agent_fut（相对当前中心的位移）直接可比。
    """

    def __init__(self, d_model: int = 128, fut_steps: int = 12):
        super().__init__()
        self.fut_steps = fut_steps
        # 逐 query MLP：D -> hidden -> Tf*2；仅作用于最后一维，保持 (B,N,·) 的前两维不变。
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, fut_steps * 2))

    def forward(self, query_emb: Tensor) -> Tensor:
        """query_emb (B,N,D) -> agent_fut_pred (B,N,Tf,2)（相对各 agent 当前中心）。

        MLP 在最后一维上作用得到 (B,N,Tf*2)，reshape 成 (B,N,Tf,2)。因对每个 query
        独立计算，输出第 i 个 query 的轨迹与检测头第 i 个 query 的框顺序对齐。
        """
        B, N = query_emb.shape[:2]
        fut = self.net(query_emb)                            # (B,N,Tf*2)
        return fut.view(B, N, self.fut_steps, 2)             # (B,N,Tf,2)


class PlanningHeadV2(nn.Module):
    """交互式规划头：ego query 先 attend BEV（道路/占据上下文），再 attend agent query
    （周围车的位置与意图），融合导航 command，回归自车未来增量轨迹（cumsum 成绝对轨迹）。

    对齐 UniAD 规划阶段的 agent/map 交互（本版简化：标准 cross-attention，不含碰撞代价
    优化层）。这是 UniAD「query 贯穿检测→运动→规划」主线的终点：检测头产出的 agent
    query embedding 作为 K/V 被规划头消费，让自车轨迹的生成能「看见」具体的某辆车。

    与旧 PlanningHead（BEV 全局平均池化）的区别：旧头把 BEV 压成一个全局向量，丢失了
    「具体哪辆车在哪」的空间信息；本头用 1 个可学习 ego query 分两步 cross-attention：
      - 层1 ego→BEV token（kv_stride 下采样后作 K/V，加位置编码）：读道路/占据上下文；
      - 层2 ego→agent query_emb（检测头输出）：读周围 agent 的个体信息，用 agent_score
        作 soft mask 让低置信度（背景）agent 贡献很小。
    每层标准「cross-attn + 残差 + LN + FFN + 残差 + LN」，结构与 DetectionHead 的 decoder
    层一致。最后把 ego embedding（可加 command embedding）过 MLP 得 Tf 个 (dx,dy) 增量，
    cumsum 成绝对轨迹。

    ------------------------------------------------------------------------------------
    soft mask 机制（层2 ego→agent 的核心设计）
    ------------------------------------------------------------------------------------
    agent_score (B,N,1) 是检测头输出的「存在置信度 logit」。我们用 sigmoid(score)∈(0,1)
    作为每个 agent 的「可信权重」w_i，并把它转成**加性注意力偏置**加到 ego→agent 的
    注意力 logits 上：

        bias_i = log(w_i + eps)                                （每个 key 一个标量偏置）
        attn_logits_i = q·k_i / sqrt(d) + bias_i

    直觉：softmax 前对第 i 个 agent 的 logit 加 log(w_i)，等价于把该 agent 的注意力权重
    乘上 w_i（exp(log w_i)=w_i）再归一化——高置信度 agent（w≈1，bias≈0）正常参与，
    低置信度/背景 agent（w≈0，bias→-∞）注意力权重被压到接近 0，几乎不贡献信息。
    这就实现了「soft mask」：不是硬性屏蔽，而是按置信度平滑加权。

    为什么用加性偏置而不是硬 mask / 直接乘 K/V：
      - nn.MultiheadAttention 的 attn_mask 接受加性浮点偏置（加到 logits 上），天然契合；
      - 直接把所有 key 乘到 0（硬屏蔽全部 agent）会让该行 softmax 变成 0/0 → NaN；
        加性偏置即便所有 w_i→0 也只是让各 logit 同量下移，softmax 仍是良定义的概率分布
        （退化为对 agent 的近似均匀注意力，但其输出在下面被整体忽略，见下）。

    ------------------------------------------------------------------------------------
    「agent 全背景」优雅退化（需求 4.3，避免 NaN）
    ------------------------------------------------------------------------------------
    关键设计：**BEV 注意力结果是规划的 base，agent 注意力是可趋于 0 的加性 refinement。**
      ego_after_bev = ego + bev_attn(...)            # base：始终可用（不依赖 agent）
      agent_refine  = agent_attn(...) * gate         # refinement：可被 gate 压到 ~0
      ego_final     = ego_after_bev + agent_refine
    其中 gate = max_i sigmoid(score_i)（batch 内每个样本一个标量，形状 (B,1,1)）：
    只要有任意一个高置信度 agent，gate≈1，refinement 正常注入；当所有 agent 都是背景
    （score 全很负 → 所有 sigmoid≈0 → gate≈0），refinement 被乘到接近 0，规划自然退化为
    「仅 BEV 规划」。即便此时层2 的 softmax 因加性偏置仍可计算（不 NaN），其输出也被
    gate 清零，不污染结果。这样「全背景」既不报错也不 NaN，且确实等价于只用 BEV。

    注意：N=0（没有 agent 维）时直接跳过层2，ego_final = ego_after_bev，同样安全。
    """

    def __init__(self, d_model: int = 128, fut_steps: int = 12,
                 bev_size: int = 100, kv_stride: int = 2, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.fut_steps = fut_steps
        self.bev_size = bev_size
        self.kv_stride = kv_stride

        # 可学习 ego query（1,D）：代表「自车」，去 attend 环境与周围 agent。
        self.ego_query = nn.Parameter(torch.randn(1, d_model))

        # ---- 层1：ego -> BEV token（下采样作 K/V，加位置编码），与 DetectionHead 一致 ----
        self.kv_pool = (nn.AvgPool2d(kv_stride) if kv_stride > 1 else nn.Identity())
        kv_size = bev_size // kv_stride if kv_stride > 1 else bev_size
        self.n_kv = kv_size * kv_size
        self.bev_pos = nn.Parameter(torch.randn(self.n_kv, d_model) * 0.02)

        self.bev_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.bev_norm1 = nn.LayerNorm(d_model)
        self.bev_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.bev_norm2 = nn.LayerNorm(d_model)

        # ---- 层2：ego -> agent query_emb（用 score 作 soft mask 的加性偏置）----
        self.agent_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.agent_norm1 = nn.LayerNorm(d_model)
        self.agent_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.agent_norm2 = nn.LayerNorm(d_model)

        # ---- 轨迹回归 MLP：ego embedding -> Tf*2 增量 ----
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, fut_steps * 2))

    def forward(self, bev: Tensor, agent_emb: Tensor, agent_score: Tensor,
                cmd_emb: Tensor | None = None) -> Tensor:
        """-> traj (B,Tf,2)。

        输入:
          bev          (B,D,S,S)：BEV 特征图（道路/占据上下文）。
          agent_emb    (B,N,D)：检测头输出的 agent query embedding（K/V）。
          agent_score  (B,N,1)：检测头输出的存在置信度 **logit**（soft mask 用，内部取 sigmoid）。
          cmd_emb      (B,D) 或 None：导航指令 embedding，加到 ego embedding 上（None 兼容不加）。

        流程:
          1) ego query 扩成 (B,1,D)；
          2) 层1 ego→BEV：BEV 下采样展平 + 位置编码作 K/V，cross-attn + 残差/LN + FFN；
          3) 层2 ego→agent：sigmoid(score) 转加性偏置作 soft mask，cross-attn + 残差/LN
             + FFN；用 gate=max sigmoid(score) 把该 refinement 整体缩放（全背景→~0 退化）；
          4) 融合 command，MLP 出 Tf 个 (dx,dy) 增量，cumsum 成绝对轨迹 (B,Tf,2)。
        """
        B = bev.shape[0]
        eps = 1e-6

        # 1) ego query 扩到 batch：(1,D) -> (B,1,D)
        ego = self.ego_query.unsqueeze(0).expand(B, -1, -1)      # (B,1,D)

        # 2) 层1 ego -> BEV token（base 上下文，不依赖 agent，始终可用）
        kv = self.kv_pool(bev)                                   # (B,D,s,s)
        kv = kv.flatten(2).transpose(1, 2)                       # (B, n_kv, D)
        kv = kv + self.bev_pos.unsqueeze(0)                      # 加位置编码
        bev_out, _ = self.bev_attn(ego, kv, kv)                  # (B,1,D)
        ego = self.bev_norm1(ego + bev_out)                      # 残差 + LN
        ego = self.bev_norm2(ego + self.bev_ffn(ego))            # FFN 残差 + LN
        # 此时 ego 已含 BEV 上下文，作为规划的 base

        # 3) 层2 ego -> agent query_emb（soft mask + gate 退化）
        N = agent_emb.shape[1]
        if N > 0:
            # sigmoid(score) 作可信权重 w_i∈(0,1)：(B,N,1) -> (B,N)
            w = torch.sigmoid(agent_score).squeeze(-1)           # (B,N)
            # soft mask：加性注意力偏置 bias_i = log(w_i+eps)，形状对齐 MHA 的 attn_mask。
            # attn_mask 支持 (B*n_heads, L_q, L_k)；此处 L_q=1，L_k=N。
            n_heads = self.agent_attn.num_heads
            bias = torch.log(w + eps)                            # (B,N)
            attn_mask = bias[:, None, None, :].expand(B, n_heads, 1, N)
            attn_mask = attn_mask.reshape(B * n_heads, 1, N)     # (B*nh,1,N) 加性偏置

            agent_out, _ = self.agent_attn(ego, agent_emb, agent_emb,
                                           attn_mask=attn_mask)  # (B,1,D)
            agent_ref = self.agent_norm1(ego + agent_out)        # 残差 + LN
            agent_ref = self.agent_norm2(agent_ref + self.agent_ffn(agent_ref))

            # gate=max_i w_i（每样本一标量）：全背景(w全≈0)→gate≈0，refinement 被清零，
            # 规划退化为仅 BEV（agent_ref 与 ego 之差被 gate 缩放后趋于 0，不污染、不 NaN）。
            gate = w.max(dim=1, keepdim=True).values.unsqueeze(-1)   # (B,1,1)
            ego = ego + gate * (agent_ref - ego)                 # refinement，可趋于 0
        # N==0：跳过层2，ego 保持仅 BEV 结果（安全退化）

        # 4) 融合 command，回归增量轨迹
        feat = ego.squeeze(1)                                    # (B,D)
        if cmd_emb is not None:
            feat = feat + cmd_emb                                # 融合导航意图
        deltas = self.head(feat).view(B, self.fut_steps, 2)      # (B,Tf,2) 每步增量
        traj = torch.cumsum(deltas, dim=1)                       # 累加 -> 绝对轨迹
        return traj




class NanoFSDv2(nn.Module):
    """图像 -> BEV -> 轨迹 + 占据 的端到端模型（多任务）。"""

    def __init__(self, d_model: int = 128, bev_size: int = 100,
                 n_heights: int = 4, n_cams: int = 6, fut_steps: int = 12,
                 n_query: int = 50, n_cls: int = 2, kv_stride: int = 2,
                 use_planner_v2: bool = True,
                 n_map_query: int = 48, n_map_points: int = 20, n_map_cls: int = 3,
                 use_vector_map: bool = True,
                 pretrained: bool = True, freeze_backbone: bool = True):
        """
        规划头版本开关（任务 4.2，决策 4）：
          use_planner_v2=True （默认）：用交互式 PlanningHeadV2（ego query 先 attend BEV、
            再 attend 检测头 agent query_emb），自车轨迹能「看见」周围 agent。
          use_planner_v2=False：回退到旧 PlanningHead（BEV 全局池化），用于对照学习。

        设计取舍：无论开关如何，`self.planner`（v1）**始终构造**，这样：
          1) 旧的 v1 checkpoint 仍能原样加载（state_dict 键齐全）；
          2) 开关可随时翻转而不缺子模块。
        仅当 use_planner_v2=True 时额外构造 self.planner_v2（它需要 agent query 交互）。

        矢量地图头开关（vector-map-head 任务 2.3，组件 4，需求 7.1/7.2/7.6）：
          use_vector_map=True （默认）：构造 self.vector_map（VectorMapHead），与
            self.detection **平行**挂在同一张 BEV 特征之后——检测头管动态 agent、矢量
            地图头管静态矢量地图元素（车道分隔线/道路边界/人行横道）。这与 UniAD 里
            TrackFormer 与 MapFormer 平行的结构一致。
          use_vector_map=False：不构造 self.vector_map，forward 返回 dict 不含 'vmap' 键
            （下游按 out.get("vmap") 容错）。
        与 use_planner_v2 同款处理：checkpoint 存该开关、重建时按开关构造，保证旧
        checkpoint（无 vector_map.* 键）仍能按同版本开关加载。
        """
        super().__init__()
        self.use_planner_v2 = use_planner_v2
        self.use_vector_map = use_vector_map
        self.backbone = ImageBackbone(pretrained=pretrained, freeze=freeze_backbone)
        self.bev = BEVCrossAttention(
            d_model=d_model, feat_channels=self.backbone.out_channels,
            bev_size=bev_size, n_heights=n_heights, n_cams=n_cams)
        # v1 规划头：始终构造（见上：兼容旧 checkpoint + 可回退）
        self.planner = PlanningHead(d_model=d_model, fut_steps=fut_steps)
        self.occ_head = OccupancyHead(d_model=d_model)      # 占据：有没有物体
        self.map_head = OccupancyHead(d_model=d_model)      # 地图：可行驶区（同结构，不同头）
        # 检测头：N 个 agent query attend BEV 出框/类别/置信度（UniAD query 主线起点）
        self.detection = DetectionHead(d_model=d_model, n_query=n_query, n_cls=n_cls,
                                       bev_size=bev_size, kv_stride=kv_stride)
        # 矢量地图头：M 个 map query attend BEV 出一条条折线/类别/置信度，与 self.detection
        # 平行（UniAD MapFormer 的类比）。仅在启用时构造；关闭时 dict 不含 'vmap' 键。
        if use_vector_map:
            self.vector_map = VectorMapHead(
                d_model=d_model, n_query=n_map_query, n_points=n_map_points,
                n_cls=n_map_cls, bev_size=bev_size, kv_stride=kv_stride)
        # 运动头：复用检测头的 query_emb 回归每个 agent 未来轨迹（UniAD query 主线第二棒）
        self.motion = MotionHead(d_model=d_model, fut_steps=fut_steps)
        # v2 交互式规划头：仅在启用时构造（UniAD query 主线终点，消费检测头 query_emb）
        if use_planner_v2:
            self.planner_v2 = PlanningHeadV2(d_model=d_model, fut_steps=fut_steps,
                                             bev_size=bev_size, kv_stride=kv_stride)
        self.cmd_emb = nn.Embedding(3, d_model)             # 导航指令 0=左 1=直 2=右

    def forward(self, images: Tensor, sample_uv: Tensor, sample_valid: Tensor,
                command: Tensor | None = None, return_bev: bool = False) -> dict:
        """统一 dict 输出（任务 5.2）：一次 forward 返回全部头的输出。

        输入:
          images       (B, V, 3, H, W)
          sample_uv    (B, S*S, V, Z, 2)
          sample_valid (B, S*S, V, Z)
          command      (B,) long，导航指令 0/1/2；None 时规划头不加意图（兼容旧调用）
          return_bev   True 时在返回 dict 中额外带上 'bev' 键（其余键始终存在）

        输出（dict）:
          {
            'traj':      (B,Tf,2)       自车未来轨迹（标准化）
            'occ':       (B,1,S,S)      占据 logit
            'map':       (B,1,S,S)      可行驶区 logit（内部由 self.map_head 产出）
            'det':       dict           检测头输出 {'box','cls','score','query_emb','agent_fut'}
            'agent_fut': (B,N,Tf,2)     他车预测未来轨迹（= det['agent_fut']，顶层再暴露一份便于读取）
            ['vmap':     dict]          use_vector_map=True 时附带矢量地图头输出
                                        {'pts','cls','score','query_emb','anchor_xy'}；
                                        use_vector_map=False 时不含此键（下游按 out.get("vmap") 容错）
            ['bev':      (B,D,S,S)]     return_bev=True 时附带 BEV 特征图
          }

        设计决策（任务 5.2）：
          1) **始终计算检测 + 运动**，让 dict 永远完整——det 与 agent_fut 恒可读。N=50
             个 query 的 decoder 开销很小（见 design.md 性能考量），不区分调用方是否需要
             感知输出，消除了旧版 `return_det` 开关带来的「元组长度随调用而变」的脆弱性。
          2) 移除旧 `return_det` 参数：调用方统一读 dict['det'] / dict['agent_fut']。
             保留 `return_bev` 参数，但它现在只控制 dict 是否包含 'bev' 键（不再改变其余
             输出的结构）。
          3) 规划头版本开关（self.use_planner_v2）不变：
             - True：PlanningHeadV2 消费已算好的 det['query_emb']/det['score'] 做 ego↔agent
               交互得 traj（检测结果直接复用，不重复计算）。
             - False：旧 v1 self.planner(bev, cmd_emb)（BEV 全局池化），不依赖检测；但 det
               仍会被计算并放进 dict，保证返回结构一致。
        """
        feats = self.backbone(images)                       # (B,V,Cf,Hf,Wf)
        bev = self.bev(feats, sample_uv, sample_valid)      # (B,D,S,S)
        cmd_emb = self.cmd_emb(command) if command is not None else None

        # 始终计算检测 + 运动，保证 dict 完整（UniAD query 主线：检测→运动）
        det = self.detection(bev)                           # dict: box/cls/score/query_emb
        det["agent_fut"] = self.motion(det["query_emb"])    # (B,N,Tf,2) 他车未来轨迹

        # 规划头版本选择 ----------------------------------------------------------------
        if self.use_planner_v2:
            # v2 交互式规划：复用已算好的 agent query_emb / score 做 ego↔agent 交互。
            traj = self.planner_v2(bev, det["query_emb"], det["score"], cmd_emb)  # (B,Tf,2)
        else:
            # v1 回退：BEV 全局池化规划，不依赖检测头（det 仍在 dict 中，结构保持一致）
            traj = self.planner(bev, cmd_emb)               # (B,Tf,2)

        occ = self.occ_head(bev)                            # (B,1,S,S) 占据
        drivable = self.map_head(bev)                       # (B,1,S,S) 可行驶区

        out = {
            "traj": traj,
            "occ": occ,
            "map": drivable,
            "det": det,
            "agent_fut": det["agent_fut"],                  # 顶层再暴露一份，便于直接读取
        }
        # 矢量地图头（与 det 平行）：启用时追加 'vmap' 键，其余键不变；关闭时 dict 不含该键
        if self.use_vector_map:
            out["vmap"] = self.vector_map(bev)              # dict: pts/cls/score/query_emb/anchor_xy
        if return_bev:
            out["bev"] = bev
        return out


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
        out = model(images, sample_uv, sample_valid, return_bev=True)
    traj, occ, drivable, bev = out["traj"], out["occ"], out["map"], out["bev"]
    det, agent_fut = out["det"], out["agent_fut"]
    print(f"traj     : {tuple(traj.shape)}  (期望 (1,{Tf},2))")
    print(f"occ      : {tuple(occ.shape)}   (期望 (1,1,{S},{S}))")
    print(f"map      : {tuple(drivable.shape)}   (期望 (1,1,{S},{S}))")
    print(f"bev      : {tuple(bev.shape)}   (期望 (1,128,{S},{S}))")
    print(f"det.box  : {tuple(det['box'].shape)}   (期望 (1,50,5))")
    print(f"det.cls  : {tuple(det['cls'].shape)}   (期望 (1,50,2))")
    print(f"det.score: {tuple(det['score'].shape)}   (期望 (1,50,1))")
    print(f"agent_fut: {tuple(agent_fut.shape)}   (期望 (1,50,{Tf},2))")
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: 总 {n_tot:,}  可训练 {n_tr:,}（backbone 冻结则不计）")
    if dev == "cuda":
        torch.cuda.synchronize()
        print(f"峰值显存: {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")
