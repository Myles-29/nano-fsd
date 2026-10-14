"""data.py — nuScenes mini 的向量化预处理、缓存与 DataLoader 封装。

包含：Sample 结构与验证（任务 2.1）、devkit 读取与 ego 坐标系转换（2.2）、
标准化/padding/缓存（2.3）、Dataset/DataLoader（2.4）、--demo 演示入口（2.6）。

张量约定：batch 维在前（单样本不含 batch 维）。坐标统一到 ego 坐标系
（以自车当前时刻为原点，x 向前、y 向左，单位米）。
"""

from dataclasses import dataclass

import torch
from torch import Tensor

# 验证容差
_EGO_ORIGIN_EPS = 1e-3        # ego 当前帧位置范数阈值（标准化后量级）
_PADDING_ZERO_EPS = 1e-6      # padding 行全零容差
# 相邻轨迹点距离上限：nuScenes 2Hz（0.5s/步），城市限速约 25 m/s ->
# 单步位移约 12.5m；标准化除以场景尺度（约 50m）后约 0.25，取 0.5 留裕量。
_MAX_STEP_DIST = 0.5


@dataclass
class Sample:
    """单个训练样本（向量化 BEV 场景），所有张量在 ego 坐标系、已标准化。

    记号：Th=历史步数, N=agent 上限, C=每 agent 特征维, Tf=未来步数。
    """

    ego_hist: Tensor    # (Th, 4)    自车历史: [x, y, vx, vy]
    agents: Tensor      # (N, Th, C) 周围 agent 历史特征, padding 到 N
    agent_mask: Tensor  # (N,)       True=有效 agent, False=padding
    # 监督信号
    agent_gt: Tensor    # (N, 2)     周围 agent 真值相对位置（感知头用）
    traj_gt: Tensor     # (Tf, 2)    自车未来轨迹真值（规划头用）
    traj_mask: Tensor   # (Tf,)      True=有效未来步, False=末尾不足补齐位

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查设计文档「验证规则」中的不变式，不满足则抛 AssertionError。"""
        # --- 形状守恒（正确性属性 1）---
        assert self.ego_hist.ndim == 2 and self.ego_hist.shape[1] == 4, \
            f"ego_hist 形状应为 (Th, 4), 实际 {tuple(self.ego_hist.shape)}"
        assert self.traj_gt.ndim == 2, \
            f"traj_gt 形状应为 (Tf, 2), 实际 {tuple(self.traj_gt.shape)}"
        Th = self.ego_hist.shape[0]
        N = self.agents.shape[0] if self.agents.ndim >= 1 else -1
        Tf = self.traj_gt.shape[0]
        assert self.agents.ndim == 3 and self.agents.shape[1] == Th, \
            f"agents 形状应为 (N, Th, C), 实际 {tuple(self.agents.shape)}"
        assert self.agent_mask.shape == (N,), \
            f"agent_mask 形状应为 (N,), 实际 {tuple(self.agent_mask.shape)}"
        assert self.agent_mask.dtype == torch.bool, \
            f"agent_mask dtype 应为 bool, 实际 {self.agent_mask.dtype}"
        assert self.agent_gt.shape == (N, 2), \
            f"agent_gt 形状应为 (N, 2), 实际 {tuple(self.agent_gt.shape)}"
        assert self.traj_gt.shape == (Tf, 2), \
            f"traj_gt 形状应为 (Tf, 2), 实际 {tuple(self.traj_gt.shape)}"
        assert self.traj_mask.shape == (Tf,), \
            f"traj_mask 形状应为 (Tf,), 实际 {tuple(self.traj_mask.shape)}"
        assert self.traj_mask.dtype == torch.bool, \
            f"traj_mask dtype 应为 bool, 实际 {self.traj_mask.dtype}"

        # --- ego 原点性（正确性属性 6）: 当前帧（最后一帧）位置≈原点 ---
        ego_now_pos = self.ego_hist[-1, :2]
        assert torch.linalg.norm(ego_now_pos) < _EGO_ORIGIN_EPS, \
            f"ego 当前帧位置应接近原点, 实际范数 {torch.linalg.norm(ego_now_pos):.4g}"

        # --- padding 行全零: agent_mask=False 的 agents 行必须全零 ---
        if (~self.agent_mask).any():
            pad_rows = self.agents[~self.agent_mask]
            assert pad_rows.abs().max() < _PADDING_ZERO_EPS, \
                "padding(agent_mask=False) 的 agents 行必须全零"

        # --- 相邻轨迹点距离合理: 仅在有效(traj_mask)连续段上检查 ---
        if Tf >= 2:
            valid = self.traj_mask
            step = self.traj_gt[1:] - self.traj_gt[:-1]      # (Tf-1, 2)
            step_dist = torch.linalg.norm(step, dim=-1)      # (Tf-1,)
            # 仅当相邻两步都有效时该步距才有意义
            pair_valid = valid[1:] & valid[:-1]              # (Tf-1,)
            if pair_valid.any():
                max_dist = step_dist[pair_valid].max()
                assert max_dist <= _MAX_STEP_DIST, \
                    f"相邻轨迹点距离过大(>{_MAX_STEP_DIST}), 实际 {max_dist:.4g}"


# =============================================================================
# 任务 2.2：devkit 读取与 ego 坐标系转换
# -----------------------------------------------------------------------------
# 本节用 nuscenes-devkit 遍历 mini 场景，抽取各 agent 的框中心/朝向/速度/类别、
# 自车历史与未来位姿，并把全局坐标统一转换到「以自车当前时刻为原点」的 ego
# 坐标系（x 向前、y 向左，单位米）。
#
# 本任务只负责「读取 + 坐标系转换」，产出 **原始**（metric、未标准化、未 padding、
# agent 数可变）的 RawSample。标准化 / padding / 缓存见任务 2.3。
#
# 依赖（nuscenes-devkit、numpy、pyquaternion）在函数内部惰性导入，使得本模块在
# 未安装 devkit 的环境下仍可被 import（例如仅跑属性测试时）。
# =============================================================================

import numpy as np

# nuScenes mini 为 2Hz 关键帧（keyframe），相邻 sample 间隔约 0.5s。
_NUSC_DT = 0.5

# 关注的动态 agent 大类（nuScenes category 前缀）-> 类别 id。
# 其余类别（如 static_object、障碍物等）忽略，只喂动态交通参与者给感知头。
_CATEGORY_TO_CLS = {
    "vehicle": 0,
    "human": 1,
    "cyclist": 2,  # vehicle.bicycle / vehicle.motorcycle 另行归并见 _category_to_cls
}
_NUM_CLASSES = 3


def _category_to_cls(category_name: str) -> int | None:
    """把 nuScenes 的细粒度 category 名映射到我们的粗类 id；无关类别返回 None。

    例：'vehicle.car'->0, 'human.pedestrian.adult'->1,
        'vehicle.bicycle'/'vehicle.motorcycle'->2。
    """
    if category_name.startswith("vehicle.bicycle") or \
            category_name.startswith("vehicle.motorcycle"):
        return _CATEGORY_TO_CLS["cyclist"]
    if category_name.startswith("vehicle"):
        return _CATEGORY_TO_CLS["vehicle"]
    if category_name.startswith("human"):
        return _CATEGORY_TO_CLS["human"]
    return None


@dataclass
class RawSample:
    """任务 2.2 的产物：单帧场景在 ego 坐标系下的 **原始**（未标准化/未 padding）数据。

    agent 数量 ``n`` 随场景变化；特征已含类别 one-hot。时间维约定：
    index 0 为最早历史帧，index -1（第 Th-1 个）为「当前帧」（ego 原点所在时刻）。
    """

    ego_hist: np.ndarray    # (Th, 4)      自车历史 [x, y, vx, vy]，ego 系
    agents: np.ndarray      # (n, Th, C)   周围 agent 历史特征，ego 系
    agent_gt: np.ndarray    # (n, 2)       各 agent 当前帧相对位置（感知头真值）
    traj_gt: np.ndarray     # (tf, 2)      自车未来轨迹，ego 系（tf 可 < Tf）
    sample_token: str       # 当前帧 sample 的 token，便于回溯/调试


def _pose_to_matrix(translation, rotation) -> np.ndarray:
    """由 ego_pose 的 translation(3,) 与 rotation(四元数 wxyz) 构造 4x4 齐次位姿矩阵。

    返回的是 ego->global 变换（把 ego 系下的点变换到 global 系）。
    """
    from pyquaternion import Quaternion

    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = np.asarray(translation, dtype=np.float64)
    return m


def _global_to_ego_xy(points_xy: np.ndarray, global_from_ego: np.ndarray) -> np.ndarray:
    """把一组 global 系的平面点 (K,2) 变换到 ego 系（x 向前、y 向左）。

    ``global_from_ego`` 是 ego->global 的 4x4 矩阵；此处用其逆做 global->ego。
    nuScenes 的 ego 位姿本身即定义了 x 向前、y 向左、z 向上，故逆变换后直接满足约定。
    """
    ego_from_global = np.linalg.inv(global_from_ego)
    pts = np.asarray(points_xy, dtype=np.float64)
    k = pts.shape[0]
    homo = np.concatenate([pts, np.zeros((k, 1)), np.ones((k, 1))], axis=1)  # (K,4)
    out = homo @ ego_from_global.T                                           # (K,4)
    return out[:, :2]


def _rotate_global_to_ego(vec_xy: np.ndarray, global_from_ego: np.ndarray) -> np.ndarray:
    """只旋转（不平移）地把 global 系的向量 (K,2) 转到 ego 系，用于速度等方向量。"""
    rot = global_from_ego[:3, :3]                 # ego->global 旋转
    rot_inv = rot.T                               # global->ego 旋转（正交阵转置即逆）
    vec = np.asarray(vec_xy, dtype=np.float64)
    k = vec.shape[0]
    v3 = np.concatenate([vec, np.zeros((k, 1))], axis=1)  # (K,3)
    out = v3 @ rot_inv.T                                   # (K,3)
    return out[:, :2]


def _yaw_global_to_ego(yaw_global: float, global_from_ego: np.ndarray) -> float:
    """把 global 系朝向角转到 ego 系：减去 ego 自身的 yaw。"""
    from pyquaternion import Quaternion

    rot = global_from_ego[:3, :3]
    ego_yaw = Quaternion(matrix=rot).yaw_pitch_roll[0]
    return yaw_global - ego_yaw


def _quat_yaw(rotation) -> float:
    """从四元数 (wxyz) 取 global 系 yaw 角。"""
    from pyquaternion import Quaternion

    return Quaternion(rotation).yaw_pitch_roll[0]


def _sample_chain(nusc, scene) -> list[str]:
    """按时间顺序返回一个 scene 内所有 sample 的 token 链。"""
    tokens: list[str] = []
    tok = scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return tokens


def _ego_pose_of_sample(nusc, sample_token: str) -> dict:
    """取某个 sample 的自车位姿（通过其 LIDAR_TOP sample_data 关联的 ego_pose）。"""
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return nusc.get("ego_pose", sd["ego_pose_token"])


def _build_raw_sample(nusc, chain: list[str], cur_idx: int,
                      hist_steps: int, fut_steps: int) -> RawSample | None:
    """以 ``chain[cur_idx]`` 为当前帧，构造一个 ego 系下的 RawSample。

    - 历史不足 ``hist_steps`` 的帧（场景开头）直接跳过，返回 None。
    - 未来允许不足 ``fut_steps``（场景结尾），traj_gt 的 tf 可小于 fut_steps，
      padding 交由任务 2.3 处理。
    """
    # 历史窗口必须凑满 hist_steps（含当前帧）
    if cur_idx - (hist_steps - 1) < 0:
        return None

    hist_tokens = chain[cur_idx - (hist_steps - 1): cur_idx + 1]   # len == hist_steps
    fut_tokens = chain[cur_idx + 1: cur_idx + 1 + fut_steps]        # 可能不足

    # 当前帧自车位姿 -> ego->global 矩阵，作为整帧坐标系基准
    cur_pose = _ego_pose_of_sample(nusc, chain[cur_idx])
    global_from_ego = _pose_to_matrix(cur_pose["translation"], cur_pose["rotation"])

    # --- 自车历史：位置(ego 系) + 速度(有限差分，ego 系) ---
    hist_global_xy = np.array(
        [_ego_pose_of_sample(nusc, t)["translation"][:2] for t in hist_tokens],
        dtype=np.float64,
    )  # (Th, 2) global
    hist_ego_xy = _global_to_ego_xy(hist_global_xy, global_from_ego)  # (Th, 2)
    # 速度：相邻帧有限差分 / dt；首帧用后向差分补齐
    vel = np.zeros_like(hist_ego_xy)
    if hist_steps >= 2:
        vel[1:] = (hist_ego_xy[1:] - hist_ego_xy[:-1]) / _NUSC_DT
        vel[0] = vel[1]
    ego_hist = np.concatenate([hist_ego_xy, vel], axis=1).astype(np.float32)  # (Th, 4)

    # --- 自车未来轨迹（ego 系）---
    if len(fut_tokens) == 0:
        return None
    fut_global_xy = np.array(
        [_ego_pose_of_sample(nusc, t)["translation"][:2] for t in fut_tokens],
        dtype=np.float64,
    )
    traj_gt = _global_to_ego_xy(fut_global_xy, global_from_ego).astype(np.float32)  # (tf,2)

    # --- 周围 agents：以 instance 为单位，在历史窗口内收集，当前帧必须可见 ---
    cur_sample = nusc.get("sample", chain[cur_idx])
    # 收集当前帧的有效 annotation（动态类别）
    inst_to_cls: dict[str, int] = {}
    cur_ann_by_inst: dict[str, dict] = {}
    for ann_tok in cur_sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        cls = _category_to_cls(ann["category_name"])
        if cls is None:
            continue
        inst_to_cls[ann["instance_token"]] = cls
        cur_ann_by_inst[ann["instance_token"]] = ann

    # 建立「每个历史帧 instance_token -> annotation」的查表，便于取历史特征
    hist_ann_lookup: list[dict[str, dict]] = []
    for t in hist_tokens:
        s = nusc.get("sample", t)
        m: dict[str, dict] = {}
        for ann_tok in s["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            m[ann["instance_token"]] = ann
        hist_ann_lookup.append(m)

    agents_list: list[np.ndarray] = []
    agent_gt_list: list[list[float]] = []
    C = 4 + 2 + _NUM_CLASSES  # [x, y, vx, vy, cos_yaw, sin_yaw, cls_onehot(3)]
    for inst_tok, cls in inst_to_cls.items():
        feat = np.zeros((hist_steps, C), dtype=np.float32)
        prev_xy = None
        for ti, m in enumerate(hist_ann_lookup):
            ann = m.get(inst_tok)
            if ann is None:
                # 该历史帧此 agent 不可见：用零特征（有效性以当前帧为准）
                prev_xy = None
                continue
            xy_g = np.array(ann["translation"][:2], dtype=np.float64)[None]  # (1,2)
            xy_e = _global_to_ego_xy(xy_g, global_from_ego)[0]              # (2,)
            yaw_e = _yaw_global_to_ego(_quat_yaw(ann["rotation"]), global_from_ego)
            # 速度：相邻历史帧有限差分（ego 系）
            if prev_xy is not None:
                v = (xy_e - prev_xy) / _NUSC_DT
            else:
                v = np.zeros(2, dtype=np.float64)
            prev_xy = xy_e
            feat[ti, 0:2] = xy_e
            feat[ti, 2:4] = v
            feat[ti, 4] = np.cos(yaw_e)
            feat[ti, 5] = np.sin(yaw_e)
            feat[ti, 6 + cls] = 1.0
        agents_list.append(feat)
        # 感知头真值：当前帧该 agent 的相对位置
        cur_xy_g = np.array(cur_ann_by_inst[inst_tok]["translation"][:2],
                            dtype=np.float64)[None]
        cur_xy_e = _global_to_ego_xy(cur_xy_g, global_from_ego)[0]
        agent_gt_list.append([float(cur_xy_e[0]), float(cur_xy_e[1])])

    if len(agents_list) == 0:
        # 无有效 agent 的样本在任务 2.3 过滤；此处仍可返回供上层决定
        agents = np.zeros((0, hist_steps, C), dtype=np.float32)
        agent_gt = np.zeros((0, 2), dtype=np.float32)
    else:
        agents = np.stack(agents_list, axis=0)                      # (n, Th, C)
        agent_gt = np.array(agent_gt_list, dtype=np.float32)        # (n, 2)

    return RawSample(
        ego_hist=ego_hist,
        agents=agents,
        agent_gt=agent_gt,
        traj_gt=traj_gt,
        sample_token=chain[cur_idx],
    )


def load_raw_samples(cfg, dataroot: str = "data/v1.0-mini",
                     version: str = "v1.0-mini", verbose: bool = False):
    """遍历 nuScenes mini 所有场景，产出 ego 系下的 RawSample 列表（任务 2.2 入口）。

    参数
    ----
    cfg: Config，使用其中的 hist_steps / fut_steps。
    dataroot: nuScenes mini 解压根目录（含 v1.0-mini/ 元数据与 samples/）。
    version: devkit 版本名。

    说明：devkit 在此惰性导入；坐标全部转换到以各样本当前帧为原点的 ego 系。
    """
    import time

    from nuscenes.nuscenes import NuScenes

    if verbose:
        print(f"load_raw_samples: 加载 devkit（{version} @ {dataroot}）...", flush=True)
    t0 = time.time()
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    n_scenes = len(nusc.scene)
    if verbose:
        print(f"load_raw_samples: devkit 就绪，耗时 {time.time() - t0:.1f}s，"
              f"共 {n_scenes} 个 scene，开始逐 scene 预处理...", flush=True)

    out: list[RawSample] = []
    for si, scene in enumerate(nusc.scene):
        chain = _sample_chain(nusc, scene)
        for cur_idx in range(len(chain)):
            rs = _build_raw_sample(nusc, chain, cur_idx,
                                   cfg.hist_steps, cfg.fut_steps)
            if rs is not None:
                out.append(rs)
        # 每 50 个 scene 报一次进度，便于判断没卡死（数据量大时很有用）
        if verbose and (si + 1) % 50 == 0:
            print(f"load_raw_samples: scene {si + 1}/{n_scenes}  "
                  f"累计 RawSample={len(out)}  ({time.time() - t0:.1f}s)", flush=True)
    if verbose:
        print(f"load_raw_samples: 共产出 {len(out)} 个 RawSample，"
              f"总耗时 {time.time() - t0:.1f}s", flush=True)
    return out


# =============================================================================
# 任务 2.3：标准化、padding 与缓存
# -----------------------------------------------------------------------------
# RawSample（metric、agent 数可变、tf 可 < Tf）-> Sample（标准化、定长 padding）
# -> 堆叠成大张量一次性 torch.save 到 .pt，训练时直接 load，避免每个 epoch 调 devkit。
#
# 标准化：位置 (x,y) 与速度 (vx,vy) 都除以 SCENE_SCALE；cos/sin/one-hot 不变。
# padding：agent 按与自车的距离（agent_gt 范数）升序排列，n > N 时只保留最近的 N 个
#          （近处 agent 对规划最相关）；不足 N 的位置填零、agent_mask=False。
# 过滤：有效 agent 数为 0 的样本丢弃（错误处理场景 2）；未来不足 Tf 时补零并置
#       traj_mask=False（场景 3）；未通过 Sample.validate 的样本计数后跳过，不崩溃。
# =============================================================================

SCENE_SCALE = 50.0  # 场景尺度（米）：坐标/速度除以它后落在 [-1, 1] 量级附近


def normalize(x):
    """米 -> 标准化单位。适用于 (..., 2) 位置或 (..., 4) [x,y,vx,vy]（np 或 Tensor）。"""
    return x / SCENE_SCALE


def denormalize(x):
    """标准化单位 -> 米。viz 中用于 traj_pred (..., 2) 等网络输出。"""
    return x * SCENE_SCALE


def raw_to_sample(raw: RawSample, cfg) -> Sample | None:
    """把一个 RawSample 转成标准化、padding 到 (N, Tf) 的 Sample。

    有效 agent 数为 0 时返回 None；不满足验证规则时由 Sample 抛 AssertionError。
    """
    N, Tf, C = cfg.n_agents, cfg.fut_steps, cfg.agent_feat_dim
    Th = raw.ego_hist.shape[0]
    n = raw.agents.shape[0]
    if n == 0:
        return None
    assert raw.agents.shape[2] == C, \
        f"agent 特征维 {raw.agents.shape[2]} 与 cfg.agent_feat_dim={C} 不一致"

    # 按距离升序，截断到最近的 N 个
    order = np.argsort(np.linalg.norm(raw.agent_gt, axis=-1), kind="stable")[:N]
    k = len(order)

    agents = np.zeros((N, Th, C), dtype=np.float32)
    agents[:k] = raw.agents[order]
    agents[:k, :, :4] = normalize(agents[:k, :, :4])   # 只缩放 [x,y,vx,vy]
    agent_gt = np.zeros((N, 2), dtype=np.float32)
    agent_gt[:k] = normalize(raw.agent_gt[order])
    agent_mask = np.zeros(N, dtype=bool)
    agent_mask[:k] = True

    # 未来轨迹：超过 Tf 截断，不足 Tf 末尾补零
    tf = min(raw.traj_gt.shape[0], Tf)
    traj_gt = np.zeros((Tf, 2), dtype=np.float32)
    traj_gt[:tf] = normalize(raw.traj_gt[:tf])
    traj_mask = np.zeros(Tf, dtype=bool)
    traj_mask[:tf] = True

    return Sample(  # __post_init__ 中执行 validate()
        ego_hist=torch.from_numpy(normalize(raw.ego_hist).astype(np.float32)),
        agents=torch.from_numpy(agents),
        agent_mask=torch.from_numpy(agent_mask),
        agent_gt=torch.from_numpy(agent_gt),
        traj_gt=torch.from_numpy(traj_gt),
        traj_mask=torch.from_numpy(traj_mask),
    )


_CACHE_KEYS = ("ego_hist", "agents", "agent_mask", "agent_gt", "traj_gt", "traj_mask")


def convert_raw_samples(raws: list[RawSample], cfg) -> tuple[dict, dict]:
    """批量转换并堆叠：返回 (cache 字典, 统计计数)。

    cache: 各字段为 (S, ...) 堆叠张量，另含 sample_tokens(list[str]) 与 scene_scale。
    """
    samples: list[Sample] = []
    tokens: list[str] = []
    n_raw: list[int] = []   # 截断前的原始 agent 数（--demo 统计截断用）
    stats = {"raw": len(raws), "no_agent": 0, "invalid": 0, "kept": 0}
    for raw in raws:
        try:
            s = raw_to_sample(raw, cfg)
        except AssertionError:
            stats["invalid"] += 1
            continue
        if s is None:
            stats["no_agent"] += 1
            continue
        samples.append(s)
        tokens.append(raw.sample_token)
        n_raw.append(raw.agents.shape[0])
    stats["kept"] = len(samples)
    assert samples, f"没有可用样本: {stats}"

    cache = {k: torch.stack([getattr(s, k) for s in samples]) for k in _CACHE_KEYS}
    cache["sample_tokens"] = tokens
    cache["n_agents_raw"] = torch.tensor(n_raw, dtype=torch.long)  # (S,)
    cache["scene_scale"] = SCENE_SCALE
    return cache, stats


def build_cache(cfg, out_path: str = "data/cache_mini.pt",
                dataroot: str = "data/v1.0-mini", version: str = "v1.0-mini",
                verbose: bool = True) -> dict:
    """devkit 读取 -> 转换 -> torch.save 到 out_path（一次性预处理）。"""
    import os

    raws = load_raw_samples(cfg, dataroot=dataroot, version=version, verbose=verbose)
    cache, stats = convert_raw_samples(raws, cfg)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(cache, out_path)
    if verbose:
        print(f"build_cache: {stats} -> {out_path}")
        for k in _CACHE_KEYS:
            print(f"  {k}: {tuple(cache[k].shape)} {cache[k].dtype}")
    return cache


def load_cache(path: str = "data/cache_mini.pt") -> dict:
    """读取 build_cache 产出的缓存字典（只含张量/字符串/浮点，weights_only 安全加载）。"""
    return torch.load(path, weights_only=True)


# =============================================================================
# 任务 2.4：Dataset / DataLoader 封装
# -----------------------------------------------------------------------------
# 缓存已是 (S, ...) 堆叠张量，Dataset 只做按索引切片；dataset[i] 返回单样本 Sample
# （无 batch 维，便于示例用法里 sample.ego_hist[None]）。DataLoader 用自定义
# collate_fn 把若干 Sample 堆叠成 Batch（batch 维在前），训练循环用 batch.xxx 访问。
# Batch 单独定义而不复用 Sample，因为 Sample.validate 假设无 batch 维的形状。
# =============================================================================

from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class Batch:
    """一个 batch 的样本，各字段为 Sample 对应字段在最前面加 batch 维 B。"""

    ego_hist: Tensor    # (B, Th, 4)
    agents: Tensor      # (B, N, Th, C)
    agent_mask: Tensor  # (B, N)      bool
    agent_gt: Tensor    # (B, N, 2)
    traj_gt: Tensor     # (B, Tf, 2)
    traj_mask: Tensor   # (B, Tf)     bool

    def to(self, device) -> "Batch":
        """把所有张量搬到 device，返回新的 Batch。"""
        return Batch(**{k: getattr(self, k).to(device) for k in _CACHE_KEYS})


def collate_samples(samples: list[Sample]) -> Batch:
    """DataLoader 的 collate_fn：把 Sample 列表沿新的第 0 维堆叠成 Batch。"""
    return Batch(**{k: torch.stack([getattr(s, k) for s in samples]) for k in _CACHE_KEYS})


class NanoFSDDataset(Dataset):
    """基于 build_cache 产出的缓存字典（或 .pt 路径）的 Dataset。"""

    def __init__(self, cache: dict | str = "data/cache_mini.pt") -> None:
        if isinstance(cache, str):
            cache = load_cache(cache)
        self.tensors = {k: cache[k] for k in _CACHE_KEYS}
        self.sample_tokens: list[str] = list(cache.get("sample_tokens", []))
        S = self.tensors["ego_hist"].shape[0]
        assert all(t.shape[0] == S for t in self.tensors.values()), \
            "缓存各字段的样本数 S 不一致"
        self._len = S

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Sample:
        # 构造 Sample 时会再次执行 validate()，保证喂给模型的样本满足不变式
        return Sample(**{k: t[idx] for k, t in self.tensors.items()})


def make_loader(cfg, cache: dict | str = "data/cache_mini.pt",
                shuffle: bool = True, drop_last: bool = False) -> DataLoader:
    """按 cfg.batch_size 产出 Batch 的 DataLoader（数据量小，单进程加载即可）。"""
    return DataLoader(NanoFSDDataset(cache), batch_size=cfg.batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      collate_fn=collate_samples)


def split_train_val_indices(n: int, val_ratio: float = 0.2,
                            seed: int = 0) -> tuple[list[int], list[int]]:
    """按固定 seed 把 [0, n) 划成 (val_idx, train_idx)，与训练用的划分完全一致。

    单一真相来源：make_train_val_loaders 与可视化选样都调用它，保证"验证集"口径一致
    （同一 seed/val_ratio 下，可视化选到的样本就是训练时的验证样本，模型未训过）。
    """
    assert 0.0 < val_ratio < 1.0, f"val_ratio 应在 (0,1)，实际 {val_ratio}"
    assert n >= 2, f"样本数太少（{n}），无法划分 train/val"
    n_val = max(1, int(n * val_ratio))
    n_val = min(n_val, n - 1)                      # 保证训练集至少 1 个样本
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    return perm[:n_val], perm[n_val:]


def make_train_val_loaders(cfg, cache: dict | str = "data/cache_mini.pt",
                           val_ratio: float = 0.2, seed: int = 0,
                           drop_last: bool = False) -> tuple[DataLoader, DataLoader]:
    """把数据集随机切成 train / val 两个 DataLoader（可复现，供严肃训练使用）。

    与 make_loader 的区别：这里做 **固定种子的随机划分**，训练集 shuffle、验证集
    不 shuffle，便于判断过拟合与挑选最优 checkpoint。

    参数
    ----
    cfg: Config，使用其中的 batch_size。
    cache: 缓存字典或 .pt 路径。
    val_ratio: 验证集占比，∈ (0, 1)。向下取整得到 val 样本数，至少 1 个。
    seed: 划分用的随机种子，保证同一 seed 下划分可复现。
    drop_last: 是否丢弃训练集最后一个不满 batch。

    返回
    ----
    (train_loader, val_loader)，各自产出 Batch。
    """
    assert 0.0 < val_ratio < 1.0, f"val_ratio 应在 (0,1)，实际 {val_ratio}"
    dataset = NanoFSDDataset(cache)
    n = len(dataset)
    assert n >= 2, f"样本数太少（{n}），无法划分 train/val"

    val_idx, train_idx = split_train_val_indices(n, val_ratio=val_ratio, seed=seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=drop_last, collate_fn=collate_samples)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collate_samples)
    return train_loader, val_loader


# =============================================================================
# 任务 2.6：演示入口 python data.py --demo（只读缓存，不重新预处理）
# =============================================================================

def demo(cfg, cache_path: str = "data/cache_mini.pt",
         out_png: str = "outputs/step2_data.png", idx: int = 0) -> None:
    """打印缓存统计与字段形状，画一张原始场景 BEV 图，并用 DataLoader 取一个 batch。"""
    import os
    import sys

    if not os.path.exists(cache_path):
        print(f"缓存 {cache_path} 不存在，请先运行 `python data.py` 生成缓存。")
        sys.exit(1)
    cache = load_cache(cache_path)
    S = cache["ego_hist"].shape[0]
    n_valid = cache["agent_mask"].sum(dim=1)                 # (S,)
    n_traj_pad = int((~cache["traj_mask"]).any(dim=1).sum())
    if "n_agents_raw" in cache:
        n_trunc = str(int((cache["n_agents_raw"] > cfg.n_agents).sum()))
    else:
        n_trunc = "N/A, regenerate cache with `python data.py`"

    print(f"[stats] {cache_path}")
    print(f"  样本数: {S}")
    print(f"  平均有效 agent 数: {n_valid.float().mean():.2f} (N={cfg.n_agents})")
    print(f"  agent 被截断(>N)的样本数: {n_trunc}")
    print(f"  未来轨迹被 pad 的样本数: {n_traj_pad} (Tf={cfg.fut_steps})")
    print("[shapes] 缓存字段 (S 在前):")
    for k in _CACHE_KEYS:
        print(f"  {k}: {tuple(cache[k].shape)} {cache[k].dtype}")

    # --- BEV 图：单个样本，denormalize 回米；ego 原点，x 向前、y 向左 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = NanoFSDDataset(cache)[idx]
    agents_xy = denormalize(s.agent_gt[s.agent_mask]).numpy()   # (k, 2)
    ego_xy = denormalize(s.ego_hist[:, :2]).numpy()             # (Th, 2)
    traj_xy = denormalize(s.traj_gt[s.traj_mask]).numpy()       # (tf, 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(agents_xy[:, 0], agents_xy[:, 1], s=15, c="gray",
               label=f"agents ({len(agents_xy)})")
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], "b.-", label="ego history")
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "g.-", label="GT future")
    ax.plot(0, 0, "b^", ms=10, label="ego (origin)")
    ax.set_aspect("equal")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    tok = cache.get("sample_tokens", [""] * S)[idx]
    ax.set_title(f"sample {idx} {tok[:8]}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[bev] 已保存 {out_png}")

    # --- DataLoader 取一个 batch ---
    batch = next(iter(make_loader(cfg, cache, shuffle=False)))
    print(f"[batch] batch_size={cfg.batch_size}:")
    for k in _CACHE_KEYS:
        t = getattr(batch, k)
        print(f"  {k}: {tuple(t.shape)} {t.dtype}")


if __name__ == "__main__":
    # 预处理（默认 mini）:
    #   python data.py
    # 预处理 trainval（元数据在 *_meta 目录；只读向量标注，不碰图像）:
    #   python data.py --dataroot data/v1.0-trainval/v1.0-trainval_meta \
    #       --version v1.0-trainval --out data/cache_trainval.pt
    # 只读缓存做演示（--cache 指定要看哪个缓存）:
    #   python data.py --demo --cache data/cache_trainval.pt
    import argparse

    from config import Config

    parser = argparse.ArgumentParser(
        description="nano-fsd 数据预处理 / 缓存演示")
    parser.add_argument("--demo", action="store_true", help="只读缓存做演示")
    parser.add_argument("--dataroot", type=str, default="data/v1.0-mini",
                        help="nuScenes 根目录（trainval 指向 *_meta 元数据目录）")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="devkit 版本名，如 v1.0-mini / v1.0-trainval")
    parser.add_argument("--out", type=str, default="data/cache_mini.pt",
                        help="预处理输出缓存路径")
    parser.add_argument("--cache", type=str, default=None,
                        help="--demo 时要读取的缓存路径；默认用 --out 的值")
    args = parser.parse_args()

    if args.demo:
        cache_path = args.cache or args.out
        demo(Config(), cache_path=cache_path)
    else:
        build_cache(Config(), out_path=args.out,
                    dataroot=args.dataroot, version=args.version)
