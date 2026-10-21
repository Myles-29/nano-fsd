"""v2 台阶2-第1步：构建"元数据缓存"（不含图像像素）。

遍历样本，把每个样本的轻量信息存成一个 .pt，供 Dataset 快速索引：
  - 6 路图像的文件绝对路径（图像在线读，不缓存像素）
  - 6 路相机的缩放后内参 K、外参 cam->ego 的 R/t
  - 真值未来轨迹（ego 系，米），以及有效步 mask

不缓存 sample_uv（太大，每样本 ~2MB）；它由相机参数在训练循环里 GPU 上现算，很快。
devkit 加载一次（~20s），之后训练完全不碰 devkit。

用法:
  python build_meta_cache.py --num 2000 --out cache/meta_2000.pt   # 小子集
  python build_meta_cache.py --out cache/meta_full.pt              # 全量
"""

import argparse
import os

import numpy as np
import torch

from bev_geometry import CAMS, DATAROOT, VERSION, _resolve_image_path

ORIG_H, ORIG_W = 900, 1600
IMG_H, IMG_W = 224, 400
FUT_STEPS = 12


def future_traj_ego(nusc, sample, fut_steps):
    """未来 fut_steps 帧自车轨迹转到当前帧 ego 系（米），返回 (traj (Tf,2), mask (Tf,))。"""
    from pyquaternion import Quaternion

    def pose(s):
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        return nusc.get("ego_pose", sd["ego_pose_token"])

    cur = pose(sample)
    m = np.eye(4)
    m[:3, :3] = Quaternion(cur["rotation"]).rotation_matrix
    m[:3, 3] = np.array(cur["translation"])
    g2e = np.linalg.inv(m)

    traj = np.zeros((fut_steps, 2), dtype=np.float32)
    mask = np.zeros(fut_steps, dtype=bool)
    tok, k = sample["next"], 0
    while tok and k < fut_steps:
        s = nusc.get("sample", tok)
        p = np.array(pose(s)["translation"] + [1.0])
        traj[k] = (g2e @ p)[:2]
        mask[k] = True
        tok, k = s["next"], k + 1
    return traj, mask


# BEV 占据图配置（和模型 BEV 对齐）
BEV_RANGE, BEV_SIZE = 50.0, 100
# 只给动态交通参与者生成占据（车/人/骑行者），和 v1 的类别过滤一致
_DYNAMIC_PREFIX = ("vehicle", "human")

# 矢量地图 GT 配置（vector-map-head；与模型 M query / P 点对齐）
MAX_MAP = 48          # 每帧矢量折线定长上限，与模型 M map query 对齐
MAP_POINTS = 20       # 每条折线固定点数 P（弧长等距重采样）
# 地图图层 -> 3 类归并：0=车道分隔线、1=道路边界（取外轮廓）、2=人行横道
_MAP_LAYERS = {
    "lane_divider": 0, "road_divider": 0,     # 0=车道分隔线
    "road_segment": 1, "drivable_area": 1,    # 1=道路边界（取外轮廓）
    "ped_crossing": 2,                        # 2=人行横道
}


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """把任意点数的折线按**弧长等距**重采样成固定 n 个点，返回 (n,2)。

    前置条件：
      - pts 为 (K,2) 的 ego 系折线点序，K>=2（点数<2 的折线由调用方跳过）。
      - n>=1 为目标点数。
    后置条件：
      - 返回 (n,2) float32，恰好 n 个点。
      - 端点保持：输出首点=原折线首点、输出末点=原折线末点（n>=2 时）。
      - 点沿累积弧长均匀分布（相邻输出点间弧长在数值容差内近似相等）。
    退化处理：
      - 总弧长为 0（所有点重合）时，返回原折线首点重复 n 次。
    """
    pts = np.asarray(pts, dtype=np.float64)
    # 相邻点欧氏距离 -> 累积弧长 cum（长度 K，cum[0]=0）
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))   # (K-1,)
    cum = np.concatenate([[0.0], np.cumsum(seg)])            # (K,)
    total = cum[-1]
    # 退化：所有点重合（总弧长为 0）-> 首点重复 n 次
    if total == 0.0:
        return np.repeat(pts[0:1], n, axis=0).astype(np.float32)

    targets = np.linspace(0.0, total, n)                     # (n,) 等距弧长目标
    out = np.empty((n, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        # 在 cum 上二分找 t 所落的段 [j, j+1]
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)                     # 夹到有效段范围
        seg_len = cum[j + 1] - cum[j]
        # 段内线性插值比例（零长段取 0，避免除零）
        frac = 0.0 if seg_len == 0.0 else (t - cum[j]) / seg_len
        out[i] = pts[j] + frac * (pts[j + 1] - pts[j])
    # 端点保持：数值上钉死首/末点为原端点，消除浮点误差
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out.astype(np.float32)


_MAP_CACHE = {}       # city -> NuScenesMap，避免每样本重载


def _get_map(city: str):
    """按城市缓存 NuScenesMap（首次加载慢，之后复用）。"""
    from nuscenes.map_expansion.map_api import NuScenesMap
    if city not in _MAP_CACHE:
        print(f"[meta]   加载地图 {city} ...", flush=True)
        _MAP_CACHE[city] = NuScenesMap(dataroot=DATAROOT, map_name=city)
    return _MAP_CACHE[city]


def build_drivable_bev(nusc, sample, city: str) -> np.ndarray:
    """生成以自车为中心、ego 对齐的"可行驶区"BEV mask (BEV_SIZE,BEV_SIZE) uint8。

    用 map API 的 get_map_mask：以自车全局 (x,y,yaw) 为中心取一个
    2R×2R 的 patch，栅格化 drivable_area 图层；再转成"x前y左"的 ego 朝向。
    """
    from pyquaternion import Quaternion

    ego = nusc.get("ego_pose",
                   nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
    ex, ey = ego["translation"][0], ego["translation"][1]
    yaw = Quaternion(ego["rotation"]).yaw_pitch_roll[0]      # 全局 yaw（弧度）
    yaw_deg = np.degrees(yaw)

    nmap = _get_map(city)
    # patch_box=(center_x, center_y, height, width)；patch_angle=自车 yaw（经诊断确认）
    patch_box = (ex, ey, 2 * BEV_RANGE, 2 * BEV_RANGE)
    mask = nmap.get_map_mask(patch_box, yaw_deg, ["drivable_area"],
                             canvas_size=(BEV_SIZE, BEV_SIZE))[0]
    # 诊断(diag_map_angle2)确认：转置 T 后，可行驶区与"行=x前,列=y左"的 ego BEV 对齐
    # （GT 轨迹/占据落在灰色路面内）。固化此校正。
    drivable = mask.T.astype(np.uint8)                       # (BEV_SIZE,BEV_SIZE)
    return drivable


def build_occupancy(nusc, sample) -> np.ndarray:
    """当前帧 agent 的 3D 框投到 BEV 俯视 100x100 栅格，框覆盖的格子=1。

    流程：框(global) --当前帧 ego_pose 逆--> ego 系 -> 取框底面 4 角 (x,y) ->
    栅格化（框的旋转矩形内的格子标 1）。返回 (BEV_SIZE, BEV_SIZE) float32。
    """
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    ego = nusc.get("ego_pose",
                   nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
    m = np.eye(4)
    m[:3, :3] = Quaternion(ego["rotation"]).rotation_matrix
    m[:3, 3] = np.array(ego["translation"])
    g2e = np.linalg.inv(m)

    occ = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    # BEV 栅格坐标：行=x(前)、列=y(左)，范围 [-R,R]
    cell = (2 * BEV_RANGE) / BEV_SIZE

    def to_ij(x, y):
        i = int((x + BEV_RANGE) / cell)
        j = int((y + BEV_RANGE) / cell)
        return i, j

    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        if not ann["category_name"].startswith(_DYNAMIC_PREFIX):
            continue
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        corners_g = box.corners().T                       # (8,3) global
        homo = np.concatenate([corners_g, np.ones((8, 1))], 1)
        corners_e = (homo @ g2e.T)[:, :3]                 # (8,3) ego
        # 底面 4 角的 (x,y)（z 最小的 4 个）
        order = np.argsort(corners_e[:, 2])[:4]
        poly = corners_e[order][:, :2]                    # (4,2) ego 系 (x,y)
        # 用框的 AABB 包围盒遍历候选格子，再判点是否在旋转矩形内
        xmin, ymin = poly.min(0); xmax, ymax = poly.max(0)
        i0, j0 = to_ij(xmin, ymin); i1, j1 = to_ij(xmax, ymax)
        for i in range(max(i0, 0), min(i1 + 1, BEV_SIZE)):
            for j in range(max(j0, 0), min(j1 + 1, BEV_SIZE)):
                px = -BEV_RANGE + (i + 0.5) * cell
                py = -BEV_RANGE + (j + 0.5) * cell
                if _point_in_quad(px, py, poly):
                    occ[i, j] = 1.0
    return occ


def _point_in_quad(px, py, quad) -> bool:
    """点 (px,py) 是否在凸四边形 quad(4,2) 内（叉积同号法）。"""
    # 按角度排序成凸多边形顺序
    c = quad.mean(0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]
    sign = None
    for k in range(4):
        ax, ay = q[k]
        bx, by = q[(k + 1) % 4]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


# 结构化 agent 列表配置
MAX_GT = 50                       # 每帧 agent 定长上限，和模型 N query 对齐
# 2 类归并：0=车(vehicle)，1=人(human，含骑行者 cyclist 就近归入人)
_CLS_VEHICLE, _CLS_HUMAN = 0, 1


def _agent_class(category_name: str) -> int | None:
    """nuScenes 类别名归并为 2 类：0=车、1=人（骑行者就近归入人）。

    - "human.*"            -> 1（行人）
    - "cycle"/"cyclist"    -> 1（骑行者就近归入人）
    - "vehicle.*"          -> 0（车，但排除无动力的 bicycle/motorcycle 本体？）
    说明：nuScenes 里骑行者本体标注在 vehicle.bicycle / vehicle.motorcycle，而骑车的人
    另有 human.pedestrian。为"就近归入人"，把 bicycle/motorcycle 归到人(1)，其余 vehicle 归车(0)。
    非动态类别返回 None（跳过）。
    """
    if category_name.startswith("human"):
        return _CLS_HUMAN
    if category_name.startswith("vehicle"):
        # 骑行类（两轮）就近归入"人"，其余机动车归"车"
        if "bicycle" in category_name or "motorcycle" in category_name:
            return _CLS_HUMAN
        return _CLS_VEHICLE
    return None


def _ego_g2e(nusc, sample) -> np.ndarray:
    """返回把 global 坐标转到该 sample 当前帧 ego 系的 4x4 变换矩阵（global->ego）。"""
    from pyquaternion import Quaternion

    ego = nusc.get("ego_pose",
                   nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
    m = np.eye(4)
    m[:3, :3] = Quaternion(ego["rotation"]).rotation_matrix
    m[:3, 3] = np.array(ego["translation"])
    return np.linalg.inv(m)


def _clip_polyline_segments(pts_ego: np.ndarray):
    """把一条 ego 系折线按"点是否落在 BEV 范围内"切成若干连续子段。

    - pts_ego: (K,2) ego 系点序 [x,y]。
    - 规则：保留满足 |x|<=BEV_RANGE 且 |y|<=BEV_RANGE 的点；一旦遇到越界点就断开，
      前后在范围内的点各自成段（跨界几何裁成多段）。
    - 返回 list[np.ndarray]，每个元素是 (k,2) 的连续在范围内点段（可能为空列表）。
      点数 <2 的段由调用方跳过。
    """
    inside = (np.abs(pts_ego[:, 0]) <= BEV_RANGE) & (np.abs(pts_ego[:, 1]) <= BEV_RANGE)
    segments, cur = [], []
    for k in range(len(pts_ego)):
        if inside[k]:
            cur.append(pts_ego[k])
        elif cur:
            segments.append(np.asarray(cur, dtype=np.float32))
            cur = []
    if cur:
        segments.append(np.asarray(cur, dtype=np.float32))
    return segments


def _coords_to_ego(coords, g2e: np.ndarray) -> np.ndarray:
    """把一串 global (x,y) 坐标经 g2e(4x4) 转到当前帧 ego 系，返回 (K,2) [x前,y左]。

    地图节点只有平面 (x,y)，z 取 0（地图在水平面上）；齐次坐标 [x,y,0,1] 乘 g2e。
    """
    arr = np.asarray(coords, dtype=np.float64)              # (K,2) global
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    homo = np.concatenate([arr[:, :2], np.zeros((len(arr), 1)),
                           np.ones((len(arr), 1))], axis=1)  # (K,4)
    ego = (homo @ g2e.T)[:, :2]                              # (K,2) ego 系 (x前,y左)
    return ego.astype(np.float32)


def build_map_polylines(nusc, sample, city: str) -> dict:
    """抽取当前帧 ego 周围 ±BEV_RANGE 的矢量地图元素，转 ego 系、裁剪、重采样、pad。

    流程（对齐设计组件 1）：
      1. 取 ego 全局 (x,y)；用 _get_map(city) 的 NuScenesMap 以 get_records_in_radius
         查询半径内 _MAP_LAYERS 各图层记录。
      2. 几何提取：
         - lane_divider / road_divider：取其 line 几何的节点点序（extract_line）。
         - road_segment / drivable_area：取多边形外轮廓边界（extract_polygon.exterior）。
         - ped_crossing：取多边形外轮廓（闭合环，extract_polygon.exterior）。
      3. 用 _ego_g2e(nusc, sample) 把每个点从 global 转到当前帧 ego 系（x 前 / y 左）。
      4. 裁剪到 |x|<=BEV_RANGE 且 |y|<=BEV_RANGE 内的点段；跨界几何裁成多段各自成折线；
         点数 <2 的段跳过。
      5. resample_polyline 到 MAP_POINTS 点；按到自车（原点）距离近者优先，超 MAX_MAP 截断。
      6. 返回定长 pad 字典（numpy，未归一化，由 Dataset 侧统一 /SCENE_SCALE）：
         map_pts (MAX_MAP,P,2)、map_cls (MAX_MAP,)、map_mask (MAX_MAP,)。

    空场景安全：ego 周围无任何地图元素时 map_mask 全 False。
    地图加载失败（城市 map 缺失/异常）时 try/except 置 map_mask 全 False 并返回（不中断构建）。
    """
    map_pts = np.zeros((MAX_MAP, MAP_POINTS, 2), dtype=np.float32)
    map_cls = np.zeros((MAX_MAP,), dtype=np.int64)
    map_mask = np.zeros((MAX_MAP,), dtype=bool)
    empty = {"map_pts": map_pts, "map_cls": map_cls, "map_mask": map_mask}

    try:
        ego = nusc.get("ego_pose",
                       nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
        ex, ey = float(ego["translation"][0]), float(ego["translation"][1])
        g2e = _ego_g2e(nusc, sample)                        # 当前帧 global->ego
        nmap = _get_map(city)
        layers = list(_MAP_LAYERS.keys())
        # 以自车全局 (x,y) 为中心、半径 BEV_RANGE 的方形 patch 查各图层记录
        recs = nmap.get_records_in_radius(ex, ey, BEV_RANGE, layers)
    except Exception as e:  # noqa: BLE001
        print(f"[meta]   矢量地图抽取失败({city}): {e}，该样本 map_mask 置全 False")
        return empty

    # 候选折线：(cls, resampled_pts (P,2), dist 到自车原点)
    candidates = []

    def _add_global_coords(coords, cls_id):
        """把一串 global 坐标转 ego、裁剪成段、重采样、登记为候选折线。"""
        if coords is None or len(coords) < 2:
            return
        pts_ego = _coords_to_ego(coords, g2e)               # (K,2) ego 系
        for seg in _clip_polyline_segments(pts_ego):
            if len(seg) < 2:                                # 点数 <2 跳过
                continue
            rs = resample_polyline(seg, MAP_POINTS)         # (P,2) 弧长等距
            # 到自车（ego 原点）距离：取折线各点距原点最小值（近者优先）
            dist = float(np.min(np.hypot(rs[:, 0], rs[:, 1])))
            candidates.append((cls_id, rs, dist))

    for layer, cls_id in _MAP_LAYERS.items():
        for tok in recs.get(layer, []):
            try:
                rec = nmap.get(layer, tok)
                if layer in ("lane_divider", "road_divider"):
                    # 线几何：extract_line -> shapely LineString
                    line = nmap.extract_line(rec["line_token"])
                    if line.is_empty:
                        continue
                    _add_global_coords(list(line.coords), cls_id)
                elif layer == "drivable_area":
                    # drivable_area 含多个多边形（polygon_tokens 复数），各取外轮廓
                    for ptok in rec["polygon_tokens"]:
                        poly = nmap.extract_polygon(ptok)
                        if poly.is_empty:
                            continue
                        _add_global_coords(list(poly.exterior.coords), cls_id)
                else:
                    # road_segment / ped_crossing：单多边形外轮廓（ped_crossing 闭合环）
                    poly = nmap.extract_polygon(rec["polygon_token"])
                    if poly.is_empty:
                        continue
                    _add_global_coords(list(poly.exterior.coords), cls_id)
            except Exception:  # noqa: BLE001
                # 单条记录几何异常不影响其余元素
                continue

    if not candidates:                                      # 空场景：mask 全 False
        return empty

    # 按到自车距离近者优先，超 MAX_MAP 截断
    candidates.sort(key=lambda c: c[2])
    for k, (cls_id, rs, _dist) in enumerate(candidates[:MAX_MAP]):
        map_pts[k] = rs
        map_cls[k] = cls_id
        map_mask[k] = True

    return {"map_pts": map_pts, "map_cls": map_cls, "map_mask": map_mask}


def build_agent_list(nusc, sample, fut_steps) -> dict:
    """解析当前帧动态 agent 的框 + 未来轨迹，返回定长 pad 到 MAX_GT 的字典。

    - 框：复用 build_occupancy 的 global->ego 变换（当前帧 ego_pose 逆），取 Box 中心 (x,y)、
      朝向 yaw（ego 系）、尺寸 (w,l)；类别归并为 2 类（0=车、1=人，骑行者就近归入人）。
    - 未来：沿 sample_annotation 的 instance 链（ann["next"]）找同一 instance 在未来帧的中心，
      统一转到"当前帧 ego 系"，减去当前中心得相对未来轨迹；instance 消失则对应步 mask=False。
    - 过滤：非动态类别（返回 None）跳过。
    - pad：超过 MAX_GT 时按到自车距离近者优先保留；无动态 agent 时 agent_mask 全 False。

    返回字典字段（均为 numpy，未做 SCENE_SCALE 归一化，由 Dataset 侧统一归一化）：
      agent_box      (MAX_GT, 5)      [x, y, yaw, w, l]（ego 系，米/弧度）
      agent_cls      (MAX_GT,)        int64，类别 id ∈ {0,1}
      agent_mask     (MAX_GT,)        bool，有效 agent 位
      agent_fut      (MAX_GT, Tf, 2)  相对各 agent 当前中心的未来轨迹（当前帧 ego 系）
      agent_fut_mask (MAX_GT, Tf)     bool，未来有效步
    """
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    g2e = _ego_g2e(nusc, sample)     # 当前帧 global->ego（与 build_occupancy 一致）

    def box_center_ego(ann) -> np.ndarray:
        """把一个 annotation 的框中心（global translation）转到当前帧 ego 系的 (x,y)。"""
        p = np.array(list(ann["translation"]) + [1.0])
        return (g2e @ p)[:2].astype(np.float32)

    # 1) 解析当前帧每个动态 agent
    agents = []  # list of dict: {cls, x, y, yaw, w, l, inst, dist}
    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        cls = _agent_class(ann["category_name"])
        if cls is None:
            continue
        # Box 在 global 系；用 g2e 旋转到 ego 系取中心/朝向
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        # 中心（ego 系）
        cx, cy = box_center_ego(ann)
        # 朝向：把框的全局朝向旋转到 ego 系，取 x前-y左 平面内的 yaw
        rot_e = g2e[:3, :3] @ box.orientation.rotation_matrix
        yaw = float(np.arctan2(rot_e[1, 0], rot_e[0, 0]))
        # 尺寸：nuScenes size=[w, l, h]，取 (w, l)
        w, l = float(ann["size"][0]), float(ann["size"][1])
        agents.append({
            "cls": cls, "x": float(cx), "y": float(cy), "yaw": yaw,
            "w": w, "l": l, "inst": ann["instance_token"], "ann_tok": ann_tok,
            "dist": float(np.hypot(cx, cy)),
        })

    # 2) 超过 MAX_GT：按到自车距离近者优先保留
    agents.sort(key=lambda a: a["dist"])
    agents = agents[:MAX_GT]
    m = len(agents)

    # 3) 预分配定长输出
    agent_box = np.zeros((MAX_GT, 5), dtype=np.float32)
    agent_cls = np.zeros((MAX_GT,), dtype=np.int64)
    agent_mask = np.zeros((MAX_GT,), dtype=bool)
    agent_fut = np.zeros((MAX_GT, fut_steps, 2), dtype=np.float32)
    agent_fut_mask = np.zeros((MAX_GT, fut_steps), dtype=bool)

    # 4) 沿 instance 链填未来相对轨迹
    for k, a in enumerate(agents):
        agent_box[k] = [a["x"], a["y"], a["yaw"], a["w"], a["l"]]
        agent_cls[k] = a["cls"]
        agent_mask[k] = True
        cur_xy = np.array([a["x"], a["y"]], dtype=np.float32)
        # 从当前 annotation 沿 "next" 走，收集未来 fut_steps 帧的同一 instance 中心
        tok, step = nusc.get("sample_annotation", a["ann_tok"])["next"], 0
        while tok and step < fut_steps:
            ann_f = nusc.get("sample_annotation", tok)
            fut_xy = box_center_ego(ann_f)            # 未来帧框中心转到"当前帧 ego 系"
            agent_fut[k, step] = fut_xy - cur_xy      # 相对当前中心
            agent_fut_mask[k, step] = True
            tok, step = ann_f["next"], step + 1
        # instance 中途消失：剩余步的 agent_fut_mask 保持 False（已初始化为 False）

    return {
        "agent_box": agent_box,
        "agent_cls": agent_cls,
        "agent_mask": agent_mask,
        "agent_fut": agent_fut,
        "agent_fut_mask": agent_fut_mask,
        "n_agent": m,
    }


# 矢量地图 3 类配色/名称（与设计组件 6 一致）：0=分隔线、1=边界、2=人行横道
_MAP_CLS_COLORS = {0: "tab:green", 1: "tab:red", 2: "tab:purple"}
_MAP_CLS_NAMES = {0: "lane divider", 1: "road boundary", 2: "ped crossing"}


def viz_vmap_gt(meta: str, num: int, out_dir: str = "outputs") -> list:
    """从缓存取样本，复用 drivable 栅格作底图，在 BEV 俯视图叠 GT 矢量折线做人眼 sanity check。

    - 从 meta 缓存（含 map_pts/map_cls/map_mask 字段）取前 num 个样本。
    - 底图：复用 drivable (S,S) 可行驶区栅格，用 pcolormesh 按物理坐标铺（横轴=y 左、纵轴=x 前）。
    - 叠加：每条有效 GT 折线按类着色画出，遵循 ego 系 x 前 / y 左 约定 -> plot(y, x)。
    - 保存 outputs/step1_vmap_gt_{idx}.png，供人眼确认分隔线/边界/人行横道位置与类别合理。

    前置：meta 为 build_meta_cache 产出的 .pt（含矢量地图字段）。
    返回：生成的 PNG 路径列表。
    """
    import matplotlib
    matplotlib.use("Agg")                      # 无显示环境，Agg 后端
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    blob = torch.load(meta, weights_only=False)
    recs = blob["records"]
    n = min(num, len(recs))
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for idx in range(n):
        rec = recs[idx]
        map_pts = np.asarray(rec["map_pts"], dtype=np.float32)   # (MAX_MAP,P,2) 米，ego 系
        map_cls = np.asarray(rec["map_cls"])                     # (MAX_MAP,)
        map_mask = np.asarray(rec["map_mask"], dtype=bool)       # (MAX_MAP,)

        fig, ax = plt.subplots(figsize=(7, 7))

        # 道路底图：drivable 栅格 (S,S)，行 i=x前、列 j=y左。用 pcolormesh 按物理坐标铺，
        # 横轴=y 左、纵轴=x 前（与 visualize.run_topdown_viz 同口径，避免转置错位）。
        drivable = rec.get("drivable")
        if drivable is not None:
            road = np.asarray(drivable, dtype=np.float32)
            S = road.shape[0]
            axis = np.linspace(-BEV_RANGE, BEV_RANGE, S)
            Xy = np.tile(axis, (S, 1))           # 横轴 y，沿列 j 变化
            Yx = np.tile(axis[:, None], (1, S))  # 纵轴 x，沿行 i 变化
            ax.pcolormesh(Xy, Yx, road, cmap="Greys", alpha=0.35, vmin=0, vmax=1,
                          shading="nearest", zorder=0)

        # 叠加 GT 折线：仅有效位，按类着色；横轴=y 左、纵轴=x 前 -> plot(y, x)
        cls_count = {0: 0, 1: 0, 2: 0}
        for k in range(len(map_mask)):
            if not map_mask[k]:
                continue
            cls = int(map_cls[k])
            cls_count[cls] = cls_count.get(cls, 0) + 1
            pts = map_pts[k]                      # (P,2) 列0=x前, 列1=y左
            color = _MAP_CLS_COLORS.get(cls, "gray")
            ax.plot(pts[:, 1], pts[:, 0], "-", color=color, lw=1.4, zorder=2)

        ax.plot(0, 0, "b^", ms=10, zorder=3)      # 自车位置
        ax.set_xlim(BEV_RANGE, -BEV_RANGE)        # 左(+y)在左侧
        ax.set_ylim(-BEV_RANGE, BEV_RANGE)        # 前(+x)在上
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("y left (m)"); ax.set_ylabel("x forward (m)")
        n_valid = int(map_mask.sum())
        ax.set_title(f"GT vector map on drivable (idx={idx}, {n_valid} polylines)")

        # 图例：仅列出本图出现过的类别
        handles = [Line2D([0], [0], color=_MAP_CLS_COLORS[c], lw=2,
                          label=f"{_MAP_CLS_NAMES[c]} ({cls_count[c]})")
                   for c in sorted(cls_count) if cls_count[c] > 0]
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8)

        path = os.path.join(out_dir, f"step1_vmap_gt_{idx}.png")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
        saved.append(path)
        print(f"[viz-vmap] idx={idx}: {n_valid} 条折线 "
              f"(分隔线={cls_count.get(0,0)} 边界={cls_count.get(1,0)} "
              f"人行横道={cls_count.get(2,0)}) -> {path}", flush=True)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=None, help="取前 N 个样本；默认全量")
    parser.add_argument("--out", type=str, default="cache/meta_full.pt")
    parser.add_argument("--viz-vmap", type=int, default=None, metavar="N",
                        help="仅从缓存取 N 个样本把 GT 矢量折线叠路面底图可视化（不构建缓存）")
    parser.add_argument("--viz-meta", type=str, default="cache/meta_2000_vmap.pt",
                        help="--viz-vmap 读取的缓存路径")
    parser.add_argument("--viz-out", type=str, default="outputs",
                        help="--viz-vmap 产物输出目录")
    args = parser.parse_args()

    # 可视化分支：只读缓存，不加载 devkit、不构建
    if args.viz_vmap is not None:
        viz_vmap_gt(args.viz_meta, args.viz_vmap, args.viz_out)
        return

    from nuscenes.nuscenes import NuScenes

    print(f"[meta] 加载 devkit（{VERSION}）...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    sx, sy = IMG_W / ORIG_W, IMG_H / ORIG_H

    n_total = len(nusc.sample)
    n = n_total if args.num is None else min(args.num, n_total)
    print(f"[meta] 处理 {n}/{n_total} 个样本...", flush=True)

    recs = []
    skipped = 0
    for i in range(n):
        sample = nusc.sample[i]
        # 只保留未来至少 1 帧的样本（能提供规划监督）
        traj, mask = future_traj_ego(nusc, sample, FUT_STEPS)
        if not mask.any():
            skipped += 1
            continue

        img_paths, Ks, Rs, ts = [], [], [], []
        ok = True
        for cam in CAMS:
            sd = nusc.get("sample_data", sample["data"][cam])
            calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            p = _resolve_image_path(sd["filename"])
            if p is None:
                ok = False
                break
            img_paths.append(p)
            from pyquaternion import Quaternion
            K = np.array(calib["camera_intrinsic"], dtype=np.float32)
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
            Ks.append(K)
            Rs.append(Quaternion(calib["rotation"]).rotation_matrix.astype(np.float32))
            ts.append(np.array(calib["translation"], dtype=np.float32))
        if not ok:
            skipped += 1
            continue

        # 导航指令（UniAD 风格，从真值轨迹推）：看末端有效点横向 y（ego 系 y 向左）
        #   y > +thr -> 左转(0)；y < -thr -> 右转(2)；否则 -> 直行(1)
        last = np.where(mask)[0].max()
        y_end = float(traj[last, 1])
        _CMD_THR = 4.0
        command = 0 if y_end > _CMD_THR else (2 if y_end < -_CMD_THR else 1)

        occ = build_occupancy(nusc, sample)               # (100,100) float32
        # 结构化 agent 列表（框 + 未来轨迹，pad 到 MAX_GT；检测/运动头监督用）
        agents = build_agent_list(nusc, sample, FUT_STEPS)
        # 当前帧所属城市（用于取对应地图）
        log = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])
        city = log["location"]
        try:
            drivable = build_drivable_bev(nusc, sample, city)
        except Exception as e:  # noqa: BLE001
            print(f"[meta]   地图生成失败({city}): {e}，该样本 drivable 置零")
            drivable = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        # 矢量地图 GT 折线（vector-map-head；复用上面已解析的 city，内部 try/except 容错）
        vmap = build_map_polylines(nusc, sample, city)
        recs.append({
            "token": sample["token"],
            "img_paths": img_paths,                       # list[str] 长度 6
            "K": np.stack(Ks),                            # (6,3,3)
            "R": np.stack(Rs),                            # (6,3,3)
            "t": np.stack(ts),                            # (6,3)
            "traj": traj,                                 # (Tf,2) 米
            "traj_mask": mask,                            # (Tf,)
            "occ": occ.astype(np.uint8),                  # (100,100) BEV 占据真值
            "drivable": drivable.astype(np.uint8),        # (100,100) 可行驶区真值
            "command": int(command),                      # 导航指令 0=左 1=直 2=右
            # 结构化 agent 真值（向后兼容：旧缓存无此 5 字段，Dataset 侧填零容错）
            "agent_box": agents["agent_box"],             # (MAX_GT,5) [x,y,yaw,w,l] ego 系
            "agent_cls": agents["agent_cls"],             # (MAX_GT,) int64 类别 0=车 1=人
            "agent_mask": agents["agent_mask"],           # (MAX_GT,) bool 有效位
            "agent_fut": agents["agent_fut"],             # (MAX_GT,Tf,2) 相对未来轨迹
            "agent_fut_mask": agents["agent_fut_mask"],   # (MAX_GT,Tf) bool 未来有效步
            # 矢量地图真值（向后兼容：旧缓存无此 3 字段，Dataset 侧填零容错）
            "map_pts": vmap["map_pts"],                   # (MAX_MAP,P,2) ego 系折线点(米)
            "map_cls": vmap["map_cls"],                   # (MAX_MAP,) int64 类别 0/1/2
            "map_mask": vmap["map_mask"],                 # (MAX_MAP,) bool 有效折线位
        })
        if (i + 1) % 500 == 0:
            print(f"[meta]   {i+1}/{n}  已收集 {len(recs)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"records": recs, "img_hw": (IMG_H, IMG_W),
                "orig_hw": (ORIG_H, ORIG_W), "fut_steps": FUT_STEPS}, args.out)
    print(f"[meta] 完成：有效 {len(recs)}，跳过 {skipped} -> {args.out}")


if __name__ == "__main__":
    main()
