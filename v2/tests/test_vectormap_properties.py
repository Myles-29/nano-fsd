"""矢量地图头（vector-map-head）属性测试与边界单测（hypothesis）。

任务 1.2：针对 `build_meta_cache.py` 的 `resample_polyline`，验证正确性属性 7
（重采样等距性）并补充三个边界单测。

- **属性 7 重采样等距性**：对任意合法折线输入，`resample_polyline(pts, n)` 恒满足：
    1. 返回恰好 n 个点，形状 `(n,2)`；
    2. 首/末点与原折线首/末点一致（端点保持）；
    3. 相邻输出点间弧长在数值容差内近似相等（弧长等距）；
    4. 对已均匀分布的输入近似幂等（再重采样同点数不显著改变几何）。

- 边界单测：2 点直线、所有点重合（退化为首点重复）、闭合环。

记号：n 为目标点数（对应 MAP_POINTS，默认 20）。

验证：正确性属性 7；需求 2.1, 2.2, 2.3, 2.5, 10.4。
"""

import os
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_meta_cache import resample_polyline  # noqa: E402


# ---------------------------------------------------------------------------
# 生成器：随机合法折线（K>=2 点，坐标在 ego 系合理范围内）
# ---------------------------------------------------------------------------

def _polylines(min_pts: int = 2, max_pts: int = 30):
    """生成 (K,2) 折线点序，K 在 [min_pts, max_pts]，坐标 ∈ [-50,50]。"""
    coord = st.floats(min_value=-50.0, max_value=50.0,
                      allow_nan=False, allow_infinity=False, width=32)
    return st.integers(min_pts, max_pts).flatmap(
        lambda k: st.lists(st.tuples(coord, coord), min_size=k, max_size=k)
    ).map(lambda rows: np.asarray(rows, dtype=np.float64))


def _arclengths(pts: np.ndarray) -> np.ndarray:
    """返回相邻点间欧氏距离（弧长段），形状 (len(pts)-1,)。"""
    return np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))


def _cumulative_arclength(pts: np.ndarray) -> np.ndarray:
    """返回每个点沿折线的累积弧长，形状 (len(pts),)，首元素为 0。"""
    seg = _arclengths(pts)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _reference_equal_arclength(path: np.ndarray, n: int) -> np.ndarray:
    """独立参考实现：沿折线累积弧长在 linspace(0,total,n) 处线性插值取点。

    用作"弧长等距"的黄金基准：第 i 个点恰好落在原折线弧长 i/(n-1)·total 处。
    对折返折线也无歧义（直接按弧长正向定位，不做逆投影）。
    """
    cum = _cumulative_arclength(path)
    total = cum[-1]
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(max(j, 0), len(path) - 2)
        seg_len = cum[j + 1] - cum[j]
        frac = 0.0 if seg_len == 0.0 else (t - cum[j]) / seg_len
        out[i] = path[j] + frac * (path[j + 1] - path[j])
    return out


# ---------------------------------------------------------------------------
# 属性 7：重采样等距性
# ---------------------------------------------------------------------------

@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(pts=_polylines(), n=st.integers(2, 40))
def test_resample_shape_and_endpoints(pts, n):
    """Feature: vector-map-head, Property 7 (resample shape + endpoints).

    返回恰好 n 点、形状 (n,2)、首/末点与原折线首/末点精确一致（端点保持）。
    """
    out = resample_polyline(pts, n)
    assert out.shape == (n, 2)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # 端点保持：实现里把首/末钉死为原端点，容差设紧（仅 float32 转换误差）。
    np.testing.assert_allclose(out[0], pts[0].astype(np.float32), atol=1e-4)
    np.testing.assert_allclose(out[-1], pts[-1].astype(np.float32), atol=1e-4)


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(pts=_polylines(), n=st.integers(3, 40))
def test_resample_equal_arclength(pts, n):
    """Feature: vector-map-head, Property 7 (equal arc-length spacing).

    弧长等距：每个输出点沿**原折线**的累积弧长应落在等距刻度上
    （第 i 个点 ≈ i/(n-1)·total）。基准用"沿原折线的弧长位置"而非"相邻输出点
    直线距离"——折线可能折返（往返同一段），此时相邻输出点直线距离会失真，
    但沿弧长的位置仍应等距（这正是实现保证的性质）。
    """
    pts = np.asarray(pts, dtype=np.float64)
    total = _cumulative_arclength(pts)[-1]
    assume(total > 1e-3)   # 跳过近乎重合的退化折线

    out = resample_polyline(pts, n).astype(np.float64)
    # 与独立的"按弧长等距插值"参考实现逐点比对：输出点必落在等距弧长刻度上。
    ref = _reference_equal_arclength(pts, n)
    np.testing.assert_allclose(out, ref, atol=1e-4 + 1e-5 * total)


def _straight_segments(max_pts: int = 20):
    """生成一条随机方向的直线段上的（任意点数）共线点序。

    直线段上的"弧长等距重采样"是精确的不动点：无拐角，再重采样不改变几何。
    这是幂等性成立的最干净的输入子集（任意带尖角的折线因拐角被跳过而非不动点）。
    """
    coord = st.floats(min_value=-40.0, max_value=40.0,
                      allow_nan=False, allow_infinity=False, width=32)
    frac = st.floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False, width=32)

    def _build(a, b, fracs):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ts = np.sort(np.array([0.0] + list(fracs) + [1.0]))
        return np.stack([a + t * (b - a) for t in ts], axis=0)

    return st.tuples(
        st.tuples(coord, coord),   # 端点 a
        st.tuples(coord, coord),   # 端点 b
        st.lists(frac, min_size=0, max_size=max_pts - 2),  # 中间点插值比例
    ).map(lambda t: _build(*t))


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seg=_straight_segments(), n=st.integers(2, 40))
def test_resample_idempotent_on_uniform(seg, n):
    """Feature: vector-map-head, Property 7 (idempotence on uniform input).

    对已均匀分布的输入再以相同点数重采样，几何不变。用直线段上的共线点序作
    输入：弧长等距重采样在直线段上是精确不动点，故 resample(resample)==resample。
    """
    total = _cumulative_arclength(seg)[-1]
    assume(total > 1e-3)   # 跳过退化（端点几乎重合）
    once = resample_polyline(seg, n)
    twice = resample_polyline(once, n)
    # 直线段上严格幂等（仅 float32 往返误差，容差随段长缩放）。
    np.testing.assert_allclose(twice, once, atol=1e-4 + 1e-5 * total)


# ---------------------------------------------------------------------------
# 边界单测：2 点直线、所有点重合、闭合环
# ---------------------------------------------------------------------------

def test_boundary_two_point_line():
    """边界：2 点直线重采样到 n 点，应落在该直线上且弧长严格等距。

    需求 2.1/2.2/2.3：形状 (n,2)、端点保持、相邻点等距。
    """
    pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    n = 5
    out = resample_polyline(pts, n)
    assert out.shape == (n, 2)
    # 应沿 x 轴均匀分布：x = 0, 2.5, 5, 7.5, 10；y 恒 0。
    expected_x = np.linspace(0.0, 10.0, n)
    np.testing.assert_allclose(out[:, 0], expected_x, atol=1e-4)
    np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-4)
    # 弧长严格等距（直线无拐角）。
    seg = _arclengths(out.astype(np.float64))
    np.testing.assert_allclose(seg, seg[0], atol=1e-4)


def test_boundary_all_points_coincident():
    """边界：所有点重合（总弧长为 0），应返回首点重复 n 次（需求 2.5）。"""
    pts = np.array([[3.0, -2.0]] * 4, dtype=np.float64)
    n = 20
    out = resample_polyline(pts, n)
    assert out.shape == (n, 2)
    # 每一行都等于首点。
    np.testing.assert_allclose(out, np.tile(pts[0].astype(np.float32), (n, 1)),
                               atol=1e-6)


def test_boundary_closed_ring():
    """边界：闭合环（首点=末点），重采样后仍闭合且端点保持。

    用一个正方形环（5 个点，首末重合）验证形状守恒与端点保持（人行横道外轮廓场景）。
    """
    pts = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],   # 闭合：末点=首点
    ], dtype=np.float64)
    n = 20
    out = resample_polyline(pts, n)
    assert out.shape == (n, 2)
    assert np.isfinite(out).all()
    # 端点保持 -> 输出仍闭合（首点=末点）。
    np.testing.assert_allclose(out[0], out[-1], atol=1e-4)
    np.testing.assert_allclose(out[0], pts[0].astype(np.float32), atol=1e-4)
    # 弧长等距：与"按弧长等距插值"参考实现逐点一致（避免正方形拐角处相邻点
    # 直线距离短于弧长造成的失真）。
    pts64 = pts.astype(np.float64)
    ref = _reference_equal_arclength(pts64, n)
    np.testing.assert_allclose(out.astype(np.float64), ref, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))


# ===========================================================================
# 任务 1.6：数据层矢量字段属性测试（hypothesis）
#
# 针对 dataset.py 的 NuScenesBEVDataset.__getitem__ 输出的矢量地图三字段，验证：
#   - 属性 1 形状守恒：map_pts (48,20,2) / map_cls (48,) / map_poly_mask (48,)，
#     无 NaN/Inf；dtype 符合契约（float32 / long / bool）。
#   - 属性 8 空场景安全：无地图元素样本 map_poly_mask 全 False、读取不报错。
#
# 两类输入（复用 test_data_properties.py 的合成缓存模式）：
#   - 带 map_* 字段的合成缓存 record（有效折线数 0..MAP_MAX+8，含 0 条的空场景）；
#   - 缺 map_* 字段的旧缓存 record（向后兼容路径：填零 + map_poly_mask 全 False）。
#
# 验证：正确性属性 1、8；需求 1.7, 3.4。
# ===========================================================================

import torch  # noqa: E402

from dataset import (  # noqa: E402
    MAP_MAX,
    MAP_POINTS,
    SCENE_SCALE,
    NuScenesBEVDataset,
)

# 测试用小图尺寸（在线读图走 PIL，用极小图省时省内存）
_IMG_H, _IMG_W = 8, 12
_N_CAM = 6
_TF = 12          # 未来步数（fut_steps），与 build_meta_cache 默认一致
_N_CLS = 3        # 地图元素类别数：0=分隔线,1=边界,2=人行横道


def _write_dummy_images(tmp_dir, n: int) -> list:
    """写 n 张极小 PNG，返回路径列表（供 record['img_paths'] 用）。"""
    from PIL import Image

    paths = []
    for i in range(n):
        arr = np.random.randint(0, 256, (_IMG_H, _IMG_W, 3), dtype=np.uint8)
        p = os.path.join(tmp_dir, f"cam_{i}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _base_record() -> dict:
    """构造一条不含 map_* 字段的最小 record（相机参数 / 轨迹等必备字段）。"""
    return {
        "img_paths": [],
        "K": np.random.randn(_N_CAM, 3, 3).astype(np.float32),
        "R": np.random.randn(_N_CAM, 3, 3).astype(np.float32),
        "t": np.random.randn(_N_CAM, 3).astype(np.float32),
        "traj": np.random.randn(_TF, 2).astype(np.float32),
        "traj_mask": np.ones(_TF, dtype=bool),
    }


def _add_map_fields(rec: dict, n_poly: int) -> dict:
    """给 record 加入矢量地图字段，前 min(n_poly,MAP_MAX) 条有效。

    有效折线坐标用较大范围（±80 米）以便验证 /SCENE_SCALE 归一化不改变形状；
    pad 的无效折线置零、map_mask 为 False。
    """
    m = min(max(n_poly, 0), MAP_MAX)

    map_pts = np.zeros((MAP_MAX, MAP_POINTS, 2), dtype=np.float32)
    map_cls = np.zeros((MAP_MAX,), dtype=np.int64)
    map_mask = np.zeros((MAP_MAX,), dtype=bool)

    if m > 0:
        map_pts[:m] = np.random.uniform(-80, 80, (m, MAP_POINTS, 2)).astype(np.float32)
        map_cls[:m] = np.random.randint(0, _N_CLS, m).astype(np.int64)
        map_mask[:m] = True

    rec["map_pts"] = map_pts
    rec["map_cls"] = map_cls
    rec["map_mask"] = map_mask
    return rec


def _make_dataset(tmp_dir, records: list) -> NuScenesBEVDataset:
    """把 records 写成临时缓存 .pt 并构造 Dataset。"""
    img_paths = _write_dummy_images(tmp_dir, _N_CAM)
    for rec in records:
        rec["img_paths"] = img_paths
    blob = {"records": records, "img_hw": (_IMG_H, _IMG_W), "fut_steps": _TF}
    meta_path = os.path.join(tmp_dir, "meta_vmap_test.pt")
    torch.save(blob, meta_path)
    return NuScenesBEVDataset(meta_path)


def assert_map_shapes(out: dict) -> None:
    """属性 1：矢量地图三字段形状与 dtype 符合契约，且 map_pts 无 NaN/Inf。"""
    assert out["map_pts"].shape == (MAP_MAX, MAP_POINTS, 2)
    assert out["map_pts"].dtype == torch.float32
    assert torch.isfinite(out["map_pts"]).all()          # 无 NaN/Inf
    assert out["map_cls"].shape == (MAP_MAX,)
    assert out["map_cls"].dtype == torch.long
    assert out["map_poly_mask"].shape == (MAP_MAX,)
    assert out["map_poly_mask"].dtype == torch.bool


# ---------------------------------------------------------------------------
# 属性 1：形状守恒（带 map_* 字段的新缓存，折线数随机含 0）
# ---------------------------------------------------------------------------

@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow,
                                 HealthCheck.function_scoped_fixture])
@given(n_poly=st.integers(0, MAP_MAX + 8))
def test_map_shape_conservation_with_fields(tmp_path_factory, n_poly):
    """Feature: vector-map-head, Property 1 (new cache with map fields).

    **Validates: 正确性属性 1；需求 1.7**
    """
    tmp_dir = str(tmp_path_factory.mktemp("cache_with_map"))
    rec = _add_map_fields(_base_record(), n_poly)
    ds = _make_dataset(tmp_dir, [rec])
    out = ds[0]
    assert_map_shapes(out)
    # 有效位数量 = min(n_poly, MAP_MAX)
    assert int(out["map_poly_mask"].sum()) == min(n_poly, MAP_MAX)


def test_map_shape_conservation_legacy_cache(tmp_path):
    """Feature: vector-map-head, Property 1 (legacy cache without map fields).

    旧缓存缺 map_* 字段 -> 向后兼容填零 + map_poly_mask 全 False。
    **Validates: 正确性属性 1；需求 3.4**
    """
    rec = _base_record()
    ds = _make_dataset(str(tmp_path), [rec])
    out = ds[0]
    assert_map_shapes(out)
    assert int(out["map_poly_mask"].sum()) == 0          # 旧缓存：mask 全 False
    assert torch.allclose(out["map_pts"], torch.zeros_like(out["map_pts"]))


@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow,
                                 HealthCheck.function_scoped_fixture])
@given(n_poly=st.integers(1, MAP_MAX))
def test_map_normalization_preserves_shape(tmp_path_factory, n_poly):
    """Feature: vector-map-head, Property 1 (normalization keeps shape).

    map_pts 的 xy 被 /SCENE_SCALE 归一化，形状仍 (MAP_MAX,P,2)，数值正确。
    **Validates: 正确性属性 1；需求 1.7, 3.4**
    """
    tmp_dir = str(tmp_path_factory.mktemp("cache_map_norm"))
    rec = _add_map_fields(_base_record(), n_poly)
    raw_pts = rec["map_pts"].copy()                      # 归一化前的原始米坐标
    ds = _make_dataset(tmp_dir, [rec])
    out = ds[0]
    assert_map_shapes(out)
    # 全部点按 SCENE_SCALE 归一化（pad 位本就是 0，/scale 仍为 0）
    np.testing.assert_allclose(out["map_pts"].numpy(), raw_pts / SCENE_SCALE,
                               rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 属性 8：空场景安全（无地图元素 map_poly_mask 全 False，读取不报错）
# ---------------------------------------------------------------------------

def test_map_empty_scene_safety_with_fields(tmp_path):
    """Feature: vector-map-head, Property 8 (empty scene, map fields present).

    0 条有效折线：map_poly_mask 全 False、读取不报错。
    **Validates: 正确性属性 8；需求 1.7, 3.4**
    """
    rec = _add_map_fields(_base_record(), 0)             # 0 条折线
    ds = _make_dataset(str(tmp_path), [rec])
    out = ds[0]                                          # 读取不报错
    assert_map_shapes(out)
    assert not out["map_poly_mask"].any()               # 全 False


def test_map_empty_scene_safety_legacy(tmp_path):
    """Feature: vector-map-head, Property 8 (empty scene, legacy cache).

    完全无 map_* 字段：向后兼容路径下 map_poly_mask 全 False、读取不报错。
    **Validates: 正确性属性 8；需求 1.7, 3.4**
    """
    rec = _base_record()                                 # 完全无 map_* 字段
    ds = _make_dataset(str(tmp_path), [rec])
    out = ds[0]                                          # 读取不报错
    assert_map_shapes(out)
    assert not out["map_poly_mask"].any()


# ===========================================================================
# 任务 2.2：地图头 VectorMapHead 属性测试（hypothesis）
#
# 针对 model.py 的 VectorMapHead.forward，验证两条正确性属性（镜像检测头属性 1、2）：
#   - 属性 1 形状守恒：随机合法 BEV 输入下，pts (B,M,P,2) / cls (B,M,n_cls) /
#     score (B,M,1) / query_emb (B,M,D) 形状恒定，且全部有限（无 NaN/Inf）。
#   - 属性 2 query 顺序/置换等变：地图头无 batch 间耦合，打乱 batch 内样本顺序后，
#     输出对应打乱（置换等变）；同一 query 的 pts/cls/score/query_emb 顺序一一对齐。
#
# 为提速：用较小的 d_model / bev_size / n_query 构造地图头（与 test_detection_properties
# 的做法一致），eval() 模式关闭随机性，torch.manual_seed 固定初始化以便复现。
#
# 验证：正确性属性 1、2；需求 4.1, 4.2, 4.3, 4.4。
# ===========================================================================

from model import VectorMapHead  # noqa: E402

# 地图头测试用的小尺寸配置（远小于生产 d_model=128/bev_size=100，纯为跑测提速）
_VD = 32          # d_model
_VS = 16          # BEV 边长 S
_VM = 12          # n_query（M）
_VP = 8           # n_points（P）
_VN_CLS = 3       # 类别数
_VKV_STRIDE = 2   # K/V 下采样步长


def _make_vmap_head(d_model: int = _VD, n_query: int = _VM, n_points: int = _VP,
                    n_cls: int = _VN_CLS, n_layers: int = 2,
                    bev_size: int = _VS) -> VectorMapHead:
    """构造一个确定性初始化的矢量地图头（固定 seed，便于复现）。"""
    torch.manual_seed(0)
    return VectorMapHead(d_model=d_model, n_query=n_query, n_points=n_points,
                         n_layers=n_layers, n_cls=n_cls, bev_size=bev_size,
                         kv_stride=_VKV_STRIDE)


# ---------------------------------------------------------------------------
# 属性 1：形状守恒
# ---------------------------------------------------------------------------

@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(1, 3), n_query=st.sampled_from([1, 5, _VM]),
       n_points=st.sampled_from([2, _VP]), n_cls=st.sampled_from([2, 3]))
def test_vmap_shape_conservation(B, n_query, n_points, n_cls):
    """Feature: vector-map-head, Property 1 (VectorMapHead output shapes).

    随机合法 BEV 输入下，地图头四个输出形状恒定且全部有限（无 NaN/Inf）。
    anchor_xy 固定为 (M,2)。

    **Validates: 正确性属性 1；需求 4.1, 4.2, 4.3**
    """
    head = _make_vmap_head(n_query=n_query, n_points=n_points, n_cls=n_cls).eval()
    bev = torch.randn(B, _VD, _VS, _VS)
    with torch.no_grad():
        out = head(bev)
    assert out["pts"].shape == (B, n_query, n_points, 2)
    assert out["cls"].shape == (B, n_query, n_cls)
    assert out["score"].shape == (B, n_query, 1)
    assert out["query_emb"].shape == (B, n_query, _VD)
    assert out["anchor_xy"].shape == (n_query, 2)
    # 全部输出均为有限浮点，无 NaN/Inf
    for k in ("pts", "cls", "score", "query_emb", "anchor_xy"):
        assert torch.isfinite(out[k]).all(), f"{k} 含 NaN/Inf"


# ---------------------------------------------------------------------------
# 属性 2：query 顺序 / 置换等变
# ---------------------------------------------------------------------------

@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(2, 4), seed=st.integers(0, 10_000))
def test_vmap_batch_permutation_equivariance(B, seed):
    """Feature: vector-map-head, Property 2 (batch permutation equivariance).

    地图头对 batch 内样本独立处理（无 batch 间耦合）。令 out = head(bev)，
    out_perm = head(bev[perm])，则必有 out_perm == out[perm]（逐键 allclose）。
    同一 query 的 pts/cls/score/query_emb 在同一置换下一同搬移，顺序对齐。

    **Validates: 正确性属性 2；需求 4.4, 4.3**
    """
    head = _make_vmap_head().eval()
    gen = torch.Generator().manual_seed(seed)
    bev = torch.randn(B, _VD, _VS, _VS, generator=gen)
    perm = torch.randperm(B, generator=gen)
    assume(not torch.equal(perm, torch.arange(B)))      # 跳过恒等置换（平凡）

    with torch.no_grad():
        out = head(bev)
        out_perm = head(bev[perm])

    # 每个带 batch 维的键：打乱输入 == 打乱输出（置换等变），四键同一 perm 对齐。
    for k in ("pts", "cls", "score", "query_emb"):
        assert torch.allclose(out_perm[k], out[k][perm], atol=1e-5), \
            f"{k} 不满足 batch 置换等变"
    # anchor_xy 与 batch 无关，恒定不变
    assert torch.allclose(out_perm["anchor_xy"], out["anchor_xy"], atol=1e-6)


def test_vmap_query_order_alignment():
    """同一 query 的 pts/cls/score/query_emb 顺序一一对齐（显式检查，不依赖随机）。

    对单样本 batch 施加 batch 维置换是平凡的；query 维对齐则通过「置换 batch 后，
    每个 query 槽位的四个输出整体搬移」间接保证——这里额外显式断言：取某一 query 槽位，
    其在 out 与 out[perm] 中的 pts/cls/score/query_emb 必定同步搬移（同索引取出相等）。

    **Validates: 正确性属性 2；需求 4.3**
    """
    B = 3
    head = _make_vmap_head().eval()
    torch.manual_seed(7)
    bev = torch.randn(B, _VD, _VS, _VS)
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        out = head(bev)
        out_perm = head(bev[perm])
    # 对每个样本 b 与每个 query 槽位 m，四个输出都来自同一 (b,m)，故同步对齐。
    for b in range(B):
        src = int(perm[b])
        for m in range(head.n_query):
            assert torch.allclose(out_perm["pts"][b, m], out["pts"][src, m], atol=1e-5)
            assert torch.allclose(out_perm["cls"][b, m], out["cls"][src, m], atol=1e-5)
            assert torch.allclose(out_perm["score"][b, m], out["score"][src, m], atol=1e-5)
            assert torch.allclose(out_perm["query_emb"][b, m],
                                  out["query_emb"][src, m], atol=1e-5)


# ===========================================================================
# 任务 3.4：折线匹配与矢量地图损失属性测试（hypothesis）
#
# 针对 train.py 的 polyline_distance / match_polyline_to_gt / vector_map_loss，
# 验证五条正确性属性（镜像检测头 test_detection_properties 的匹配/损失风格）：
#   - 属性 3 匹配正确性：令部分预测折线精确=GT（且锚点就位），匹配必须把这些 GT 单射
#     指派到对应预测 query（仅来自有效 GT、每 query 单次指派）；锚点与 polyline_distance
#     两种基准都验证。
#   - 属性 4 损失零点：匹配 query 折线=GT、类别强命中、存在性对齐时，折线回归分量 ~0
#     （因梯度安全 sqrt-eps，用 atol 容差），总 vector_map_loss 由很小的 CE/BCE 主导。
#   - 属性 5 顺序鲁棒性：polyline_distance(a,reverse(a))~0；且 d(a,b)==d(a,reverse(b))。
#   - 属性 6 mask 单调性：追加 pad 无效折线（map_poly_mask=False）不改变有效折线上的 loss。
#   - 属性 8 空场景安全：M_gt=0 时 loss 有限、回归/类别贡献 0、反传无 NaN。
#
# 为提速用较小的 M/P/n_cls 构造假 vmap 与 GT（沿用上文地图头测试的小配置风格）。
# 注意 polyline_distance(a,a) 因 sqrt(.+eps) 约为 1e-8（非精确 0），"零"断言统一用 atol。
#
# 验证：正确性属性 3、4、5、6、8；需求 5.4, 5.5, 6.4, 6.5, 6.6, 6.7, 10.3。
# ===========================================================================

import torch.nn.functional as F  # noqa: E402

from train import (  # noqa: E402
    match_polyline_to_gt,
    polyline_distance,
    vector_map_loss,
)

# 匹配/损失测试用的小配置（远小于生产 M=48/P=20/n_cls=3，纯为跑测提速）
_LM = 10          # M（map query 数）
_LMAXMAP = 10     # MAX_MAP（GT 定长上限）
_LP = 6           # P（点/折线）
_LN_CLS = 3       # 类别数
# polyline_distance 的"零点"因梯度安全 sqrt(sq+1e-16) 实为 ~1e-8；统一用此 atol 判零。
_ZERO_ATOL = 1e-4


# ---------------------------------------------------------------------------
# 生成器：随机折线 / 分散质心的有效 GT
# ---------------------------------------------------------------------------

def _rand_polyline(P, gen):
    """随机一条 (P,2) 折线，坐标 ∈ [-1,1]（SCENE_SCALE 归一化尺度量级）。"""
    return (torch.rand(P, 2, generator=gen) - 0.5) * 2.0


def _spread_gt(M_gt, P, gen, n_cls=_LN_CLS, max_map=_LMAXMAP):
    """构造 M_gt 条质心两两远离的有效 GT 折线（pad 到 max_map）。

    质心用一维网格（间隔 10，远大于折线自身 ±1 的点抖动），保证"锚点落在某 GT 质心上"时
    最近的一定是该 GT 自己，避免贪心匹配歧义（镜像 detection 的 _random_gt 思路）。
    返回 (gt_pts (max_map,P,2), gt_cls (max_map,), gt_mask (max_map,), centroids (M_gt,2))。
    """
    gt_pts = torch.zeros(max_map, P, 2)
    gt_cls = torch.zeros(max_map, dtype=torch.long)
    gt_mask = torch.zeros(max_map, dtype=torch.bool)
    cx = torch.arange(M_gt, dtype=torch.float32) * 10.0 - (M_gt - 1) * 5.0  # 远离的 x
    cy = (torch.rand(M_gt, generator=gen) - 0.5) * 2.0
    centroids = torch.stack([cx, cy], dim=1)                 # (M_gt,2)
    for k in range(M_gt):
        local = (torch.rand(P, 2, generator=gen) - 0.5) * 2.0   # ±1 的局部点形
        local = local - local.mean(0, keepdim=True)            # 去均值 -> 质心=0
        gt_pts[k] = local + centroids[k]                        # 平移到目标质心
    if M_gt > 0:
        gt_cls[:M_gt] = torch.randint(0, n_cls, (M_gt,), generator=gen)
    gt_mask[:M_gt] = True
    return gt_pts, gt_cls, gt_mask, centroids


# ---------------------------------------------------------------------------
# 属性 5：顺序鲁棒性（polyline_distance 对 GT 翻转不变、翻转自身距离为 0）
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(P=st.integers(2, 20), seed=st.integers(0, 10_000))
def test_polyline_distance_order_robust(P, seed):
    """Feature: vector-map-head, Property 5 (order-robustness of polyline_distance).

    道路线无方向：polyline_distance(a, reverse(a)) ~0（翻转自身仍是同一几何）；
    且对任意 a,b 有 d(a,b)==d(a,reverse(b))（对 GT 点序翻转不变）。
    因梯度安全 sqrt(sq+eps)，"零"用 atol 判定（非精确 0）。

    **Validates: 正确性属性 5；需求 5.5, 6.5**
    """
    gen = torch.Generator().manual_seed(seed)
    a = _rand_polyline(P, gen)
    b = _rand_polyline(P, gen)

    # a 与其翻转的距离 ~0（min(正序,反序) 中反序对齐精确重合）
    d_self_rev = polyline_distance(a, a.flip(-2))
    assert d_self_rev.item() < _ZERO_ATOL, \
        f"d(a,reverse(a)) 非零：{d_self_rev.item():.3e}"

    # 对 GT 翻转不变：d(a,b) == d(a,reverse(b))（min 对称于翻转 GT）
    d_ab = polyline_distance(a, b)
    d_arevb = polyline_distance(a, b.flip(-2))
    assert torch.allclose(d_ab, d_arevb, atol=1e-6), \
        f"d(a,b)={d_ab.item():.6f} != d(a,reverse(b))={d_arevb.item():.6f}"


def test_polyline_distance_zero_on_equal():
    """polyline_distance(a,a) ~0（显式检查，不依赖随机）。

    因梯度安全 sqrt(sq+1e-16)，相等输入距离约为 1e-8（而非精确 0），用 atol 判零。

    **Validates: 正确性属性 5；需求 6.4**
    """
    a = torch.tensor([[0.0, 0.0], [1.0, 0.5], [2.0, -0.3], [3.0, 0.1]])
    d = polyline_distance(a, a)
    assert d.item() < _ZERO_ATOL
    assert torch.isfinite(d)


# ---------------------------------------------------------------------------
# 属性 3：匹配正确性（精确等于 GT 的预测折线必被单射指派到对应 GT）
# ---------------------------------------------------------------------------

@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(M_gt=st.integers(1, _LMAXMAP), seed=st.integers(0, 10_000))
def test_match_correctness_anchor(M_gt, seed):
    """Feature: vector-map-head, Property 3 (anchor-based exact-GT matching).

    锚点基准：令每条有效 GT 的锚点恰好落在该 GT 质心上（其余 query 锚点放远处），且该
    query 的预测折线精确=GT。匹配必须把每个 GT 单射指派到对应 query（仅来自有效 GT）。

    **Validates: 正确性属性 3；需求 5.4**
    """
    gen = torch.Generator().manual_seed(seed)
    M = _LM
    assume(M_gt <= M)
    gt_pts, gt_cls, gt_mask, centroids = _spread_gt(M_gt, _LP, gen)
    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()  # (M_gt,)

    # 选 M_gt 个 query 槽位，锚点钉到对应 GT 质心、预测折线精确拷贝 GT；其余 query 放远处。
    pred_pts = torch.full((M, _LP, 2), 1000.0)
    anchor = torch.full((M, 2), 1000.0)
    perm = torch.randperm(M, generator=gen)[:M_gt]
    expected = {}
    for k in range(M_gt):
        q = int(perm[k])
        g = int(valid_gt_idx[k])
        anchor[q] = centroids[k]                        # 锚点落在该 GT 质心
        pred_pts[q] = gt_pts[g]                          # 预测折线精确=GT
        expected[g] = q

    pred_idx, matched_gt_idx = match_polyline_to_gt(pred_pts, gt_pts, gt_mask,
                                                    anchor=anchor)
    assert pred_idx.numel() == M_gt and matched_gt_idx.numel() == M_gt
    # 单射：预测/GT 索引均不重复（每 query 单次指派）
    assert len(set(pred_idx.tolist())) == M_gt
    assert len(set(matched_gt_idx.tolist())) == M_gt
    got = {int(g): int(p) for g, p in zip(matched_gt_idx.tolist(), pred_idx.tolist())}
    assert got == expected, f"锚点匹配不符：got={got} expected={expected}"
    # 匹配到的 GT 必须都是有效 GT
    assert set(matched_gt_idx.tolist()).issubset(set(valid_gt_idx.tolist()))


@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(M_gt=st.integers(1, _LMAXMAP), seed=st.integers(0, 10_000))
def test_match_correctness_polyline_basis(M_gt, seed):
    """Feature: vector-map-head, Property 3 (polyline_distance-based matching).

    anchor=None 基准：直接用 polyline_distance 做距离矩阵。令 M_gt 个 query 的预测折线
    精确=各 GT（质心两两远离 -> 折线距离最近的一定是对应 GT），匹配必须单射指派对应 GT。

    **Validates: 正确性属性 3；需求 5.4**
    """
    gen = torch.Generator().manual_seed(seed)
    M = _LM
    assume(M_gt <= M)
    gt_pts, gt_cls, gt_mask, centroids = _spread_gt(M_gt, _LP, gen)
    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()

    # 背景 query 折线放到极远处（折线距离巨大），M_gt 个 query 精确拷贝对应 GT。
    pred_pts = torch.full((M, _LP, 2), 1000.0)
    perm = torch.randperm(M, generator=gen)[:M_gt]
    expected = {}
    for k in range(M_gt):
        q = int(perm[k])
        g = int(valid_gt_idx[k])
        pred_pts[q] = gt_pts[g]
        expected[g] = q

    pred_idx, matched_gt_idx = match_polyline_to_gt(pred_pts, gt_pts, gt_mask,
                                                    anchor=None)
    assert pred_idx.numel() == M_gt and matched_gt_idx.numel() == M_gt
    assert len(set(pred_idx.tolist())) == M_gt
    assert len(set(matched_gt_idx.tolist())) == M_gt
    got = {int(g): int(p) for g, p in zip(matched_gt_idx.tolist(), pred_idx.tolist())}
    assert got == expected, f"polyline 基准匹配不符：got={got} expected={expected}"
    assert set(matched_gt_idx.tolist()).issubset(set(valid_gt_idx.tolist()))


def test_match_empty_scene_vmap():
    """M_gt=0（无有效 GT）时匹配返回两个空张量（不报错）。

    **Validates: 正确性属性 3、8；需求 5.5, 6.7**
    """
    gen = torch.Generator().manual_seed(0)
    pred_pts = _rand_polyline(_LP, gen).unsqueeze(0).repeat(_LM, 1, 1)
    gt_pts = torch.zeros(_LMAXMAP, _LP, 2)
    gt_mask = torch.zeros(_LMAXMAP, dtype=torch.bool)
    pred_idx, gt_idx = match_polyline_to_gt(pred_pts, gt_pts, gt_mask)
    assert pred_idx.numel() == 0 and gt_idx.numel() == 0


# ---------------------------------------------------------------------------
# 属性 4：损失零点（匹配折线=GT、类别命中、存在性对齐 -> 回归分量 ~0，总 loss 很小）
# ---------------------------------------------------------------------------

@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(M_gt=st.integers(1, _LMAXMAP), seed=st.integers(0, 10_000))
def test_loss_zero_point(M_gt, seed):
    """Feature: vector-map-head, Property 4 (loss near zero at the GT).

    匹配 query 的折线精确=GT、类别 logit 强命中 GT 类、存在性 score 对匹配 query 置高、
    背景置低。此时：折线回归分量 ~0（因 sqrt-eps 用 atol）、类别 CE 与存在性 BCE 因大 logit
    间隔趋近 0。断言总 vector_map_loss 很小，且折线回归分量（polyline_distance 之和）~0。

    **Validates: 正确性属性 4；需求 6.4**
    """
    gen = torch.Generator().manual_seed(seed)
    M = _LM
    assume(M_gt <= M)
    gt_pts, gt_cls, gt_mask, centroids = _spread_gt(M_gt, _LP, gen)
    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()

    pred_pts = torch.full((1, M, _LP, 2), 1000.0)
    anchor = torch.full((M, 2), 1000.0)
    cls_logit = torch.zeros(1, M, _LN_CLS)
    score_logit = torch.full((1, M, 1), -20.0)             # 背景存在性置低
    perm = torch.randperm(M, generator=gen)[:M_gt]
    for k in range(M_gt):
        q = int(perm[k])
        g = int(valid_gt_idx[k])
        anchor[q] = centroids[k]
        pred_pts[0, q] = gt_pts[g]                          # 折线精确=GT -> 回归~0
        cls_logit[0, q, int(gt_cls[g])] = 30.0             # 类别强命中 -> CE≈0
        score_logit[0, q, 0] = 20.0                         # 匹配 query 存在性置高

    vmap = {"pts": pred_pts, "cls": cls_logit, "score": score_logit, "anchor_xy": anchor}
    loss = vector_map_loss(vmap, gt_pts.unsqueeze(0), gt_cls.unsqueeze(0),
                           gt_mask.unsqueeze(0))
    assert torch.isfinite(loss), "loss 含 NaN/Inf"
    assert loss.item() < 1e-3, f"损失零点不满足：loss={loss.item():.3e}"

    # 折线回归分量：显式复算匹配对的 polyline_distance 之和应 ~0（因 sqrt-eps 用 atol）
    pred_idx, gt_idx = match_polyline_to_gt(pred_pts[0], gt_pts, gt_mask, anchor=anchor)
    reg = polyline_distance(pred_pts[0, pred_idx], gt_pts[gt_idx]).sum()
    assert reg.item() < _ZERO_ATOL * max(M_gt, 1), \
        f"折线回归分量非零：{reg.item():.3e}"


# ---------------------------------------------------------------------------
# 属性 6：mask 单调性（追加 pad 无效折线不改变有效折线上的 loss）
# ---------------------------------------------------------------------------

@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(M_gt=st.integers(1, _LMAXMAP - 2), seed=st.integers(0, 10_000))
def test_mask_monotonicity(M_gt, seed):
    """Feature: vector-map-head, Property 6 (invalid pad polylines don't change loss).

    同样的有效折线内容下，往 GT 里追加若干 map_poly_mask=False 的 pad 槽位（内容任意），
    vector_map_loss 必须不变（无效位被 mask 忽略）。随机固定、vmap 固定以保证可比。

    **Validates: 正确性属性 6；需求 6.6**
    """
    gen = torch.Generator().manual_seed(seed)
    M = _LM
    assume(M_gt <= M)
    gt_pts, gt_cls, gt_mask, centroids = _spread_gt(M_gt, _LP, gen, max_map=_LMAXMAP)
    valid_gt_idx = torch.nonzero(gt_mask, as_tuple=False).flatten()

    # 固定一个 vmap（匹配 query 命中 GT，其余背景），两种 GT 下完全一致。
    pred_pts = torch.full((1, M, _LP, 2), 1000.0)
    anchor = torch.full((M, 2), 1000.0)
    cls_logit = torch.randn(1, M, _LN_CLS, generator=gen)
    score_logit = torch.full((1, M, 1), -5.0)
    perm = torch.randperm(M, generator=gen)[:M_gt]
    for k in range(M_gt):
        q = int(perm[k])
        g = int(valid_gt_idx[k])
        anchor[q] = centroids[k]
        pred_pts[0, q] = gt_pts[g] + (torch.rand(_LP, 2, generator=gen) - 0.5) * 0.2
        score_logit[0, q, 0] = 5.0
    vmap = {"pts": pred_pts, "cls": cls_logit, "score": score_logit, "anchor_xy": anchor}

    # 基线 loss（当前有效内容）
    loss_base = vector_map_loss(vmap, gt_pts.unsqueeze(0), gt_cls.unsqueeze(0),
                                gt_mask.unsqueeze(0))

    # 追加 pad 无效折线：扩大 MAX_MAP 维，mask 仍只有原 M_gt 位为 True，新增位内容任意、mask=False
    extra = 3
    big_pts = torch.cat([gt_pts, (torch.rand(extra, _LP, 2, generator=gen) - 0.5) * 2.0], 0)
    big_cls = torch.cat([gt_cls, torch.randint(0, _LN_CLS, (extra,), generator=gen)], 0)
    big_mask = torch.cat([gt_mask, torch.zeros(extra, dtype=torch.bool)], 0)
    loss_pad = vector_map_loss(vmap, big_pts.unsqueeze(0), big_cls.unsqueeze(0),
                               big_mask.unsqueeze(0))

    assert torch.allclose(loss_base, loss_pad, atol=1e-6), \
        f"mask 单调性不满足：base={loss_base.item():.6f} pad={loss_pad.item():.6f}"


# ---------------------------------------------------------------------------
# 属性 8：空场景安全（M_gt=0 -> loss 有限、回归/类别贡献 0、反传无 NaN）
# ---------------------------------------------------------------------------

@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(B=st.integers(1, 3), seed=st.integers(0, 10_000))
def test_empty_scene_loss_safe(B, seed):
    """Feature: vector-map-head, Property 8 (M_gt=0: finite loss, no NaN on backward).

    整个 batch 无有效 GT（gt_mask 全 False）时：vector_map_loss 返回有限标量（折线回归 /
    类别贡献为 0，仅存在性 BCE 对全背景生效），且反向传播不产生 NaN（梯度链路安全）。

    **Validates: 正确性属性 8；需求 6.7**
    """
    gen = torch.Generator().manual_seed(seed)
    M = _LM
    pred_pts = torch.randn(B, M, _LP, 2, generator=gen, requires_grad=True)
    cls_logit = torch.randn(B, M, _LN_CLS, generator=gen, requires_grad=True)
    score_logit = torch.randn(B, M, 1, generator=gen, requires_grad=True)
    vmap = {"pts": pred_pts, "cls": cls_logit, "score": score_logit}

    gt_pts = torch.zeros(B, _LMAXMAP, _LP, 2)
    gt_cls = torch.zeros(B, _LMAXMAP, dtype=torch.long)
    gt_mask = torch.zeros(B, _LMAXMAP, dtype=torch.bool)   # 全无效 -> M_gt=0

    loss = vector_map_loss(vmap, gt_pts, gt_cls, gt_mask)
    assert torch.isfinite(loss), "M_gt=0 时 loss 非有限"

    # 空场景：折线回归/类别贡献为 0，loss 应等于纯背景存在性 BCE
    expected = F.binary_cross_entropy_with_logits(
        score_logit.squeeze(-1), torch.zeros(B, M))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6), \
        f"空场景 loss 应等于纯 BCE：loss={loss.item():.6f} bce={expected.item():.6f}"

    loss.backward()
    # 反传无 NaN：pts 不参与 loss（回归贡献 0）梯度应为 0/None；cls 同理；score 应有有限梯度
    assert pred_pts.grad is None or torch.isfinite(pred_pts.grad).all()
    assert cls_logit.grad is None or torch.isfinite(cls_logit.grad).all()
    assert score_logit.grad is not None and torch.isfinite(score_logit.grad).all()
