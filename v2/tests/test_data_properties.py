"""任务 1.4 数据层属性测试（hypothesis）。

针对 `dataset.py` 的 `NuScenesBEVDataset.__getitem__` 输出，验证两条正确性属性：

- **属性 1 形状守恒**：新增 agent 字段形状恒为
  `agent_box (50,5)`、`agent_cls (50,)`、`agent_mask (50,)`、
  `agent_fut (50,Tf,2)`、`agent_fut_mask (50,Tf)`；dtype 符合契约。
- **属性 6 空场景安全**：无 agent 样本 `agent_mask` 全 False，读取不报错。

两类输入：
- 带 agent 字段的合成缓存 record（agent 数 0..MAX_GT+8，含 0 个的空场景）；
- 缺 agent 字段的旧缓存 record（向后兼容路径：填零 + mask 全 False）。

验证：正确性属性 1、6；需求 1.6, 1.7。
"""

import os
import sys

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MAX_GT, SCENE_SCALE, NuScenesBEVDataset  # noqa: E402

# 测试用小图尺寸（在线读图走 PIL，用极小图省时省内存）
IMG_H, IMG_W = 8, 12
N_CAM = 6
TF = 12          # 未来步数（fut_steps），与 build_meta_cache 默认一致


# ---------------------------------------------------------------------------
# 构造临时缓存 + 极小 PNG 图，使 Dataset 能在线读图并组装样本
# ---------------------------------------------------------------------------

def _write_dummy_images(tmp_dir, n: int) -> list[str]:
    """写 n 张极小 PNG，返回路径列表（供 record['img_paths'] 用）。"""
    from PIL import Image

    paths = []
    for i in range(n):
        arr = np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
        p = os.path.join(tmp_dir, f"cam_{i}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _base_record(img_paths: list[str]) -> dict:
    """构造一条不含 agent 字段的最小 record（相机参数 / 轨迹等必备字段）。"""
    return {
        "img_paths": img_paths,
        "K": np.random.randn(N_CAM, 3, 3).astype(np.float32),
        "R": np.random.randn(N_CAM, 3, 3).astype(np.float32),
        "t": np.random.randn(N_CAM, 3).astype(np.float32),
        "traj": np.random.randn(TF, 2).astype(np.float32),
        "traj_mask": np.ones(TF, dtype=bool),
    }


def _add_agent_fields(rec: dict, n_agent: int) -> dict:
    """给 record 加入 agent 字段，前 n_agent 个有效（截断到 MAX_GT）。"""
    m = min(max(n_agent, 0), MAX_GT)

    agent_box = np.zeros((MAX_GT, 5), dtype=np.float32)
    agent_cls = np.zeros((MAX_GT,), dtype=np.int64)
    agent_mask = np.zeros((MAX_GT,), dtype=bool)
    agent_fut = np.zeros((MAX_GT, TF, 2), dtype=np.float32)
    agent_fut_mask = np.zeros((MAX_GT, TF), dtype=bool)

    if m > 0:
        # [x, y, yaw, w, l]；x,y 用较大范围以便验证归一化不改变形状
        agent_box[:m, 0:2] = np.random.uniform(-80, 80, (m, 2)).astype(np.float32)
        agent_box[:m, 2] = np.random.uniform(-np.pi, np.pi, m).astype(np.float32)
        agent_box[:m, 3:5] = np.random.uniform(1.0, 5.0, (m, 2)).astype(np.float32)
        agent_cls[:m] = np.random.randint(0, 2, m).astype(np.int64)
        agent_mask[:m] = True
        agent_fut[:m] = np.random.uniform(-30, 30, (m, TF, 2)).astype(np.float32)
        # 未来 mask 随机前缀有效（模拟 instance 中途消失）
        for k in range(m):
            valid_steps = np.random.randint(0, TF + 1)
            agent_fut_mask[k, :valid_steps] = True

    rec["agent_box"] = agent_box
    rec["agent_cls"] = agent_cls
    rec["agent_mask"] = agent_mask
    rec["agent_fut"] = agent_fut
    rec["agent_fut_mask"] = agent_fut_mask
    return rec


def _make_dataset(tmp_dir, records: list[dict]) -> NuScenesBEVDataset:
    """把 records 写成临时缓存 .pt 并构造 Dataset。"""
    img_paths = _write_dummy_images(tmp_dir, N_CAM)
    for rec in records:
        rec["img_paths"] = img_paths
    blob = {"records": records, "img_hw": (IMG_H, IMG_W), "fut_steps": TF}
    meta_path = os.path.join(tmp_dir, "meta_test.pt")
    torch.save(blob, meta_path)
    return NuScenesBEVDataset(meta_path)


# ---------------------------------------------------------------------------
# 共用断言
# ---------------------------------------------------------------------------

def assert_agent_shapes(out: dict) -> None:
    """属性 1：各 agent 字段形状与 dtype 符合契约。"""
    assert out["agent_box"].shape == (MAX_GT, 5)
    assert out["agent_box"].dtype == torch.float32
    assert out["agent_cls"].shape == (MAX_GT,)
    assert out["agent_cls"].dtype == torch.long
    assert out["agent_mask"].shape == (MAX_GT,)
    assert out["agent_mask"].dtype == torch.bool
    assert out["agent_fut"].shape == (MAX_GT, TF, 2)
    assert out["agent_fut"].dtype == torch.float32
    assert out["agent_fut_mask"].shape == (MAX_GT, TF)
    assert out["agent_fut_mask"].dtype == torch.bool


# ---------------------------------------------------------------------------
# 属性 1：形状守恒（带 agent 字段的新缓存，agent 数随机含 0）
# ---------------------------------------------------------------------------

@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow,
                                 HealthCheck.function_scoped_fixture])
@given(n_agent=st.integers(0, MAX_GT + 8))
def test_shape_conservation_with_agent_fields(tmp_path_factory, n_agent):
    """Feature: nano-fsd-v2, Property 1 (new cache with agent fields).

    **Validates: 正确性属性 1；需求 1.7**
    """
    tmp_dir = str(tmp_path_factory.mktemp("cache_with_agents"))
    rec = _add_agent_fields(_base_record([]), n_agent)
    ds = _make_dataset(tmp_dir, [rec])
    out = ds[0]
    assert_agent_shapes(out)
    # 有效位数量 = min(n_agent, MAX_GT)
    assert int(out["agent_mask"].sum()) == min(n_agent, MAX_GT)


# ---------------------------------------------------------------------------
# 属性 1：形状守恒（旧缓存缺 agent 字段 -> 向后兼容填零）
# ---------------------------------------------------------------------------

def test_shape_conservation_legacy_cache(tmp_path):
    """Feature: nano-fsd-v2, Property 1 (legacy cache without agent fields).

    **Validates: 正确性属性 1；需求 1.7**
    """
    rec = _base_record([])
    ds = _make_dataset(str(tmp_path), [rec])
    out = ds[0]
    assert_agent_shapes(out)
    # 旧缓存：mask 全 False
    assert int(out["agent_mask"].sum()) == 0
    assert not out["agent_fut_mask"].any()


# ---------------------------------------------------------------------------
# 属性 6：空场景安全（无 agent 样本 agent_mask 全 False，读取不报错）
# ---------------------------------------------------------------------------

def test_empty_scene_safety_with_fields(tmp_path):
    """Feature: nano-fsd-v2, Property 6 (empty scene, agent fields present).

    **Validates: 正确性属性 6；需求 1.6**
    """
    rec = _add_agent_fields(_base_record([]), 0)   # 0 个 agent
    ds = _make_dataset(str(tmp_path), [rec])
    out = ds[0]                                     # 读取不报错
    assert_agent_shapes(out)
    assert not out["agent_mask"].any()              # 全 False
    assert not out["agent_fut_mask"].any()


def test_empty_scene_safety_legacy(tmp_path):
    """Feature: nano-fsd-v2, Property 6 (empty scene, legacy cache).

    **Validates: 正确性属性 6；需求 1.6, 1.7**
    """
    rec = _base_record([])                          # 完全无 agent 字段
    ds = _make_dataset(str(tmp_path), [rec])
    out = ds[0]                                     # 读取不报错
    assert_agent_shapes(out)
    assert not out["agent_mask"].any()


# ---------------------------------------------------------------------------
# 归一化不破坏形状：x,y 列被 SCENE_SCALE 归一化，形状仍 (MAX_GT,5)
# ---------------------------------------------------------------------------

@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(n_agent=st.integers(1, MAX_GT))
def test_normalization_preserves_shape(tmp_path_factory, n_agent):
    """Feature: nano-fsd-v2, Property 1 (normalization keeps shape).

    **Validates: 正确性属性 1；需求 1.7**
    """
    tmp_dir = str(tmp_path_factory.mktemp("cache_norm"))
    rec = _add_agent_fields(_base_record([]), n_agent)
    raw_xy = rec["agent_box"][:, 0:2].copy()        # 归一化前的原始 x,y
    ds = _make_dataset(tmp_dir, [rec])
    out = ds[0]
    assert_agent_shapes(out)
    # x,y 两列按 SCENE_SCALE 归一化（有效位上）
    m = min(n_agent, MAX_GT)
    got_xy = out["agent_box"][:m, 0:2].numpy()
    np.testing.assert_allclose(got_xy, raw_xy[:m] / SCENE_SCALE, rtol=1e-5, atol=1e-6)
