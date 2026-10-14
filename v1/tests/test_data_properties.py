"""任务 2.5 数据层属性测试（hypothesis）。

属性 1 形状守恒：Sample 各字段形状为 (Th,4)/(N,Th,C)/(N,)/(N,2)/(Tf,2)/(Tf,)。
属性 6 ego 原点性：ego_hist[-1] 位置分量范数 < ε。

两类输入：
- 真实缓存 data/cache_mini.pt 中随机抽取的样本（缓存缺失则跳过）；
- 合成输入：随机全局位姿 + 随机 agent 数（含 >N 截断与 0 个）+ 随机未来长度 0..Tf，
  经 _global_to_ego_xy（ego 坐标系转换）与 raw_to_sample（标准化/padding）得到 Sample。
"""

import os
import sys

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from data import (_EGO_ORIGIN_EPS, NanoFSDDataset, RawSample,  # noqa: E402
                  _global_to_ego_xy, raw_to_sample)

CFG = Config()
N, TH, TF, C = CFG.n_agents, CFG.hist_steps, CFG.fut_steps, CFG.agent_feat_dim
CACHE = "data/cache_mini.pt"
DT = 0.5


def assert_shapes(s) -> None:
    """属性 1：各字段形状与 dtype 符合契约。"""
    assert s.ego_hist.shape == (TH, 4)
    assert s.agents.shape == (N, TH, C)
    assert s.agent_mask.shape == (N,) and s.agent_mask.dtype == torch.bool
    assert s.agent_gt.shape == (N, 2)
    assert s.traj_gt.shape == (TF, 2)
    assert s.traj_mask.shape == (TF,) and s.traj_mask.dtype == torch.bool


def assert_ego_origin(s) -> None:
    """属性 6：当前帧 ego 位置接近原点。"""
    assert torch.linalg.norm(s.ego_hist[-1, :2]) < _EGO_ORIGIN_EPS


# ---------------------------------------------------------------------------
# 真实缓存样本
# ---------------------------------------------------------------------------

_DS = NanoFSDDataset(CACHE) if os.path.exists(CACHE) else None


@pytest.mark.skipif(_DS is None, reason="缓存不存在")
@settings(max_examples=50, deadline=None)
@given(data=st.data())
def test_cached_samples_shape_and_ego_origin(data):
    """Feature: nano-fsd, Property 1 & 6 (cached samples).

    **Validates: 正确性属性 1、6**
    """
    idx = data.draw(st.integers(0, len(_DS) - 1), label="idx")
    s = _DS[idx]
    assert_shapes(s)
    assert_ego_origin(s)


# ---------------------------------------------------------------------------
# 合成输入：全局位姿 -> ego 坐标系 -> raw_to_sample
# ---------------------------------------------------------------------------

def _global_from_ego(tx: float, ty: float, yaw: float) -> np.ndarray:
    """构造 ego->global 4x4 齐次矩阵（绕 z 轴旋转 yaw，平移 (tx, ty)）。"""
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:2, 3] = [tx, ty]
    return m


@st.composite
def raw_samples(draw):
    """随机全局场景 -> ego 系 RawSample（agent 数 0..N+16，未来长度 0..Tf）。"""
    tx = draw(st.floats(-2000, 2000), label="tx")
    ty = draw(st.floats(-2000, 2000), label="ty")
    yaw = draw(st.floats(-np.pi, np.pi), label="yaw")
    speed = draw(st.floats(0, 20), label="speed")          # m/s，单步 ≤ 10m
    n = draw(st.integers(0, N + 16), label="n_agents")
    tf = draw(st.integers(0, TF), label="tf")
    seed = draw(st.integers(0, 2**31 - 1), label="seed")
    rng = np.random.default_rng(seed)

    g_from_e = _global_from_ego(tx, ty, yaw)
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    # 自车沿当前朝向匀速行驶；时间 t∈[-(Th-1), tf]，t=0 为当前帧
    t_all = np.arange(-(TH - 1), tf + 1, dtype=np.float64)
    ego_global = np.array([tx, ty]) + (speed * DT * t_all)[:, None] * heading
    ego_xy = _global_to_ego_xy(ego_global, g_from_e)        # (Th+tf, 2)
    hist_xy = ego_xy[:TH]
    vel = np.zeros_like(hist_xy)
    vel[1:] = (hist_xy[1:] - hist_xy[:-1]) / DT
    vel[0] = vel[1] if TH >= 2 else 0.0
    ego_hist = np.concatenate([hist_xy, vel], 1).astype(np.float32)
    traj_gt = ego_xy[TH:].astype(np.float32).reshape(tf, 2)

    # 周围 agent：全局位置随机分布在自车附近，转到 ego 系
    agents = np.zeros((n, TH, C), dtype=np.float32)
    agent_gt = np.zeros((n, 2), dtype=np.float32)
    if n > 0:
        a_global = np.array([tx, ty]) + rng.uniform(-60, 60, (n * TH, 2))
        a_ego = _global_to_ego_xy(a_global, g_from_e).reshape(n, TH, 2)
        ang = rng.uniform(-np.pi, np.pi, (n, TH))
        agents[:, :, 0:2] = a_ego
        agents[:, :, 2:4] = rng.uniform(-10, 10, (n, TH, 2))
        agents[:, :, 4] = np.cos(ang)
        agents[:, :, 5] = np.sin(ang)
        agents[np.arange(n), :, 6 + rng.integers(0, 3, n)] = 1.0
        agent_gt = agents[:, -1, :2].copy()
    return RawSample(ego_hist, agents, agent_gt, traj_gt, sample_token=f"syn{seed}")


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(raw=raw_samples())
def test_synthetic_samples_shape_and_ego_origin(raw):
    """Feature: nano-fsd, Property 1 & 6 (synthetic raw -> Sample).

    **Validates: 正确性属性 1、6**
    """
    s = raw_to_sample(raw, CFG)
    n, tf = raw.agents.shape[0], raw.traj_gt.shape[0]
    if n == 0:
        assert s is None          # 无有效 agent 的样本被过滤
        return
    assert_shapes(s)
    assert_ego_origin(s)
    # padding / 截断后的 mask 计数与输入一致
    assert int(s.agent_mask.sum()) == min(n, N)
    assert int(s.traj_mask.sum()) == tf
