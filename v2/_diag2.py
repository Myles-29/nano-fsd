"""独立诊断：不靠"命中率假设"，直接比较道路栅格形状 vs 轨迹几何方向。

对一个轨迹明显非直的样本：
  - 道路栅格 road (S,S) 的占据重心、以及 road>0 像素在两个数组轴上的分布范围；
  - GT 轨迹在物理 (x,y) 上的位移向量（主方向）。
分别打印，人工判断道路沿数组哪个轴延展、轨迹沿物理哪个方向，推出正确映射。
"""
import numpy as np
from dataset import NuScenesBEVDataset, BEV_RANGE, BEV_SIZE
from visualize import val_indices

ds = NuScenesBEVDataset("cache/meta_2000_agent.pt")
vidx = val_indices(len(ds), 0.2, 0)
S = BEV_SIZE

# 选一个轨迹位移大的样本，方向更明显
best = None
for idx in vidx[:200]:
    rec = ds.records[idx]
    traj = rec["traj"][rec["traj_mask"]]
    if len(traj) < 3:
        continue
    disp = traj[-1] - traj[0]
    mag = np.hypot(*disp)
    if best is None or mag > best[1]:
        best = (idx, mag, disp, traj, rec)

idx, mag, disp, traj, rec = best
road = np.asarray(rec["drivable"], dtype=np.float32)
print(f"样本 idx={idx}  轨迹总位移={mag:.1f} m")
print(f"轨迹物理位移向量 (dx前, dy左) = ({disp[0]:.1f}, {disp[1]:.1f})")
print(f"  -> 轨迹主要沿 {'x前后' if abs(disp[0])>abs(disp[1]) else 'y左右'} 方向")

ys, xs = np.where(road > 0.5)   # 注意: np.where 返回 (dim0_idx, dim1_idx)
if len(ys):
    d0_span = ys.max() - ys.min()   # 第0维(行)跨度
    d1_span = xs.max() - xs.min()   # 第1维(列)跨度
    print(f"道路像素: 第0维(行)跨度={d0_span}  第1维(列)跨度={d1_span}")
    print(f"  -> 道路主要沿数组 {'第0维(行)' if d0_span>d1_span else '第1维(列)'} 延展")

# 轨迹端点在物理->数组索引（两种候选），看落点
print("轨迹端点物理坐标与候选数组索引:")
for x, y in [traj[0], traj[-1]]:
    xi = int((x + BEV_RANGE) / (2*BEV_RANGE) * S)
    yi = int((y + BEV_RANGE) / (2*BEV_RANGE) * S)
    print(f"  phys(x={x:.1f},y={y:.1f}) -> xi={xi}, yi={yi}; "
          f"road[xi,yi]={road[xi,yi] if 0<=xi<S and 0<=yi<S else 'oob'}, "
          f"road[yi,xi]={road[yi,xi] if 0<=xi<S and 0<=yi<S else 'oob'}")
