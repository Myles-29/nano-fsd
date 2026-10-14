"""v2 数据管线：基于元数据缓存的在线读图 Dataset + DataLoader。

Dataset 每个样本：读 6 路图(在线、缩放) + 带出相机参数 K/R/t + 真值轨迹。
相机投影几何(sample_uv)不在这里算——放到训练循环里用 K/R/t 在 GPU 上批量现算，
最快且省 CPU（见 geometry_from_batch）。

轨迹按 SCENE_SCALE=50 标准化，和 v1/overfit 一致。
"""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from bev_encoder import build_sampling_grid

SCENE_SCALE = 50.0
BEV_RANGE, BEV_SIZE = 50.0, 100
# 高度锚点：覆盖地面(0)到卡车顶(~3m)，让 BEV 采样能命中车身主体（车窗/车顶），
# 改善占据预测（之前最高只有 0.5m，采不到车身，占据差）。
HEIGHTS = [-1.0, 0.0, 1.0, 2.0, 3.0]


class NuScenesBEVDataset(Dataset):
    """从 meta 缓存索引样本；__getitem__ 在线读 6 路图并组装一个样本的张量。"""

    def __init__(self, meta_path: str):
        blob = torch.load(meta_path, weights_only=False)
        self.records = blob["records"]
        self.img_hw = blob["img_hw"]       # (H,W) 缩放后
        self.fut_steps = blob["fut_steps"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        rec = self.records[idx]
        H, W = self.img_hw
        imgs = []
        for p in rec["img_paths"]:
            img = Image.open(p).convert("RGB").resize((W, H))
            arr = np.asarray(img, dtype=np.float32).copy() / 255.0   # (H,W,3) [0,1]
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))      # (3,H,W)
        images = torch.stack(imgs)                                   # (6,3,H,W)

        traj = torch.from_numpy(rec["traj"]).float() / SCENE_SCALE   # (Tf,2) 标准化
        out = {
            "images": images,
            "K": torch.from_numpy(rec["K"]).float(),     # (6,3,3)
            "R": torch.from_numpy(rec["R"]).float(),     # (6,3,3)
            "t": torch.from_numpy(rec["t"]).float(),     # (6,3)
            "traj_gt": traj,
            "traj_mask": torch.from_numpy(rec["traj_mask"]),
        }
        if "occ" in rec:
            out["occ"] = torch.from_numpy(rec["occ"]).float()   # (S,S) 占据真值 0/1
        if "drivable" in rec:
            out["drivable"] = torch.from_numpy(rec["drivable"]).float()  # (S,S) 可行驶区 0/1
        if "command" in rec:
            out["command"] = torch.tensor(rec["command"], dtype=torch.long)  # 0/1/2
        return out


def collate(batch: list[dict]) -> dict:
    """堆叠成 batch（batch 维在前）。"""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


@torch.no_grad()
def geometry_from_batch(K: Tensor, R: Tensor, t: Tensor, img_hw,
                        device) -> tuple[Tensor, Tensor]:
    """由一个 batch 的相机参数，批量算 BEV 采样网格（在 device 上，训练循环里调）。

    输入 K/R/t: (B,6,3,3)/(B,6,3). 输出 sample_uv (B,S*S,6,Z,2), sample_valid (B,S*S,6,Z)。
    """
    B = K.shape[0]
    xs = torch.linspace(-BEV_RANGE, BEV_RANGE, BEV_SIZE, device=device)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    bev_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)   # (Q,2)
    heights = torch.tensor(HEIGHTS, device=device)

    uvs, vals = [], []
    for b in range(B):
        uv, val = build_sampling_grid(
            bev_xy, heights, K[b].to(device), R[b].to(device), t[b].to(device),
            (7, 13), img_hw)
        uvs.append(uv)
        vals.append(val)
    return torch.stack(uvs), torch.stack(vals)


def make_loaders(meta_path: str, batch_size: int = 2, val_ratio: float = 0.2,
                 seed: int = 0, num_workers: int = 6):
    """train/val DataLoader（固定 seed 划分，可复现）。"""
    ds = NuScenesBEVDataset(meta_path)
    n = len(ds)
    n_val = max(1, int(n * val_ratio))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, collate_fn=collate,
                       pin_memory=True, drop_last=True, persistent_workers=num_workers > 0)
    val = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, collate_fn=collate,
                     pin_memory=True, persistent_workers=num_workers > 0)
    return train, val, (len(train_idx), len(val_idx))


if __name__ == "__main__":
    # 自测：读 meta 缓存，取一个 batch，验证形状 + 几何可见占比。
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="cache/meta_2000.pt")
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, (ntr, nva) = make_loaders(args.meta, batch_size=2, num_workers=0)
    print(f"[ds] train={ntr}  val={nva}")
    batch = next(iter(train))
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    uv, valid = geometry_from_batch(batch["K"], batch["R"], batch["t"],
                                    (224, 400), dev)
    print(f"  sample_uv: {tuple(uv.shape)}  可见占比={valid.float().mean():.3f}")
