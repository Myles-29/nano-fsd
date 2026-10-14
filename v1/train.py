"""train.py — 分阶段训练入口（阶段一联合训练 / 阶段二冻结感知栈）。

骨架文件：planning_loss、train_stage1、train_stage2 及命令行入口
在后续任务（4.2、6、7）中逐步实现。

本文件同时存放感知头与规划头共用的损失函数（masked_mse 等）。
记号: B=batch, N=agent 上限, Tf=未来步数, K=每个元素的输出维（如位置 2 维）。
"""

import torch
from torch import Tensor

from data import denormalize, make_loader, make_train_val_loaders
from model import NanoFSD


def masked_mse(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """带 mask 的均方误差（感知头与规划头共用的核心损失）。

    输入:
      pred (..., K)  预测值
      gt   (..., K)  真值，形状与 pred 相同
      mask (...)     bool，True=有效元素，False=padding（忽略）
    输出:
      标量 loss ≥ 0

    不变式: 分母为有效元素数，避免被 padding 稀释；用 clamp(min=1) 防除零，
    因此 mask 全 False（空场景）时安全返回 0。
    """
    err = ((pred - gt) ** 2).sum(dim=-1)        # (...,) 每个元素的平方误差
    err = err * mask.float()                     # 屏蔽 padding 位
    return err.sum() / mask.float().sum().clamp(min=1)


def planning_loss(traj_pred: Tensor, traj_gt: Tensor, traj_mask: Tensor) -> Tensor:
    """规划头损失：仅在有效未来步上计算的带 mask 均方误差。

    输入:
      traj_pred (B, Tf, 2)  预测的自车未来轨迹（ego 坐标系）
      traj_gt   (B, Tf, 2)  真值未来轨迹，形状与 traj_pred 相同
      traj_mask (B, Tf)     bool，True=有效时间步，False=padding（场景末尾不足 Tf）
    输出:
      标量 loss ≥ 0

    后置条件:
      - 返回标量 loss ≥ 0
      - 当 traj_pred == traj_gt（在有效位）时 loss == 0
      - 仅在 traj_mask == True 的时间步上计算，padding 未来步被忽略（场景 3）

    复用 masked_mse：对 (B, Tf, 2) 在最后一维求平方误差，按 (B, Tf) 的 mask 屏蔽。
    """
    return masked_mse(traj_pred, traj_gt, traj_mask)


def _resolve_device(device: str) -> str:
    """安全解析设备：cfg.device 默认 'cuda'，但无 GPU 时回退到 'cpu'。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _with_suffix(path: str, suffix: str) -> str:
    """在文件名（扩展名之前）插入后缀：checkpoint_stage1.pt -> checkpoint_stage1_best.pt。"""
    import os

    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _set_seed(seed: int) -> None:
    """固定随机种子，让训练尽量可复现（权重初始化、shuffle、dropout 等）。"""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_tag(cache: str) -> str:
    """从缓存路径猜一个简短的数据集标识，用于实验目录命名（trainval / mini / 文件名）。"""
    import os

    name = os.path.basename(cache).lower()
    if "trainval" in name:
        return "trainval"
    if "mini" in name:
        return "mini"
    # 回退：用去扩展名的文件名（去掉常见前缀 cache_）
    stem = os.path.splitext(os.path.basename(cache))[0]
    return stem.replace("cache_", "") or "data"


def _make_run_dir(stage: str, cfg, epochs: int, seed: int, cache: str) -> str:
    """生成实验目录名：runs/{时间戳}_{阶段}_{数据集}_ep{E}_lr{LR}_bs{BS}_seed{S}。

    参数总结直接编进目录名，便于一眼区分不同实验、横向对比。
    """
    import os
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds = _dataset_tag(cache)
    # lr 用紧凑写法（去掉末尾多余的 0）
    lr_str = f"{cfg.lr:g}"
    name = (f"{stamp}_{stage}_{ds}_ep{epochs}_lr{lr_str}_"
            f"bs{cfg.batch_size}_seed{seed}")
    return os.path.join("runs", name)


class _Tee:
    """把写入同时转发到多个流（用于让 stdout 既打印到终端又落盘到 train.log）。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


class _log_to_dir:
    """上下文管理器：进入时把 stdout/stderr 同时 tee 到 run_dir/train.log，退出复原。

    用法:
        with _log_to_dir(run_dir):
            ...  # 期间所有 print 既显示在终端，也写入 run_dir/train.log
    """

    def __init__(self, run_dir: str, filename: str = "train.log"):
        import os

        os.makedirs(run_dir, exist_ok=True)
        self._path = os.path.join(run_dir, filename)
        self._fh = None
        self._old_out = None
        self._old_err = None

    def __enter__(self):
        import sys

        self._fh = open(self._path, "a", buffering=1)  # 行缓冲，实时落盘
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._old_out, self._fh)
        sys.stderr = _Tee(self._old_err, self._fh)
        return self

    def __exit__(self, *exc):
        import sys

        sys.stdout, sys.stderr = self._old_out, self._old_err
        if self._fh is not None:
            self._fh.close()
        return False


def _auto_visualize(task: str, ckpt_path: str, cache: str, cfg, out_dir: str,
                    val_ratio: float = 0.2, seed: int = 0, k: int = 10) -> None:
    """训练结束后自动用 best 权重出可视化图到实验目录（规划/感知）。

    task="planning" 画规划轨迹图；task="perception" 画感知位置图。只从验证集选样
    （传入与训练一致的 val_ratio/seed），保证展示的是模型未训过的样本。viz 在此惰性
    导入，避免循环依赖；可视化失败不影响训练结果，只打印告警。
    """
    try:
        import viz
    except Exception as e:  # noqa: BLE001
        print(f"[viz] 跳过自动可视化（导入 viz 失败）: {e}")
        return
    try:
        if task == "perception":
            viz.run_perception_from_checkpoint(
                ckpt_path=ckpt_path, cache=cache, cfg=cfg, out_dir=out_dir,
                k=k, val_ratio=val_ratio, seed=seed)
        else:
            viz.run_from_checkpoint(
                ckpt_path=ckpt_path, cache=cache, cfg=cfg, out_dir=out_dir,
                k=k, val_ratio=val_ratio, seed=seed)
    except Exception as e:  # noqa: BLE001
        print(f"[viz] 自动可视化失败（不影响训练产物）: {e}")


@torch.no_grad()
def evaluate_perception(model: NanoFSD, loader, device: str) -> dict:
    """在给定 loader 上评估感知头：返回平均 masked_mse 与平均位移误差 ADE（米）。

    - loss: 与训练同口径的 masked_mse（标准化单位下），按 batch 的有效元素数加权平均。
    - ade_m: 仅在有效 agent 上，预测与真值相对位置的欧氏距离，denormalize 回米后求平均。
      ADE 比纯 loss 更直观——它直接告诉你"平均差了几米"。

    loader 为空（无样本）时返回 nan，避免除零。
    """
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    weight_sum = 0.0          # 有效元素数，用于按样本量加权平均 loss
    dist_sum = 0.0            # 有效 agent 的米制距离累加
    valid_count = 0           # 有效 agent 计数
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.ego_hist, batch.agents, batch.agent_mask,
                     mode="perception")                      # (B, N, 2)
        mask = batch.agent_mask.bool()                       # (B, N)
        n_valid = int(mask.sum())
        if n_valid == 0:
            continue
        # loss：按有效元素数加权，口径与 masked_mse 一致
        loss_sum += masked_mse(pred, batch.agent_gt, mask).item() * n_valid
        weight_sum += n_valid
        # ADE（米）：denormalize 后算欧氏距离
        err_m = denormalize(pred) - denormalize(batch.agent_gt)   # (B, N, 2) 米
        dist = torch.linalg.norm(err_m, dim=-1)                   # (B, N)
        dist_sum += float(dist[mask].sum())
        valid_count += n_valid

    if was_training:
        model.train()

    if weight_sum == 0:
        return {"loss": float("nan"), "ade_m": float("nan")}
    return {"loss": loss_sum / weight_sum, "ade_m": dist_sum / valid_count}


@torch.no_grad()
def evaluate_planning(model: NanoFSD, loader, device: str) -> dict:
    """在给定 loader 上评估规划头：返回平均 planning_loss 与平均位移误差 ADE（米）。

    - loss: 与训练同口径的 planning_loss（标准化单位），按有效未来步数加权平均。
    - ade_m: 仅在有效未来步（traj_mask）上，预测与真值轨迹的欧氏距离，denormalize
      回米后求平均。ADE 直接告诉你"未来轨迹平均差几米"，比纯 loss 直观。

    loader 为空时返回 nan。
    """
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    weight_sum = 0.0          # 有效未来步数，用于按样本量加权平均 loss
    dist_sum = 0.0            # 有效步的米制距离累加
    valid_count = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.ego_hist, batch.agents, batch.agent_mask,
                     mode="planning")                        # (B, Tf, 2)
        mask = batch.traj_mask.bool()                        # (B, Tf)
        n_valid = int(mask.sum())
        if n_valid == 0:
            continue
        loss_sum += planning_loss(pred, batch.traj_gt, mask).item() * n_valid
        weight_sum += n_valid
        err_m = denormalize(pred) - denormalize(batch.traj_gt)    # (B, Tf, 2) 米
        dist = torch.linalg.norm(err_m, dim=-1)                   # (B, Tf)
        dist_sum += float(dist[mask].sum())
        valid_count += n_valid

    if was_training:
        model.train()

    if weight_sum == 0:
        return {"loss": float("nan"), "ade_m": float("nan")}
    return {"loss": loss_sum / weight_sum, "ade_m": dist_sum / valid_count}


def train_stage1(cfg, epochs: int = 20, cache: str = "data/cache_mini.pt",
                 out_path: str = "checkpoint_stage1.pt",
                 verbose: bool = True, val_ratio: float = 0.2,
                 seed: int = 0, run_dir: str | None = None,
                 patience: int | None = None) -> tuple[NanoFSD, dict]:
    """阶段一：联合训练 backbone + 感知头（设计引用：算法 A、阶段一时序图）。

    这是一个严肃的训练任务（非 demo）：固定随机种子、划分 train/val、每个 epoch
    在验证集上评估感知 loss 与位移误差 ADE（米）、按 val 指标保存最优权重、可选
    early stopping，并把配置与训练曲线落盘到 run_dir 以便复现与横向对比。

    前置: data.py 能产出含 agent_gt 的 batch。
    后置:
      - out_path 保存最后一个 epoch 的权重（含 backbone + perception_head）。
      - 同目录下 <out_path 去扩展名>_best.pt 保存 val ADE 最优的权重。
      - run_dir（给定时）落盘 config.json 与 history.json。

    参数
    ----
    cfg: Config，使用其中的 lr / batch_size / device 等。
    epochs: 训练轮数。
    cache: 预处理缓存路径。
    out_path: 最终 checkpoint 保存路径（最优权重另存 *_best.pt）。
    verbose: True 时每个 epoch 打印 train/val 指标。
    val_ratio: 验证集占比（固定 seed 随机划分，可复现）。
    seed: 全局随机种子（权重初始化 / shuffle / 划分）。
    run_dir: 实验目录；给定时落盘 config.json 与 history.json。None 则不落盘。
    patience: early stopping 容忍的"val ADE 连续不改善"轮数；None 关闭。

    返回
    ----
    (model, history)：训练后的模型与训练历史 dict，键包含
      train_loss(list)、val_loss(list)、val_ade(list)、
      best_epoch(int)、best_val_ade(float)、best_path(str)。
    为向后兼容，history["train_loss"] 即旧版返回的逐 epoch 训练 loss 列表。
    """
    import json
    import os

    _set_seed(seed)
    device = _resolve_device(cfg.device)

    # 确保 checkpoint 的父目录存在（最优权重在第一个 epoch 就会写入）
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = NanoFSD(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)  # 全部参数都训
    train_loader, val_loader = make_train_val_loaders(
        cfg, cache=cache, val_ratio=val_ratio, seed=seed)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    best_path = _with_suffix(out_path, "_best")
    if verbose:
        print(f"[stage1] device={device}  seed={seed}  "
              f"train={n_train}  val={n_val}  lr={cfg.lr}  bs={cfg.batch_size}")

    history = {"train_loss": [], "val_loss": [], "val_ade": []}
    best_val_ade = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch.ego_hist, batch.agents,
                         batch.agent_mask, mode="perception")   # (B, N, 2)
            loss = masked_mse(pred, batch.agent_gt, batch.agent_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss_sum += loss.item()
            n_batches += 1

        train_loss = epoch_loss_sum / max(n_batches, 1)
        val = evaluate_perception(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_ade"].append(val["ade_m"])

        improved = val["ade_m"] < best_val_ade
        if improved:
            best_val_ade = val["ade_m"]
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save({"model": model.state_dict()}, best_path)
        else:
            epochs_since_improve += 1

        if verbose:
            star = " *" if improved else ""
            print(f"[stage1] epoch {epoch + 1:>3}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val['loss']:.6f}  "
                  f"val_ADE={val['ade_m']:.3f} m{star}")

        if patience is not None and epochs_since_improve >= patience:
            if verbose:
                print(f"[stage1] early stop：val ADE 连续 {patience} 轮未改善")
            break

    torch.save({"model": model.state_dict()}, out_path)
    history["best_epoch"] = best_epoch
    history["best_val_ade"] = best_val_ade
    history["best_path"] = best_path

    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        from dataclasses import fields as _fields
        cfg_dump = {f.name: getattr(cfg, f.name) for f in _fields(cfg)}
        cfg_dump.update({"epochs": epochs, "val_ratio": val_ratio, "seed": seed,
                         "resolved_device": device})
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg_dump, f, indent=2, ensure_ascii=False)
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[stage1] 最终权重 -> {out_path}")
        print(f"[stage1] 最优权重 -> {best_path}  "
              f"(epoch {best_epoch}, val_ADE={best_val_ade:.3f} m)")
    return model, history


def _load_stage1_ckpt(path: str, device: str) -> dict:
    """安全加载 stage1 checkpoint；文件不存在时报错提示先跑阶段一（错误处理场景 1）。

    train_stage1 保存的是 {"model": state_dict}，内容仅为权重张量，优先用
    weights_only=True 加载（更安全）；若当前 torch 版本不支持该参数则回退。
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到阶段一 checkpoint：{path!r}。"
            f"请先运行 `python train.py --stage 1` 完成阶段一感知预训练后再跑阶段二。")

    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # 老版本 torch 无 weights_only 参数，回退到普通加载
        ckpt = torch.load(path, map_location=device)
    return ckpt


def _frozen_param_snapshot(model: NanoFSD) -> dict[str, Tensor]:
    """克隆 backbone + perception_head 当前权重，用于训练前后比对（证明冻结生效）。

    返回 {参数名: 该参数的 detach().clone()}，脱离计算图且独立于后续原地更新。
    """
    snap: dict[str, Tensor] = {}
    for name, p in model.backbone.named_parameters():
        snap[f"backbone.{name}"] = p.detach().clone()
    for name, p in model.perception_head.named_parameters():
        snap[f"perception_head.{name}"] = p.detach().clone()
    return snap


def _max_abs_diff_vs_snapshot(model: NanoFSD, snapshot: dict[str, Tensor]) -> float:
    """返回 backbone + perception_head 当前权重与快照之间的最大绝对差（标量 float）。

    冻结生效时该值应恒为 0.0（正确性属性 3：阶段二前后感知栈逐元素相等）。
    """
    max_diff = 0.0
    for name, p in model.backbone.named_parameters():
        d = (p.detach() - snapshot[f"backbone.{name}"]).abs().max().item()
        max_diff = max(max_diff, d)
    for name, p in model.perception_head.named_parameters():
        d = (p.detach() - snapshot[f"perception_head.{name}"]).abs().max().item()
        max_diff = max(max_diff, d)
    return max_diff


def train_stage2(cfg, epochs: int = 20, cache: str = "data/cache_mini.pt",
                 stage1_ckpt: str = "checkpoint_stage1.pt",
                 out_path: str = "checkpoint_stage2.pt",
                 verbose: bool = True, return_snapshots: bool = False,
                 val_ratio: float | None = None, seed: int = 0,
                 run_dir: str | None = None, patience: int | None = None,
                 return_history: bool = False):
    """阶段二：冻结感知栈，只训练规划头（设计引用：算法 B、错误处理场景 1/4、性能考量）。

    核心学习点「分阶段冻结训练」：加载阶段一的 backbone + 感知头权重并冻结，
    optimizer 只收集规划头参数，前向只走规划头、仅回传规划头梯度。

    前置: 存在 checkpoint_stage1.pt。
    后置: 保存 checkpoint_stage2.pt；backbone/感知头权重不变，仅 planning_head 更新。

    参数
    ----
    cfg: Config，使用其中的 lr / batch_size / device 等。
    epochs: 训练轮数（数据集很小，默认 20）。
    cache: 预处理缓存路径，传给 make_loader。
    stage1_ckpt: 阶段一 checkpoint 路径（含 backbone + perception_head 权重）。
    out_path: 阶段二 checkpoint 保存路径。
    verbose: True 时打印缺失/多余键与每个 epoch 的平均规划 loss。
    return_snapshots: False（默认）保持向后兼容，返回 (model, loss_history) 二元组；
        True 时额外在冻结后、训练前对 backbone + perception_head 权重做快照，
        训练后计算最大绝对差，返回 (model, loss_history, info) 三元组，其中 info
        含 trainable_count / frozen_count / max_abs_diff（供演示 demo_stage2 使用）。
    val_ratio: None（默认）走旧行为（全量训练、无验证集，供 demo 用）；给定 (0,1) 时
        划分 train/val，每个 epoch 评估规划 val loss 与规划 ADE（米），按 val ADE 保存
        最优权重 <out_path 去扩展名>_best.pt，并支持 early stopping。
    seed: 随机种子；建议与阶段一一致，使 val 划分相同（冻结 backbone 未在这些样本上训）。
    run_dir: 实验目录；给定时落盘 config.json 与 history.json。
    patience: early stopping 容忍的"val ADE 连续不改善"轮数；None 关闭。
    return_history: True 时返回 (model, history) —— history 为含 train_loss/val_loss/
        val_ade/best_epoch/best_val_ade/best_path 的 dict（正式训练入口用）。
        与 return_snapshots 互斥；两者都 False 时保持旧的 (model, loss_history) 返回。

    返回
    ----
    return_history=True:   (model, history)
    return_snapshots=True: (model, loss_history, info)
    否则:                   (model, loss_history)
    """
    import os

    # 设备安全解析：cfg.device 默认 "cuda"，无 GPU 时回退 "cpu"
    _set_seed(seed)
    device = _resolve_device(cfg.device)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = NanoFSD(cfg).to(device)

    # ---------- 任务 7.1：加载 stage1 权重并冻结感知栈 ----------
    # 加载 checkpoint（文件缺失时报错提示先跑阶段一，错误处理场景 1）
    ckpt = _load_stage1_ckpt(stage1_ckpt, device)

    # strict=False 加载：规划头是阶段二新初始化的，不在 stage1 权重里，属正常现象
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(f"[stage2] 从 {stage1_ckpt} 加载 stage1 权重（strict=False）")
        # 缺失键预期包含 planning_head.*（阶段二新初始化，不在 stage1 中）—— 正常
        print(f"[stage2] 缺失键 missing_keys ({len(missing_keys)} 个，"
              f"预期含 planning_head.*，为新初始化): {list(missing_keys)}")
        # 多余键预期为空（stage1 不应含 stage2 没有的键）
        print(f"[stage2] 多余键 unexpected_keys ({len(unexpected_keys)} 个): "
              f"{list(unexpected_keys)}")

    # ❄️ 冻结 backbone + 感知头，planning_head 保持可训练并置 eval()
    model.freeze_perception_stack()

    # optimizer 只收集 requires_grad=True 的参数（即规划头）
    trainable = [p for p in model.parameters() if p.requires_grad]
    # 自检：所有可训练参数都必须属于 planning_head（按参数对象 id 比对，最稳妥）
    planning_ids = {id(p) for p in model.planning_head.parameters()}
    trainable_ids = {id(p) for p in trainable}
    assert trainable_ids == planning_ids, (
        "冻结自检失败：可训练参数集合与 planning_head 参数集合不一致。"
        f" 可训练张量数={len(trainable_ids)}, planning_head 张量数={len(planning_ids)}，"
        " 说明 freeze_perception_stack 未正确冻结感知栈。")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr)

    # ---------- 任务 7.6：训练前统计参数量 + 快照冻结权重 ----------
    # 可训练 / 冻结参数量（按 numel 统计），用于演示打印与自测。
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    # 冻结后、训练前快照 backbone + perception_head 权重；训练后比对最大绝对差（应为 0）。
    snapshot = _frozen_param_snapshot(model) if return_snapshots else None

    # ---------- 任务 7.2：主循环，只更新规划头 ----------
    # val_ratio=None -> 旧行为（全量训练，无验证）；给定则划分 train/val 做严肃训练。
    use_val = val_ratio is not None
    if use_val:
        train_loader, val_loader = make_train_val_loaders(
            cfg, cache=cache, val_ratio=val_ratio, seed=seed)
        if verbose:
            print(f"[stage2] device={device}  seed={seed}  "
                  f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
                  f"lr={cfg.lr}  bs={cfg.batch_size}")
    else:
        train_loader = make_loader(cfg, cache=cache, shuffle=True)
        val_loader = None

    # 说明（错误处理场景 4 / 性能考量）：阶段二理想情况下可把 backbone 前向包在
    # torch.no_grad() 里省显存。但 NanoFSD.forward 内部 backbone 与 planning_head
    # 是串联执行的，若整体包进 no_grad 会切断 planning_head 的梯度，导致无法训练；
    # 若只对 backbone 包 no_grad 需改造 NanoFSD.forward 的内部结构。
    # 这里以正确性为先：backbone + 感知头参数已 requires_grad=False 且处于 eval()，
    # autograd 不会为这些冻结叶子累积梯度、optimizer 也只持有规划头参数，
    # 因此权重不会被更新。no_grad 仅是可选的显存优化，不影响正确性，故不在此引入。
    best_path = _with_suffix(out_path, "_best")
    history = {"train_loss": [], "val_loss": [], "val_ade": []}
    loss_history: list[float] = []
    best_val_ade = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            # mode="planning" 前向 -> (B, Tf, 2)；正常前向（不包 no_grad）以保留规划头梯度
            traj_pred = model(batch.ego_hist, batch.agents,
                              batch.agent_mask, mode="planning")
            # 仅在有效未来步上算损失（traj_mask==False 的 padding 步被忽略）
            loss = planning_loss(traj_pred, batch.traj_gt, batch.traj_mask)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss_sum += loss.item()
            n_batches += 1

        avg_loss = epoch_loss_sum / max(n_batches, 1)
        loss_history.append(avg_loss)
        history["train_loss"].append(avg_loss)

        if use_val:
            val = evaluate_planning(model, val_loader, device)
            # evaluate_planning 内部会临时 eval 并按 model.training 还原，可能把冻结栈
            # 误还原为 train 态；这里重新冻结以保证 backbone/感知头始终处于 eval()。
            model.freeze_perception_stack()
            history["val_loss"].append(val["loss"])
            history["val_ade"].append(val["ade_m"])

            improved = val["ade_m"] < best_val_ade
            if improved:
                best_val_ade = val["ade_m"]
                best_epoch = epoch + 1
                epochs_since_improve = 0
                torch.save({"model": model.state_dict()}, best_path)
            else:
                epochs_since_improve += 1

            if verbose:
                star = " *" if improved else ""
                print(f"[stage2] epoch {epoch + 1:>3}/{epochs}  "
                      f"train_loss={avg_loss:.6f}  val_loss={val['loss']:.6f}  "
                      f"val_ADE={val['ade_m']:.3f} m{star}")

            if patience is not None and epochs_since_improve >= patience:
                if verbose:
                    print(f"[stage2] early stop：val ADE 连续 {patience} 轮未改善")
                break
        else:
            if verbose:
                print(f"[stage2] epoch {epoch + 1:>3}/{epochs}  "
                      f"平均规划 loss = {avg_loss:.6f}")

    torch.save({"model": model.state_dict()}, out_path)
    if verbose:
        print(f"[stage2] 已保存 checkpoint 到 {out_path}")

    if use_val:
        history["best_epoch"] = best_epoch
        history["best_val_ade"] = best_val_ade
        history["best_path"] = best_path
        if run_dir is not None:
            import json
            from dataclasses import fields as _fields
            os.makedirs(run_dir, exist_ok=True)
            cfg_dump = {f.name: getattr(cfg, f.name) for f in _fields(cfg)}
            cfg_dump.update({"epochs": epochs, "val_ratio": val_ratio, "seed": seed,
                             "stage1_ckpt": stage1_ckpt, "resolved_device": device})
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(cfg_dump, f, indent=2, ensure_ascii=False)
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        if verbose and best_epoch > 0:
            print(f"[stage2] 最优权重 -> {best_path}  "
                  f"(epoch {best_epoch}, val_ADE={best_val_ade:.3f} m)")

    if return_history:
        return model, history
    if return_snapshots:
        # 训练后比对：冻结生效时 max_abs_diff 应恒为 0.0（正确性属性 3）。
        max_abs_diff = _max_abs_diff_vs_snapshot(model, snapshot)
        info = {
            "trainable_count": trainable_count,
            "frozen_count": frozen_count,
            "max_abs_diff": max_abs_diff,
        }
        return model, loss_history, info
    return model, loss_history


def train_stage_e2e(cfg, epochs: int = 100, cache: str = "data/cache_mini.pt",
                    out_path: str = "checkpoint_e2e.pt",
                    verbose: bool = True, val_ratio: float = 0.2,
                    seed: int = 0, run_dir: str | None = None,
                    patience: int | None = None) -> tuple[NanoFSD, dict]:
    """端到端训练（不冻结）：backbone + 规划头一起用规划 loss 从头训。

    与阶段二的区别：**不加载感知预训练权重、不冻结**，AdamW 收集全部参数，仅用
    planning_loss 回传。backbone 的场景表示完全由规划信号塑造（对照实验：验证
    "先感知预训练再冻结" 相比 "直接端到端" 的价值）。

    与阶段一/二一致的严肃训练设施：固定 seed、train/val 划分、每 epoch 评估规划
    val loss 与 ADE（米）、按 val ADE 存最优权重、可选 early stopping、实验目录落盘。

    返回 (model, history)，history 含 train_loss/val_loss/val_ade/best_epoch/
    best_val_ade/best_path。
    """
    import json
    import os
    from dataclasses import fields as _fields

    _set_seed(seed)
    device = _resolve_device(cfg.device)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 关键区别：不 load_state_dict、不 freeze_perception_stack，全参数可训
    model = NanoFSD(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train_loader, val_loader = make_train_val_loaders(
        cfg, cache=cache, val_ratio=val_ratio, seed=seed)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    total_params = sum(p.numel() for p in model.parameters())
    best_path = _with_suffix(out_path, "_best")
    if verbose:
        print(f"[e2e] device={device}  seed={seed}  train={n_train}  val={n_val}  "
              f"lr={cfg.lr}  bs={cfg.batch_size}")
        print(f"[e2e] 端到端：全部 {total_params:,} 参数均可训练（不冻结）")

    history = {"train_loss": [], "val_loss": [], "val_ade": []}
    best_val_ade = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            # mode="planning" 前向；全模型参与反传（backbone + 规划头都更新）
            traj_pred = model(batch.ego_hist, batch.agents,
                              batch.agent_mask, mode="planning")
            loss = planning_loss(traj_pred, batch.traj_gt, batch.traj_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss_sum += loss.item()
            n_batches += 1

        train_loss = epoch_loss_sum / max(n_batches, 1)
        val = evaluate_planning(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_ade"].append(val["ade_m"])

        improved = val["ade_m"] < best_val_ade
        if improved:
            best_val_ade = val["ade_m"]
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save({"model": model.state_dict()}, best_path)
        else:
            epochs_since_improve += 1

        if verbose:
            star = " *" if improved else ""
            print(f"[e2e] epoch {epoch + 1:>3}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val['loss']:.6f}  "
                  f"val_ADE={val['ade_m']:.3f} m{star}")

        if patience is not None and epochs_since_improve >= patience:
            if verbose:
                print(f"[e2e] early stop：val ADE 连续 {patience} 轮未改善")
            break

    torch.save({"model": model.state_dict()}, out_path)
    history["best_epoch"] = best_epoch
    history["best_val_ade"] = best_val_ade
    history["best_path"] = best_path

    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        cfg_dump = {f.name: getattr(cfg, f.name) for f in _fields(cfg)}
        cfg_dump.update({"mode": "e2e", "epochs": epochs, "val_ratio": val_ratio,
                         "seed": seed, "resolved_device": device})
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg_dump, f, indent=2, ensure_ascii=False)
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[e2e] 最终权重 -> {out_path}")
        print(f"[e2e] 最优权重 -> {best_path}  "
              f"(epoch {best_epoch}, val_ADE={best_val_ade:.3f} m)")
    return model, history


def _plot_stage1_loss(loss_history: list[float],
                      val_loss: list[float] | None = None,
                      out_png: str = "outputs/step6_stage1_loss.png") -> None:
    """保存阶段一 train / val 感知 loss 曲线（x=epoch, y=masked_mse）。

    给定 val_loss 时叠加验证曲线，便于观察过拟合（train 降而 val 不降）。
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(loss_history) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, loss_history, "b.-", label="train loss")
    if val_loss is not None and len(val_loss) == len(epochs):
        ax.plot(epochs, val_loss, "r.-", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("perception loss (masked_mse)")
    ax.set_title("stage1 train/val loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1] 已保存 loss 曲线到 {out_png}")


def _plot_stage1_perception(model: NanoFSD, cfg, device: str,
                            cache: str = "data/cache_mini.pt",
                            out_png: str = "outputs/step6_perception.png",
                            idx: int = 0) -> None:
    """取一个样本跑感知头前向，画预测 vs 真值 agent 位置散点图（仅有效 agent、米制）。"""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data import NanoFSDDataset, denormalize

    # 取单个样本，加上 batch 维 B=1
    sample = NanoFSDDataset(cache)[idx]
    ego_hist = sample.ego_hist[None].to(device)      # (1, Th, 4)
    agents = sample.agents[None].to(device)          # (1, N, Th, C)
    agent_mask = sample.agent_mask[None].to(device)  # (1, N)

    model.eval()
    with torch.no_grad():
        pred = model(ego_hist, agents, agent_mask, mode="perception")[0]  # (N, 2)

    # 仅保留有效 agent，denormalize 回米制单位
    mask = sample.agent_mask                                   # (N,) bool
    pred_m = denormalize(pred.cpu()[mask]).numpy()             # (k, 2)
    gt_m = denormalize(sample.agent_gt[mask]).numpy()          # (k, 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_m[:, 0], gt_m[:, 1], s=40, c="green", marker="o",
               label=f"GT ({len(gt_m)})")
    ax.scatter(pred_m[:, 0], pred_m[:, 1], s=40, c="red", marker="x",
               label="Pred")
    ax.plot(0, 0, "b^", ms=10, label="ego (origin)")
    ax.set_aspect("equal")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(f"stage1 perception: pred vs GT (sample {idx})")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1] 已保存感知散点图到 {out_png}")


def demo_stage1(cfg, epochs: int | None = None,
                cache: str = "data/cache_mini.pt") -> tuple[NanoFSD, list[float]]:
    """阶段一演示入口（设计引用：算法 A、示例用法）。

    跑 `train_stage1` 得到模型与 loss 曲线，然后产出两张图：
      1. outputs/step6_stage1_loss.png —— 每个 epoch 的平均感知 loss 曲线。
      2. outputs/step6_perception.png  —— 取一个样本画感知头预测 vs 真值
         agent 位置散点（仅有效 agent，denormalize 回米制单位）。

    epochs=None 时沿用 train_stage1 的默认轮数（20）。
    """
    device = _resolve_device(cfg.device)
    if epochs is not None:
        model, history = train_stage1(cfg, epochs=epochs, cache=cache)
    else:
        model, history = train_stage1(cfg, cache=cache)

    _plot_stage1_loss(history["train_loss"], val_loss=history.get("val_loss"))
    _plot_stage1_perception(model, cfg, device, cache=cache)
    return model, history


def _plot_stage2_loss(loss_history: list[float],
                      out_png: str = "outputs/step7_stage2_loss.png") -> None:
    """保存阶段二训练的平均规划 loss 曲线（x=epoch, y=avg planning loss）。"""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(loss_history) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, loss_history, "r.-", label="avg planning loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("avg planning loss (masked_mse)")
    ax.set_title("stage2 training loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage2] 已保存 loss 曲线到 {out_png}")


def _plot_stage2_loss_trainval(train_loss: list[float],
                               val_loss: list[float] | None = None,
                               out_png: str = "outputs/step7_stage2_loss.png") -> None:
    """保存阶段二 train / val 规划 loss 曲线（给定 val 时叠加，便于看过拟合）。"""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_loss) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, "b.-", label="train loss")
    if val_loss is not None and len(val_loss) == len(epochs):
        ax.plot(epochs, val_loss, "r.-", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("planning loss (masked_mse)")
    ax.set_title("stage2 train/val loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage2] 已保存 loss 曲线到 {out_png}")


def demo_stage2(cfg, epochs: int | None = None,
                cache: str = "data/cache_mini.pt") -> tuple[NanoFSD, list[float]]:
    """阶段二演示入口（设计引用：算法 B、正确性属性 3）。

    跑 `train_stage2`（return_snapshots=True）得到模型、规划 loss 曲线与统计信息，
    围绕核心学习点「分阶段冻结训练」打印可观测证据并产图：
      1. 训练开始前打印可训练 / 冻结参数量（证明只有规划头参与训练）。
      2. 训练前快照 backbone + perception_head 权重，训练后打印与快照的最大绝对差
         （冻结生效时应为 0.0 —— 正确性属性 3）。
      3. 保存每个 epoch 的平均规划 loss 曲线到 outputs/step7_stage2_loss.png。

    epochs=None 时沿用 train_stage2 的默认轮数（20）。
    """
    if epochs is not None:
        model, loss_history, info = train_stage2(
            cfg, epochs=epochs, cache=cache, return_snapshots=True)
    else:
        model, loss_history, info = train_stage2(
            cfg, cache=cache, return_snapshots=True)

    # ---------- 打印参数量：可训练（规划头） vs 冻结（感知栈） ----------
    print("=" * 60)
    print(f"[stage2] 可训练参数量 = {info['trainable_count']:,}（仅 planning_head）")
    print(f"[stage2] 冻结参数量   = {info['frozen_count']:,}（backbone + perception_head）")

    # ---------- 冻结证据：训练前后感知栈权重最大绝对差（应为 0） ----------
    max_abs_diff = info["max_abs_diff"]
    print(f"[stage2] 感知栈权重训练前后最大绝对差 = {max_abs_diff:.3e}  (期望 = 0)")
    if max_abs_diff == 0.0:
        print("[stage2] ✅ 冻结生效：backbone + perception_head 权重完全未变（正确性属性 3）")
    else:
        print("[stage2] ⚠️ 冻结异常：感知栈权重发生变化，请检查 freeze_perception_stack")
    print("=" * 60)

    _plot_stage2_loss(loss_history)
    return model, loss_history


def demo_loss() -> None:
    """演示损失函数的三个核心性质（设计引用：算法 C、错误处理场景 2/3）。

    1. 损失零点：pred=gt 时 masked_mse / planning_loss ≈ 0（属性 4）。
    2. mask 不变性：往 padding 位填随机垃圾值前后，loss 完全不变（属性 5）。
    3. 全 padding：mask 全 False 时 loss 为 0 且不报错（场景 2）。

    CPU-only、确定性（manual_seed）。
    """
    from config import Config

    torch.manual_seed(0)
    cfg = Config()
    N, Tf = cfg.n_agents, cfg.fut_steps  # N=32, Tf=12

    # ---------- 1. 损失零点：pred == gt ----------
    print("=" * 60)
    print("[1] 损失零点示例（pred=gt 时 loss ≈ 0）")
    agent_gt = torch.randn(N, 2)
    agent_mask = torch.rand(N) > 0.5          # 随机有效/padding
    agent_pred = agent_gt.clone()             # 令 pred=gt
    mse0 = masked_mse(agent_pred, agent_gt, agent_mask)

    traj_gt = torch.randn(1, Tf, 2)
    traj_mask = torch.rand(1, Tf) > 0.3
    traj_pred = traj_gt.clone()
    plan0 = planning_loss(traj_pred, traj_gt, traj_mask)

    print(f"  有效 agent 数 = {int(agent_mask.sum())}/{N}")
    print(f"  masked_mse(pred=gt)    = {mse0.item():.3e}  (期望 ≈ 0)")
    print(f"  planning_loss(pred=gt) = {plan0.item():.3e}  (期望 ≈ 0)")

    # ---------- 2. mask 不变性：往 padding 位填垃圾值 ----------
    print("=" * 60)
    print("[2] mask 不变性示例（污染 padding 位，loss 不变）")
    pred = torch.randn(N, 2)
    gt = torch.randn(N, 2)
    mask = torch.rand(N) > 0.5
    loss_before = masked_mse(pred, gt, mask)

    # 复制一份，仅在 padding 位（mask==False）填入随机垃圾值
    pred_dirty = pred.clone()
    pad = ~mask
    garbage = torch.randn(int(pad.sum()), 2) * 1e4   # 大幅度垃圾值
    pred_dirty[pad] = garbage
    loss_after = masked_mse(pred_dirty, gt, mask)

    print(f"  padding 位数 = {int(pad.sum())}/{N}")
    print(f"  污染前 masked_mse = {loss_before.item():.6f}")
    print(f"  污染后 masked_mse = {loss_after.item():.6f}")
    print(f"  是否相等          = {torch.equal(loss_before, loss_after)}  (期望 True)")

    # ---------- 3. 全 padding：mask 全 False ----------
    print("=" * 60)
    print("[3] 全 padding 示例（mask 全 False，loss=0 且不报错）")
    pred_empty = torch.randn(N, 2)
    gt_empty = torch.randn(N, 2)
    mask_empty = torch.zeros(N, dtype=torch.bool)   # 全 False
    loss_empty = masked_mse(pred_empty, gt_empty, mask_empty)

    traj_mask_empty = torch.zeros(1, Tf, dtype=torch.bool)
    plan_empty = planning_loss(torch.randn(1, Tf, 2), torch.randn(1, Tf, 2),
                               traj_mask_empty)

    print(f"  masked_mse(全 padding)    = {loss_empty.item():.3e}  (期望 = 0)")
    print(f"  planning_loss(全 padding) = {plan_empty.item():.3e}  (期望 = 0)")
    print("=" * 60)


def run_stage1_training(cfg, epochs: int, cache: str = "data/cache_mini.pt",
                        val_ratio: float = 0.2, seed: int = 0,
                        run_dir: str | None = None,
                        patience: int | None = None) -> tuple[NanoFSD, dict]:
    """阶段一正式训练入口（严肃任务，非 demo）。

    自动给 run_dir 生成带时间戳的默认目录；训练后把 train/val loss 曲线存到该目录，
    最终与最优 checkpoint 分别落盘。返回 (model, history)。
    """
    import os

    if run_dir is None:
        run_dir = _make_run_dir("stage1", cfg, epochs, seed, cache)
    out_path = os.path.join(run_dir, "checkpoint_stage1.pt")

    with _log_to_dir(run_dir):
        model, history = train_stage1(
            cfg, epochs=epochs, cache=cache, out_path=out_path,
            val_ratio=val_ratio, seed=seed, run_dir=run_dir, patience=patience)

        # 训练曲线存到 run_dir（train/val loss 叠加，便于看过拟合）
        _plot_stage1_loss(history["train_loss"], val_loss=history.get("val_loss"),
                          out_png=os.path.join(run_dir, "stage1_loss.png"))
        # 训练完自动用最优感知权重出可视化图到实验目录
        print(f"[stage1] 训练完成，自动可视化（用最优权重 {history['best_path']}）...")
        _auto_visualize("perception", history["best_path"], cache, cfg, run_dir,
                        val_ratio=val_ratio, seed=seed)
        print(f"[stage1] 实验目录: {run_dir}")
    return model, history


def run_stage2_training(cfg, epochs: int, stage1_ckpt: str,
                        cache: str = "data/cache_mini.pt",
                        val_ratio: float = 0.2, seed: int = 0,
                        run_dir: str | None = None,
                        patience: int | None = None) -> tuple[NanoFSD, dict]:
    """阶段二正式训练入口（严肃任务，非 demo）：冻结感知栈、只训规划头。

    加载指定的阶段一感知权重 stage1_ckpt 并冻结，带 train/val 划分、每个 epoch 评估
    规划 val loss 与规划 ADE（米）、按 val ADE 保存最优规划权重，并把配置与训练曲线
    落盘到带时间戳的实验目录。seed 建议与阶段一一致，使 val 划分相同（更干净的评估）。
    返回 (model, history)。
    """
    import os

    if run_dir is None:
        run_dir = _make_run_dir("stage2", cfg, epochs, seed, cache)
    out_path = os.path.join(run_dir, "checkpoint_stage2.pt")

    with _log_to_dir(run_dir):
        model, history = train_stage2(
            cfg, epochs=epochs, cache=cache, stage1_ckpt=stage1_ckpt,
            out_path=out_path, val_ratio=val_ratio, seed=seed, run_dir=run_dir,
            patience=patience, return_history=True)

        _plot_stage2_loss_trainval(history["train_loss"], history.get("val_loss"),
                                   out_png=os.path.join(run_dir, "stage2_loss.png"))
        best = history.get("best_path", out_path)
        print(f"[stage2] 训练完成，自动可视化（用最优权重 {best}）...")
        _auto_visualize("planning", best, cache, cfg, run_dir,
                        val_ratio=val_ratio, seed=seed)
        print(f"[stage2] 实验目录: {run_dir}")
    return model, history


def run_stage_e2e_training(cfg, epochs: int, cache: str = "data/cache_mini.pt",
                           val_ratio: float = 0.2, seed: int = 0,
                           run_dir: str | None = None,
                           patience: int | None = None) -> tuple[NanoFSD, dict]:
    """端到端训练正式入口（不冻结，全参数 + 规划 loss）。

    自动生成带时间戳的实验目录，训练后把 train/val 规划 loss 曲线存到该目录，
    最终权重与最优权重分别落盘。返回 (model, history)。
    """
    import os

    if run_dir is None:
        run_dir = _make_run_dir("e2e", cfg, epochs, seed, cache)
    out_path = os.path.join(run_dir, "checkpoint_e2e.pt")

    with _log_to_dir(run_dir):
        model, history = train_stage_e2e(
            cfg, epochs=epochs, cache=cache, out_path=out_path,
            val_ratio=val_ratio, seed=seed, run_dir=run_dir, patience=patience)

        _plot_stage2_loss_trainval(history["train_loss"], history.get("val_loss"),
                                   out_png=os.path.join(run_dir, "e2e_loss.png"))
        best = history.get("best_path", out_path)
        print(f"[e2e] 训练完成，自动可视化（用最优权重 {best}）...")
        _auto_visualize("planning", best, cache, cfg, run_dir,
                        val_ratio=val_ratio, seed=seed)
        print(f"[e2e] 实验目录: {run_dir}")
    return model, history


if __name__ == "__main__":
    import argparse

    from config import Config

    parser = argparse.ArgumentParser(
        description="nano-fsd 训练入口（分阶段训练 / 损失演示）")
    parser.add_argument("--demo-loss", action="store_true",
                        help="运行损失函数演示：零点 / mask 不变性 / 全 padding")
    parser.add_argument("--stage", type=str, choices=["1", "2", "e2e"], default=None,
                        help="运行指定训练阶段：1=联合训练 backbone+感知头；"
                             "2=冻结感知栈只训规划头；e2e=端到端(不冻结,全参数+规划loss)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖默认训练轮数（可选）")
    # 严肃训练任务开关：带验证集、最优权重、可复现、落盘实验目录（作用于 --stage 1/2）。
    parser.add_argument("--train", action="store_true",
                        help="以正式训练模式运行 --stage 1/2（带 train/val 划分、验证指标、"
                             "最优权重保存、实验目录落盘），而非 demo 模式")
    parser.add_argument("--stage1-ckpt", type=str, default="checkpoint_stage1.pt",
                        help="阶段二加载的感知权重路径（建议用阶段一的 *_best.pt）")
    parser.add_argument("--cache", type=str, default="/home/miaozijian/life/mzj_planning/data/cache_trainval.pt",
                        help="数据缓存路径（如 data/cache_trainval.pt）")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集占比")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（可复现）")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="实验目录；默认 runs/stage{1,2}_<时间戳>")
    parser.add_argument("--patience", type=int, default=None,
                        help="early stopping 容忍轮数；默认关闭")
    args = parser.parse_args()

    def _apply_overrides(cfg):
        if args.lr is not None:
            cfg.lr = args.lr
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        return cfg

    if args.demo_loss:
        demo_loss()
    elif args.stage == "1":
        cfg = _apply_overrides(Config())
        if args.train:
            # 正式训练：带验证集、最优权重、实验目录落盘。
            epochs = args.epochs if args.epochs is not None else 100
            run_stage1_training(cfg, epochs=epochs, cache=args.cache,
                                val_ratio=args.val_ratio,
                                seed=args.seed, run_dir=args.run_dir,
                                patience=args.patience)
        else:
            # demo 模式：快速跑并产出演示图（向后兼容原行为）。
            demo_stage1(cfg, epochs=args.epochs)
    elif args.stage == "2":
        cfg = _apply_overrides(Config())
        if args.train:
            # 正式训练：加载指定感知权重冻结，带验证集、规划 ADE、最优权重、实验目录落盘。
            epochs = args.epochs if args.epochs is not None else 100
            run_stage2_training(cfg, epochs=epochs, stage1_ckpt=args.stage1_ckpt,
                                cache=args.cache, val_ratio=args.val_ratio,
                                seed=args.seed, run_dir=args.run_dir,
                                patience=args.patience)
        else:
            # demo 模式：向后兼容（从 checkpoint_stage1.pt 加载，产出冻结证据与 loss 图）。
            demo_stage2(cfg, epochs=args.epochs)
    elif args.stage == "e2e":
        # 端到端：不冻结、全参数 + 规划 loss；始终为正式训练模式（无 demo）。
        cfg = _apply_overrides(Config())
        epochs = args.epochs if args.epochs is not None else 100
        run_stage_e2e_training(cfg, epochs=epochs, cache=args.cache,
                               val_ratio=args.val_ratio, seed=args.seed,
                               run_dir=args.run_dir, patience=args.patience)
    else:
        parser.print_help()
