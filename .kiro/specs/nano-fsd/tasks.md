# 实现计划：nano-fsd

## 概述

本计划把 design.md 末尾的「分步施工路线」（6 步）展开为可独立验证的编码任务，用于你亲手一步步手写实现一个 nanoGPT 量级的端到端自动驾驶模仿学习项目。核心学习点是**分阶段冻结训练**：阶段一联合训练 backbone + 感知头，阶段二冻结感知栈（backbone + 感知头）只训练规划头。

每一步都能独立跑通并自测，并配有一个「演示入口」子任务：完成该步后直接运行一条命令，就能看到打印的形状/统计或生成的图片。最终成功标准是：在 BEV 图上画出规划头预测的自车未来轨迹 vs 真值轨迹，视觉上基本吻合。

技术栈：Python + PyTorch，属性测试用 hypothesis，可视化用 matplotlib。所有张量 batch 维在前，坐标统一到 ego 坐标系。

设计中的正确性属性（供下文引用）：
1. 形状守恒 · 2. padding 不变性 · 3. 冻结正确性 · 4. 损失零点 · 5. mask 单调性 · 6. ego 原点性 · 7. 过拟合可行性

## 任务

- [x] 1. 搭建项目骨架与配置
  - [x] 1.1 创建 `config.py`
    - 用 dataclass 实现 `Config`，集中管理超参：`n_agents`、`hist_steps`、`fut_steps`、`agent_feat_dim`、`d_model`、`n_heads`、`n_layers`、`lr`、`batch_size`、`device`
    - 设计默认值（N=32, Th=4, Tf=12, C=9, D=128, heads=4, layers=3, lr=3e-4, bs=64）
    - 注：`agent_feat_dim` 已由 8 调整为 9 = `[x, y, vx, vy, cos, sin, onehot×3]`
    - _设计引用：配置对象_
  - [x] 1.2 建立项目目录与依赖清单
    - 创建 `model.py`、`data.py`、`train.py`、`viz.py` 空文件骨架
    - 写 `requirements.txt`：torch、nuscenes-devkit、numpy、matplotlib、hypothesis、tqdm
    - _设计引用：概述、依赖_
  - [x] 1.3 演示入口：`python config.py`
    - 在 `config.py` 加 `if __name__ == "__main__":`，逐行打印 `Config` 全部默认值
    - 额外打印 `agent_feat_dim=9` 各维含义 `[x, y, vx, vy, cos, sin, onehot×3]`
    - _设计引用：配置对象_

- [x] 2. 数据层 data.py — 结构与预处理缓存
  - [x] 2.0 【前置准备·非编码】下载 nuScenes v1.0-mini
    - 到 nuScenes 官网（nuscenes.org）注册账号并下载 `v1.0-mini`（约 4GB），解压到本地目录
    - 安装 `nuscenes-devkit`，确认能用 devkit 打开 mini 集
    - 说明：此步需人工在官网注册获取数据，无法由编码完成；后续数据任务依赖它
  - [x] 2.1 定义 `Sample` 数据结构
    - 用 dataclass 实现：`ego_hist (Th,4)`、`agents (N,Th,C)`、`agent_mask (N,)`、`agent_gt (N,2)`、`traj_gt (Tf,2)`、`traj_mask (Tf,)`
    - 加入设计中的验证规则断言：ego 当前帧位置≈原点、padding 行全零、相邻轨迹点距离合理
    - _设计引用：输入样本结构、验证规则_
  - [x] 2.2 实现 devkit 读取与 ego 坐标系转换
    - 用 devkit 遍历 mini 场景，取各 agent 框中心/朝向/速度/类别、自车历史与未来位姿
    - 将全局坐标转换到以自车当前时刻为原点的 ego 坐标系（x 向前、y 向左）
    - _设计引用：坐标系约定、关键设计决策_
  - [x] 2.3 实现标准化、padding 与缓存
    - 坐标除以场景尺度（如 50m）标准化；提供配套的 `denormalize`
    - 将 agent 数 padding 到 `N`，无效位填零并置 `agent_mask`；未来轨迹不足 `Tf` 时 pad 并置 `traj_mask`
    - 过滤有效 agent 数为 0 的样本；一次性预处理为向量张量缓存到磁盘 `.pt`
    - _设计引用：数据模型、错误处理场景 2/3、性能考量_
  - [x] 2.4 实现 `Dataset`/`DataLoader` 封装
    - 从 `.pt` 缓存加载，按 `Config.batch_size` 产出 batch（张量 batch 维在前）
    - _设计引用：主流程时序图、示例用法_
  - [x] 2.5 为数据层写属性测试（hypothesis）
    - **属性 1 形状守恒**：任意样本的各字段形状符合 `(Th,4)/(N,Th,C)/(N,)/(N,2)/(Tf,2)/(Tf,)`
    - **属性 6 ego 原点性**：`ego_hist[-1]` 位置分量范数 < ε
    - _验证：正确性属性 1、6_
  - [x] 2.6 演示入口：`python data.py --demo`
    - 前提：已用 `python data.py` 生成 `data/cache_mini.pt`；`--demo` 只读缓存，不重新预处理
    - 打印统计：样本数、平均有效 agent 数、agent 被截断（超过 N）的样本数、未来轨迹被 pad 的样本数、各字段形状
    - 保存一张原始场景 BEV 图到 `outputs/step2_data.png`：周围 agent 位置、自车历史、GT 未来轨迹（`denormalize` 回米制单位）
    - 若 2.4 已完成：再用 `DataLoader` 取一个 batch 并打印各字段形状（batch 维在前）
    - _设计引用：数据模型、坐标系约定、示例用法_

- [x] 3. 模型层 model.py — Backbone + 两个头 + 顶层整合
  - [x] 3.1 实现 `Backbone`（场景编码器）
    - 把自车作为 token 0、每个 agent 作为 token 1..N，用 `nn.TransformerEncoderLayer` 堆叠 `n_layers` 层
    - padding agent 通过 `key_padding_mask` 屏蔽；输出 `scene_emb (B, N+1, D)`，token 顺序与输入一一对应
    - _设计引用：组件 1、函数 1 的前/后置条件与循环不变式_
  - [x] 3.2 实现 `PerceptionHead`（感知头）
    - 取 token 1..N 过 MLP 输出 `agent_pred (B, N, 2)`（方案 A：相对位置回归）
    - _设计引用：组件 2、感知头输出方案 A_
  - [x] 3.3 实现 `PlanningHead`（规划头）
    - 取 ego token（token 0）过 MLP 回归 `traj_pred (B, Tf, 2)`
    - _设计引用：组件 3_
  - [x] 3.4 实现 `NanoFSD` 顶层整合
    - 组合 backbone + perception_head + planning_head
    - `forward(ego_hist, agents, agent_mask, mode)`：`mode ∈ {"perception","planning"}` 返回对应头输出
    - _设计引用：组件 4_
  - [x] 3.5 实现 `freeze_perception_stack`（核心学习点）
    - 冻结 backbone + perception_head 全部参数（`requires_grad=False`）并置 `eval()`
    - 保持 planning_head 参数 `requires_grad=True`
    - _设计引用：组件 4、函数 3 的前/后置条件_
  - [x] 3.6 为模型写形状与 padding 属性测试（hypothesis）
    - **属性 1 形状守恒**：随机合法输入下 backbone 输出恒 `(B,N+1,D)`、planning_head 恒 `(B,Tf,2)`
    - **属性 2 padding 不变性**：仅在 padding agent 内容上不同的两组输入，有效位输出完全一致
    - _验证：正确性属性 1、2_
  - [x] 3.7 为冻结开关写单元测试
    - **属性 3 冻结正确性（静态部分）**：调用 `freeze_perception_stack` 后，backbone/感知头参数 `requires_grad==False`，规划头 `requires_grad==True`
    - _验证：正确性属性 3_
  - [x] 3.8 演示入口：`python model.py`
    - 用随机输入（含部分 padding agent）跑一次前向，打印 `scene_emb`、`agent_pred`、`traj_pred` 的形状
    - 打印各模块参数量：backbone / perception_head / planning_head / 合计
    - 冻结演示：打印调用 `freeze_perception_stack` 前后各模块 `requires_grad=True` 的参数个数
    - _设计引用：组件 1–4、函数 3_

- [x] 4. 损失函数（感知头与规划头共用，统一放在 train.py）
  - [x] 4.1 实现 `masked_mse`
    - 按设计伪代码：逐元素平方误差乘 mask，分母用 `clamp(min=1)` 防除零
    - _设计引用：算法 C、错误处理场景 2_
  - [x] 4.2 实现 `planning_loss`
    - 复用 masked 思路，仅在 `traj_mask==True` 的时间步计算
    - _设计引用：函数 2 的前/后置条件、错误处理场景 3_
  - [x] 4.3 为损失函数写属性测试（hypothesis）
    - **属性 4 损失零点**：随机 gt、令 pred=gt，断言 `masked_mse`/`planning_loss`≈0
    - **属性 5 mask 单调性**：增加 padding（更多无效位）不改变有效位上的 loss 值
    - _验证：正确性属性 4、5_
  - [x] 4.4 演示入口：`python train.py --demo-loss`
    - 打印零点示例：`pred=gt` 时 `masked_mse`、`planning_loss` 的值（≈0）
    - 打印 mask 不变性示例：在 padding 位填随机垃圾值前后，loss 值相同
    - 打印全 padding 示例：`mask` 全 False 时 loss 为 0 且不报错
    - _设计引用：算法 C、错误处理场景 2/3_

- [x] 5. 检查点 — 确保基础模块通过
  - 确保所有测试通过，若有疑问请询问用户。
  - 此时 data/model/loss 三层已可用随机或真实数据跑通前向与损失

- [x] 6. train.py 阶段一 — 联合训练 backbone + 感知头
  - [x] 6.1 实现 `train_stage1` 主循环
    - 构建 `NanoFSD`，`AdamW` 收集全部参数；`mode="perception"` 前向，用 `masked_mse(pred, agent_gt, agent_mask)` 回传
    - 训练完保存 `checkpoint_stage1.pt`（含 backbone + perception_head 权重）
    - _设计引用：算法 A、阶段一时序图_
  - [x] 6.2 加入命令行入口 `--stage 1`
    - 让 `python train.py --stage 1` 可直接运行阶段一
    - _设计引用：示例用法_
  - [x] 6.3 单 batch 过拟合 sanity check（感知）
    - **属性 7 过拟合可行性**：固定一个 batch 反复训练若干步，断言感知 loss 显著下降（接近 0）
    - _验证：正确性属性 7_
  - [x] 6.4 演示输出：`python train.py --stage 1`
    - 每个 epoch 打印平均感知 loss；训练结束保存 loss 曲线到 `outputs/step6_stage1_loss.png`
    - 取一个样本，画感知头预测的 agent 位置 vs GT 位置散点图（仅有效 agent，米制单位）到 `outputs/step6_perception.png`
    - _设计引用：算法 A、示例用法_

- [x] 7. train.py 阶段二 — 冻结感知栈，只训规划头（核心学习点）
  - [x] 7.1 实现 stage1 权重加载与冻结
    - 加载 `checkpoint_stage1.pt`，`load_state_dict(..., strict=False)`（规划头为新初始化），打印缺失/多余键
    - 调用 `model.freeze_perception_stack()`；`optimizer` 只收集 `requires_grad=True` 的参数并自检这些参数都属于 planning_head
    - 若 stage1 文件不存在则报错提示先跑阶段一
    - _设计引用：算法 B、错误处理场景 1_
  - [x] 7.2 实现 `train_stage2` 主循环
    - backbone 前向包在 `torch.no_grad()` 省显存；`mode="planning"` 前向，用 `planning_loss(traj_pred, traj_gt, traj_mask)` 回传
    - 只更新 planning_head；训练完保存 `checkpoint_stage2.pt`
    - _设计引用：算法 B、错误处理场景 4、性能考量_
  - [x] 7.3 加入命令行入口 `--stage 2`
    - 让 `python train.py --stage 2` 可直接运行阶段二
    - _设计引用：示例用法_
  - [x] 7.4 冻结端到端验证测试
    - **属性 3 冻结正确性（动态部分）**：跑一个阶段二训练步，前后对比 backbone 与 perception_head 参数逐元素相等；planning_head 至少一个参数发生变化
    - _验证：正确性属性 3_
  - [x] 7.5 单 batch 过拟合 sanity check（规划）
    - **属性 7 过拟合可行性**：固定 batch 反复训练，断言规划 loss 被压到接近 0（证明梯度链路通畅）
    - _验证：正确性属性 7_
  - [x] 7.6 演示输出：`python train.py --stage 2`
    - 训练开始前打印可训练 / 冻结参数量
    - 训练前快照 backbone + perception_head 权重，训练后打印与快照的最大绝对差（应为 0）
    - 每个 epoch 打印平均规划 loss；保存 loss 曲线到 `outputs/step7_stage2_loss.png`
    - _设计引用：算法 B、正确性属性 3_

- [x] 8. 检查点 — 确保分阶段训练通过
  - 确保所有测试通过，若有疑问请询问用户。
  - 此时两个 checkpoint 均已产出，冻结训练链路验证完毕

- [x] 9. viz.py — BEV 轨迹可视化（达成成功标准）
  - [x] 9.1 实现 `visualize`
    - 在同一张 BEV 图上画周围 agent 框、GT 轨迹（绿）、Pred 轨迹（红）、自车原点（蓝三角）
    - `set_aspect('equal')` + 图例
    - _设计引用：算法 D、组件时序图_
  - [x] 9.2 实现可视化入口 `--ckpt`
    - 加载 `checkpoint_stage2.pt`，`model.eval()`，对样本 `no_grad` 前向得到轨迹，`denormalize` 后调用 `visualize`
    - 让 `python viz.py --ckpt checkpoint_stage2.pt` 产出对比图
    - 成功判据：直行场景两条线接近，转弯场景预测能跟上弯道趋势
    - _设计引用：示例用法、成功标准_
  - [x] 9.3 演示输出：多样本保存
    - `python viz.py --ckpt checkpoint_stage2.pt` 对若干样本（至少包含直行与转弯）各存一张图到 `outputs/step9_traj_*.png`
    - 每张图标题标注样本序号与该样本的平均位移误差（米）
    - _设计引用：成功标准_

- [x] 10. 最终检查点 — 确保全部通过
  - 确保所有测试通过，若有疑问请询问用户。
  - 在 BEV 图上确认预测轨迹 vs 真值轨迹视觉吻合，达成项目成功标准

## 说明

- 标注 `*` 的子任务为可选（单元测试、属性测试、集成测试），追求最快跑通可先跳过；核心实现任务不带 `*`，必须实现。
- 每一步都有演示入口（1.3、2.6、3.8、4.4、6.4、7.6、9.3），图片统一输出到 `outputs/` 目录（不存在时自动创建）。
- 任务 2.0 是非编码的前置准备项（官网注册下载 nuScenes mini），需人工完成后才能跑真实数据。
- 每个任务引用了设计文档的具体章节/属性以便追溯。
- 属性测试统一用 hypothesis，每条正确性属性对应独立子任务并标注属性编号。
- 检查点（任务 5、8、10）用于分段验证管道是否打通。
- 核心学习点「分阶段冻结训练」集中体现在任务 3.5、6、7，务必亲手跑一遍并通过冻结正确性验证（属性 3）。
