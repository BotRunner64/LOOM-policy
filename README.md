# LOOM-policy

多 domain co-train 的统一策略：部署时参数冻结，只靠参考 segment 适配。设计以 Notion「LOOM / policy」页为准。

输入是图像、state、文本，都在同一条 30 fps 帧序列上；数据只读 [LOOM-data 导出契约](https://github.com/BotRunner64/LOOM-data/blob/main/docs/contract/export-contract.md)（LeRobot v3.0，同步 30 fps）。

## 结构

1. 每个 segment（参考 / 当前）过同一个 reader（Perceiver IO），读成 L 个 latent。
2. 一个样本所有 segment 的 latent + 50 个动作槽 → 共享的全注意力 transformer。
3. 动作槽直接回归未来 50 帧（1.67 秒），L1。

主干计算量和参考长度无关；参考的 latent 不依赖当前段，部署时只算一次。

**state & action**（每臂 平移 3 + 旋转 6D + 夹爪 1，双臂 20 维）

- 输入 step-wise relative：先抽帧，再算相邻观测帧之间的 `T_prev⁻¹ · T_now`。
- 输出 chunk-wise relative：第 t+1 … t+50 帧的实测 state 相对第 t 帧，`T_t⁻¹ · T_{t+k}`。
- 夹爪开度取绝对值。输入、输出各用一套均值/标准差归一化，随 checkpoint 保存。

**token 的 meta data**：帧偏移（相对 segment 末帧，动作槽就是 +k）、相机、segment 类型（当前 / 参考）。

**采样**：当前段是以 t 结尾的 `current_frames` 个观测帧；参考段是同一数据集其它 episode 的一个窗口（最多 `ref_frames` 个观测帧），训练时每个样本随机取 0 … `max_refs` 段。参考只从 train episode 里取，验证集的 episode 不会被当作 context。

## 代码

| 文件 | 内容 |
|---|---|
| `loom_policy/config.py` | 全部配置；待定项 `obs_stride`（抽帧步长）、`num_latents`（L）在这里改 |
| `loom_policy/geometry.py` | 契约 state ↔ 相对位姿特征，误差度量 |
| `loom_policy/data.py` | 读契约数据集、采样 segment、collate |
| `loom_policy/model.py` | reader、共享 transformer、动作槽 |
| `loom_policy/train.py` | 训练（单进程或 torchrun DDP），checkpoint + resume |
| `loom_policy/eval.py` | 离线开环评估：同一个模型，只换参考数量 |
| `loom_policy/policy.py` | 部署封装：参考编码一次，之后每步只编码当前段 |

## 使用

```bash
pip install -e ".[dev]"
pytest
```

```bash
loom-train --data <导出根目录或数据集目录>... --output runs/<name> --config configs/base.json --set batch_size=64
```

`--data` 接受数据集目录、源目录或导出根目录。`--output` 下已有 `last.pt` 时自动续训，并沿用该 run 自己的配置。多卡：

```bash
torchrun --nproc_per_node 8 -m loom_policy.train --data <...> --output runs/<name> --config configs/base.json
```

```bash
loom-eval --checkpoint runs/<name>/last.pt --data <...> --refs 0 1 2
```

部署：

```python
from loom_policy.policy import Policy

policy = Policy('runs/<name>/last.pt')
policy.set_references([{'states': ..., 'images': ..., 'cameras': ['top', 'left_wrist', 'right_wrist'], 'instruction': ...}])
targets = policy.act(states, images, cameras, instruction, previous=...)   # [50, 26] 目标位姿，与输入 state 同一参考系
```

传入的帧按 `obs_stride` 间隔、从旧到新排列。

## 还没做

- 当前段不压缩（直接进主干）的变体；现在当前段和参考段一样过 reader。
- 多段参考之间不区分先后。
- 没有预训练权重，全部随机初始化。
