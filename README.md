# LOOM-policy

LOOM 的统一事件序列策略：图像 patch、step-wise relative state、UTF-8 文本和带噪动作查询，经共享元数据 MLP 编码后进入同一个 Transformer，以 **conditional flow matching head** 生成 chunk-wise relative 双臂动作块。

实现依据是 [Notion LOOM-policy](https://app.notion.com/p/3f90249b500e80999eebe508023adde0) 中 **2026-10-22 的 `loom-policy-design-v2.html`**，以及相邻 `LOOM-data/docs/contract/export-contract.md`。已废弃的 context/current 双分支、高压缩 encoder、CLS 等结构不在此实现中。

动作头按用户指定采用 flow matching：训练拟合从高斯噪声到动作块的速度场，推理从噪声经 ODE 积分生成动作。UTF-8 byte 文本映射、时间锚点及异步 state 合并仍是明确的实现选择。尚无真实数据训练收敛、跨域适配或机器人任务成功率结论。

## 安装

```bash
conda env create -f environment.yml
conda activate loom-policy
```

也可在已有 Python >=3.11 环境安装：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

仅需 PyTorch、NumPy、PyArrow、PyAV、Pillow；不下载预训练权重，不依赖 LOOM-data 的 Python 源码或源数据集内部字段。

## 数据 → 训练 → 恢复 → 推理

数据根目录必须是 LOOM-data 正式导出格式（`episodes.jsonl`、四路 state Parquet、逐相机 MP4/Parquet）。**没有 action 列**；本实现用未来实测 EEF/夹爪状态构造监督。这是未来观测轨迹的行为克隆代理目标，不是源控制器实际下发的指令。

先按物理 episode 划分训练和验证，再分别创建 query manifest；不要随机拆同一 episode 的窗口。可用 `--episodes` 明确选择目标集，用 `--references` 提供固定示例（会从目标集中排除）。

```bash
python -m loom_policy.index \
  --dataset /inspire/hdd/global_user/czxs253130598/path/to/export \
  --output runs/train.jsonl --episodes episode_0001 episode_0002 \
  --stride 16 --horizon 8 --dt-ns 50000000

# configs/baseline.json 默认读取 runs/train.jsonl，输出 runs/baseline/
python -m loom_policy.train --config configs/baseline.json
python -m loom_policy.train --config configs/baseline.json \
  --resume runs/baseline/last.pt --steps 2000

python -m loom_policy.predict --checkpoint runs/baseline/last.pt \
  --manifest runs/train.jsonl --index 0 --flow-steps 10 --seed 0 \
  --output runs/prediction.json
```

配置和 manifest 中的相对路径分别相对该 JSON/JSONL 文件目录解析。配置可加 `validation_manifest`；训练和验证的物理 episode（含参考）重叠时会拒绝启动。`--steps` 是包含已恢复步数的总训练步数。checkpoint 保存模型、优化器、样本游标、epoch 和随机数状态，原子替换 `last.pt`。CPU 测试覆盖中途恢复与连续训练逐参数一致；GPU 恢复没有逐位一致性保证。只加载自己产生或可信来源的 checkpoint。

默认是单进程 CPU 调试配置。GPU 单卡改 `train.device` 为 `cuda`，并按资源预算调整图像尺寸、patch、层数和 batch；代码尚不提供 DDP/FSDP、在线 KV cache 或机器人实时执行器。正式大训练应在低优先级 Job 中调用训练命令并将输出放在共享盘，以 checkpoint 恢复；本仓库不会自动申请集群资源。

训练 loss 和 `validation_loss` 都是 masked conditional flow-matching velocity MSE。验证使用独立固定种子的噪声和 flow time，避免消耗训练 RNG。推理 `--flow-steps` 控制 Euler 积分步数，缺省取模型配置 `flow_steps=10`；固定 `--seed` 可复现初始噪声。Checkpoint 格式为 v2，只支持当前 flow matching 模型。

## Manifest

每行显式描述一次 query，时间均为原始 episode 共同 clock 下的 int64 纳秒：

```json
{"dataset":"/path/to/export","episode_id":"episode_0002","query_time_ns":1000000000,"offsets_ns":[50000000,100000000],"references":[{"episode_id":"episode_0001"}]}
```

参考可提供不同 `dataset` 和可选 `end_ns`，缺省为参考 episode 末端。目标 episode 不能作为任何参考重新引入。末端不足完整 horizon 的字段以 loss mask 忽略，不零填为监督，不做越界外推。`max_age_ns` 限制 backward-as-of 查询允许的陈旧程度。indexer 自动跳过锚点缺失或完全没有未来监督的 query；直接编写 manifest 时这些问题会报错。

## 公开模型接口

```python
import torch
from loom_policy import Event, Segment, Query, PolicyInput, LoomPolicy

inputs = PolicyInput(
    segments=[
        Segment("goal", "goal-image-source", [
            Event(0.0, "cam", "image", torch.rand(3, 64, 64))]),
        Segment("target", "live-physical-trajectory", [
            Event(1.0, "cam", "image", torch.rand(3, 64, 64)),
            Event(0.0, "instruction", "text", "把方块放进盒子")])],
    query=Query("target", 1.0, (0.05, 0.10)))
model = LoomPolicy().eval()
with torch.no_grad():
    actions = model.sample_actions(
        [inputs], generator=torch.Generator().manual_seed(0))[0]  # [2,20]
    # 未训练模型，只展示接口
```

`Segment` 可以只有图像、文本或 state；缺失模态不产生 token。目标历史就是目标段更早的事件。不同物理轨迹需要不同 `source_id`，同一物理轨迹必须复用同一 ID，调用方须保留真实来源，不能通过改名绕过可见性。数据适配器使用 resolved episode 路径作为来源。视频示例、goal image 可以直接经此接口提供；标准导出适配器仍严格要求契约中的四路 state 与至少一个相机。

底层 `model(inputs, noisy_actions, flow_times)` 返回 `[H,20]` **速度场**；训练用 `model.flow_matching_loss(samples)`，推理用 `model.eval().sample_actions(...)` 返回最终动作。物理 query offsets 的单位为秒，flow time 是独立的 `[0,1]` 生成进度。

更多语义、维度、可见性和实现选择见 [docs/design.md](docs/design.md)，验证记录见 [docs/validation.md](docs/validation.md)。
