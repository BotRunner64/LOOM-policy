# LOOM-policy

多 domain co-train 的统一策略：部署时参数冻结，只靠参考 segment 适配。设计见 Notion「LOOM / policy」页，数据只读 [LOOM-data 导出契约](https://github.com/BotRunner64/LOOM-data/blob/main/docs/contract/export-contract.md)。

1. 当前和参考 segment 共用一个 reader（Perceiver IO）。默认当前段输出 256 个 latent；同一样本的所有参考段合计输出 256 个。一段参考用 256 个，两段各用 128 个，无法整除时排在前面的参考各多用一个。
2. 512 维 Reader 输出与 50 个动作槽拼接后，线性投影到 2560 维，进入 25 层、40 头的 Transformer 主干。整网共 1,995,094,548 个可训练参数。
3. 动作槽接收已知的动作前缀，并行回归剩余帧，后缀使用 L1 损失。

没有参考时主干接收 306 个 token，有参考时接收 562 个。`num_latents` 同时设置当前段和全部参考的预算。

新训练默认按每路 RGB 图像的原始宽高比缩放，高宽取整到 patch 的倍数。`image_size` 定义位置嵌入的基准网格和每张图的像素预算。默认 `192×256`、patch 边长 16，即每张图最多 192 个视觉 token。训练和部署共用缩放规则，`Policy.act` 和 `set_references` 都接受按帧、相机排列的不同尺寸 RGB 数组。训练也可通过 `--set adaptive_rgb=false` 选用固定尺寸。

## 环境

```bash
conda env create -f environment.yml
conda activate loom-policy && pytest
```

## 使用

新训练默认使用 LOOM 2B。`--data` 可以是数据集目录、源目录或导出根目录。如果输出目录已有 `last.pt`，会按 checkpoint 中的配置续训。

```bash
loom-policy train --data <数据目录>... --output runs/<name>
```

多卡：

```bash
torchrun --nproc_per_node 8 -m loom_policy train --data <数据目录>... --output runs/<name>
```

离线评估，同一个模型只换参考数量：

```bash
loom-policy eval --checkpoint runs/<name>/last.pt --data <数据目录>... --refs 0 1 2
```

部署用 `loom_policy.policy.Policy`：`set_references` 把参考编码一次，`act` 每步返回未来 50 帧的目标位姿。

### 本机 ReBot 策略服务

在 LOOM-policy 仓库中启用环境，首次安装控制器仓库内的通信包和服务依赖：

```bash
conda activate loom-policy
python -m pip install /path/to/rebot_control/policy_link -e '.[rebot]'
loom-policy serve --checkpoint ../models/rebot-packing-p1p99/step-20000.pt
```

服务使用 LOOM 的 PyTorch 推理，在 GPU 预热后监听 `ws://127.0.0.1:8010`。控制端沿用 `rebot policy` 的连接方式，指定 `--port 8010`。服务接收 top、left_wrist、right_wrist 三路 JPEG 和双臂绝对状态，按 30 Hz 返回目标。Packing 模型默认使用 3 步 RTC 前缀，可用 `--rtc-prefix-steps` 选择 1 至 6 步。

ReBot 单帧无参考实验使用 `--config configs/rebot_observation.json`。每个样本读取当前三视角图像、任务指令和数据集坐标系下的双臂绝对状态（每臂 xyz、按列展开的旋转矩阵和夹爪开合度，共 26 维）。动作目标仍以当前位姿为基准，共 20 维，训练前缀长度为 0 到 6。状态和动作分别按 P1/P99 缩放位置与夹爪，旋转坐标保留原值。默认从训练样本估计分位数。

Packing 从头训练使用 `configs/rebot_packing_2b.json`，8 卡各 8 个样本，训练 20,000 步，全量 episode 参与训练。该配置从相邻的 openpi_rebot 仓库读取本次 packing 数据的 P1/P99 统计文件，将状态和动作的位置与夹爪缩放到分位点对应的 −1 和 +1，保留超出分位点的值及旋转原值。部署和续训从 checkpoint 读取归一化参数。AdamW 峰值学习率为 1e-4，预热 1,000 步，余弦衰减，每 500 步保存一次。用同一输出目录重启会加载最新 checkpoint。

### 动作前缀

训练时每个样本均匀抽取 0 到 `max_prefix_len` 帧作为前缀，默认上限为 4，至少留一帧用于预测。可按推理延迟对应的控制帧数调整上限。已知槽使用归一化动作的线性投影，待预测槽使用可学习的槽向量，两者都加上对应的时间编码。常规验证指标使用空前缀。训练和部署需要包含动作投影层的 checkpoint。

部署时传入从当前观测的下一帧起、队列里已承诺的连续目标位姿，形状为 `[d, 26]`，坐标系与当前观测一致。返回完整 chunk，前 `d` 帧原样保留，队列从 `chunk[d:]` 接上新动作。

```python
chunk = policy.act(states, images, cameras, instruction, prefix_states=queued_targets)
```

配置项都在 `loom_policy/config.py`，用 `--set key=value` 覆盖。
