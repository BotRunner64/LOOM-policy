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

### loom:v2 全量预训练

`configs/loom_v2_pretrain_2b.json` 从头训练 LOOM 2B。六个数据来源均使用零参考段（`max_refs=0`）。每个样本取当前 episode 内最多两帧观测、任务指令和 50 帧动作目标。不留验证集，归一化统计从训练集估计。在 `pretrain-1kh-20261120` 上，一轮遍历 118,639,793 个训练锚点。GBS 1024 对应 115,860 次更新，尾批 177 个样本。

WSD 在前 0.005 轮线性预热至 `3e-4`，维持到 0.9 轮，最后 0.1 轮余弦衰减至零。每 50 次更新记录训练指标，每 1,000 次更新和训练结束时更新 `last.pt`。

训练从 LOOM-data 的打包视频读取图像（`loom-data pack`，格式见 LOOM-data 的打包格式契约）。`--packed` 指向打包根目录，其下每个数据集对应 `<源>/<布局>`；state、指令和 episode 元数据仍从 `--data` 的导出数据读取。打包数据与导出数据不一致时训练拒绝启动。不给 `--packed` 时逐个打开 MP4，在共享盘上加载跟不上 GPU。

在启智 PHASE-X 项目提交普通优先级的单机 8 卡作业，挂载 `--dataset loom:v2`，项目公共目录挂载为只读。打包数据在项目的 qb-ilm2 盘上，checkpoint 和日志写入全局个人盘。

```bash
work=/inspire/hdd/global_user/czxs253130598
public=/inspire/qb-ilm2/project/phase-x-top-down-prior-conditioned-visual-perceptual-modulation/public
cd "$work/projects/my_projects/LOOM-policy" && \
  "$work/cache/conda/envs/loom-policy/bin/python" -m torch.distributed.run \
    --standalone --nproc-per-node=8 --module loom_policy train \
    --config configs/loom_v2_pretrain_2b.json \
    --data /inspire/dataset/loom/v2/pretrain-1kh-20261120 \
    --packed "$public/dataset/LOOM-data/packed/pretrain-1kh-20261120" \
    --output "$work/runs/loom-v2-pretrain-2b"
```

预设每卡最多 32 个样本，8 卡时每次更新累积 4 轮前反向，每卡 16 个 DataLoader worker，每个 worker 用 16 个线程并行读一个批次。8×H100 实测约 920 samples/s，每步等数据约 5 ms，显存峰值 62 GiB，一轮约 36 小时。H200 上可用 `--micro-batch-size 64` 把累积轮数减半；H100 上每卡 64 个样本会显存不足。指令上限为 640 个 UTF-8 字节，这版数据的最长指令为 619 字节。换卡续训时挂载同一数据版本，沿用原输出目录。

多机训练用整机多节点作业（`--nodes 2`，每台 `8,160,1800`）。平台给每台机器注入 `PET_NNODES`、`PET_NODE_RANK`、`PET_MASTER_ADDR`、`PET_MASTER_PORT`；平台自己设的 `RANK`、`WORLD_SIZE` 数的是机器，不是进程，启动前去掉。每台机器运行同一条命令：

```bash
cd "$work/projects/my_projects/LOOM-policy" && \
  env -u RANK -u WORLD_SIZE "$work/cache/conda/envs/loom-policy/bin/python" -m torch.distributed.run \
    --nnodes "$PET_NNODES" --node-rank "$PET_NODE_RANK" \
    --master-addr "$PET_MASTER_ADDR" --master-port "$PET_MASTER_PORT" \
    --nproc-per-node=8 --module loom_policy train \
    --config configs/loom_v2_pretrain_2b.json \
    --data /inspire/dataset/loom/v2/pretrain-1kh-20261120 \
    --packed "$public/dataset/LOOM-data/packed/pretrain-1kh-20261120" \
    --output "$work/runs/loom-v2-pretrain-2b"
```

16 卡时每卡 32 个样本，累积 2 轮。2×8 H100 实测约 1,610 samples/s，是 8 卡的 1.75 倍，一轮约 20.5 小时。跨机梯度同步走 RoCE，每台 8 张 400 Gb/s 网卡。GBS 512 时 16 卡每卡只有一轮前反向，每步固定的梯度同步和优化器开销摊不开，只有约 1,420 samples/s；梯度改用 bf16 同步只快约 2%，没有采用。

主干的 25 个 Block 在 CUDA 上用 `torch.compile` 编译，启动后前几步较慢；用 `--set compile=false` 关闭。注意力无 mask 时固定用 FlashAttention-2，有 padding mask 的 Reader 交叉注意力用 memory-efficient attention，不用 cuDNN attention：Reader 的 padding 长度每批都变，cuDNN 要为每个新形状重新规划。

### Packing 单任务训练

Packing 预设默认训练 100 个 epoch。每轮打乱所有有完整未来动作窗口的训练锚点，每个锚点用一次。当前 Packing 数据每轮有 115,888 个锚点，GBS 128 下每轮更新 906 次（尾批 48 个样本），全程共更新 90,600 次。

WSD 在前 0.5 轮线性预热，随后维持峰值学习率 0.0001，完成 90 轮后开始最后 10 轮的余弦衰减。下面的命令使用完整的 100 轮计划，第一段作业在第 25 轮结束时保存退出。

```bash
torchrun --nproc_per_node 8 -m loom_policy train --config configs/rebot_packing_2b_gbs128_wsd.json \
  --data <packing数据目录> --output runs/packing-wsd --stop-at-epoch 25
```

续训时使用同一输出目录，以及相同的数据和数据目录顺序。预设每卡最多 16 个样本。GBS 128 的完整更新在 2、4、8 卡上分别需要 4、2、1 轮前反向。16、32、64 卡均只需一轮，每卡分别取 8、4、2 个样本。其他卡数按 rank 均分余数。可用 `--micro-batch-size` 设置本次作业的每卡上限。采样位置和 RTC 前缀由已提交的全局更新确定，跨卡数或 worker 数恢复时可接回同一批数据。

下一段可用 `--stop-at-epoch 50` 在第 50 轮结束后保存。轮内中断也可以从最近的 checkpoint 接着训练。稳定阶段可用 `--wsd-end-epoch` 调整计划总轮数，衰减仍占最后 10 轮。

离线评估默认使用零参考段。Packing 使用全量 episode 训练，下面在训练集上检查拟合情况。

```bash
loom-policy eval --checkpoint runs/packing-wsd/last.pt --data <packing数据目录> --split train
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

Packing 预设从相邻的 openpi_rebot 仓库读取 P1/P99 统计文件，每 500 次更新保存一次 checkpoint。部署和续训从 checkpoint 读取归一化参数。

### 动作前缀

训练时每个样本均匀抽取 0 到 `max_prefix_len` 帧作为前缀，默认上限为 4，至少留一帧用于预测。可按推理延迟对应的控制帧数调整上限。已知槽使用归一化动作的线性投影，待预测槽使用可学习的槽向量，两者都加上对应的时间编码。常规验证指标使用空前缀。训练和部署需要包含动作投影层的 checkpoint。

部署时传入从当前观测的下一帧起、队列里已承诺的连续目标位姿，形状为 `[d, 26]`，坐标系与当前观测一致。返回完整 chunk，前 `d` 帧原样保留，队列从 `chunk[d:]` 接上新动作。

```python
chunk = policy.act(states, images, cameras, instruction, prefix_states=queued_targets)
```

配置项都在 `loom_policy/config.py`，用 `--set key=value` 覆盖。
