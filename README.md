# LOOM-policy

图像经过小型 ViT、文本经过小型 BERT 式 encoder，保留完整 token 序列；state 经 MLP，随后与固定 action 输出 token 一起经过共享 Transformer，动作 token 的 hidden states 经共享 MLP 直接回归当前时刻往后的双臂 action chunk。参考和当前轨迹共用输入映射与主干。

```text
参考观测 + 当前轨迹截至 t 的观测
  图像 → 小型 ViT：patch映射 + 2D位置编码 + 2层自注意力
  文本 → 小型 BERT式encoder：字节/位置Embedding + 2层双向注意力
  state → MLP
        ↓ 保留各patch/文本token，加共享metadata
观测 token + H个action token → 4层共享Transformer
        ↓ 取action hidden states [H,D]
共享残差 MLP → [H,20] 动作块
        ↓ SO(3)投影 + 当前位姿锚点
双臂绝对目标位姿 + 夹爪开合
```

采用并行 action token 读出与 L1 回归：每个未来时间步一个固定输出 token，默认8步共8个。token 内不输入真实动作，只由固定位置编码区分时间步；每个 hidden state 经同一个 MLP 输出该步完整的20维动作。借鉴 [OpenVLA-OFT](https://github.com/moojink/openvla-oft) 的并行连续回归，不沿用其每维一个 token 的布局。没有观测池化、带噪动作、采样循环或可变未来时间查询。

所有网络随机初始化、端到端训练，不加载或下载预训练权重。ViT为pre-LN/GELU结构，BERT式文本encoder为post-LN/GELU结构；默认各2层、D=128、4 heads、FFN=512。参考和当前观测共用同一套模态encoder。没有CLS或池化，输入token数量保持不变。

文本仍使用UTF-8字节词表（256项），每字节一个token，使用可学习位置Embedding；它是BERT式架构，不使用WordPiece、预训练语言知识或MLM/NSP头。`max_text_tokens=512`限制单个文本事件的字节数，超限报错，不静默截断。

## 安装

```bash
conda env create -f environment.yml
conda activate loom-policy
# 或已有 Python >=3.11 环境：
python -m pip install -e '.[dev]'
python -m pytest -q
```

## 数据与输出时序

只消费相邻 `LOOM-data/docs/contract/export-contract.md` 的正式导出格式：`episodes.jsonl`、四路异步 state Parquet、逐相机 H.264 MP4/时间索引。不依赖单个源数据集字段。

输出长度和周期固定在模型配置中，默认 `chunk_size=8`、`action_interval_ns=50000000`。在当前时刻 t，预测 t+Δt 到 t+HΔt 的目标；数据没有 action，标签由未来实测 EEF/夹爪构造。第一项对应下一个控制步的目标，不是把当前观测原样复制为第一个动作。

各臂 EEF 使用自己的当前观测锚点，输出 chunk 内共享锚点。每臂为位置3、旋转6D、夹爪1，共20维；输入 state 为 step-wise body-relative 双臂 pose12 与实际夹爪，共26维。缺失/过期/超出流末尾的标签字段以 loss mask 排除。

“截至 t”只是输入取样规则：当前轨迹未来图像/state 不输入，未来实测值只制作标签；不需要专门的未来信息网络模块。参考轨迹可以完整提供，但不能把当前轨迹未来部分重标为参考。

## 索引、训练、恢复、预测

先按物理 episode 划分训练和验证，再分别生成索引。参考用 `--references` 指定并从目标集中排除。索引和训练使用同一配置读取输出时序：

```bash
python -m loom_policy.index --config configs/baseline.json \
  --dataset /inspire/hdd/global_user/czxs253130598/path/to/export \
  --output runs/train.jsonl --episodes episode_0001 episode_0002 --stride 16
python -m loom_policy.train --config configs/baseline.json
python -m loom_policy.train --config configs/baseline.json \
  --resume runs/baseline/last.pt --steps 2000
python -m loom_policy.predict --checkpoint runs/baseline/last.pt \
  --manifest runs/train.jsonl --index 0 --output runs/prediction.json
```

manifest 每行只有数据来源、当前截止时刻和参考，不携带输出查询：

```json
{"dataset":"/path/to/export","episode_id":"episode_0002","time_ns":1000000000,"references":[{"episode_id":"episode_0001"}]}
```

参考可指定不同 `dataset` 和可选 `end_ns`。配置和 manifest 的相对路径分别相对各自文件所在目录解析。可在配置加 `validation_manifest`，所有训练/验证物理源（含参考）必须隔离。checkpoint v6 原子保存模型、优化器、数据游标、epoch 与 RNG；`--steps` 为包含恢复步数的总训练步数。训练及验证指标均为有效动作字段的 L1。

## 模型接口

```python
import torch
from loom_policy import Event, Segment, PolicyInput, LoomPolicy

inputs = PolicyInput(
    current=Segment("current", "live-episode", [
        Event(1.0, "cam", "image", torch.rand(3, 64, 64)),
        Event(0.0, "instruction", "text", "把方块放进盒子")]),
    time=1.0,
    references=[Segment("goal", "goal-image", [
        Event(0.0, "cam", "image", torch.rand(3, 64, 64))])])
model = LoomPolicy().eval()
with torch.no_grad():
    actions = model([inputs])  # [batch=1, chunk_size=8, action_dim=20]
```

缺失模态不产生 token。`current` 和 `time` 定义观测边界；模型在观测后追加固定数量的 action 输出槽，不携带未来实测值。参数未经训练时，上例只展示接口。推理只需一次前向，输出确定；没有噪声、采样步数或随机 seed 参数。

当前为单进程研究基线。默认 CPU 调试，GPU 单卡改 `train.device=cuda`。尚不提供分布式训练、在线缓存或机器人执行器，也没有真实任务效果结论。实现细节见 [docs/design.md](docs/design.md)，验证见 [docs/validation.md](docs/validation.md)。

开发期直接替换废弃实现，同步删除旧代码、配置、文档和失效实验产物，不维护兼容路径。
