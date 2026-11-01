# 当前设计：随机初始化 ViT / BERT encoder + 共享 Transformer

## 设计依据

优先级为用户当前决定、数据契约、已有架构草案：用户指定小型ViT图像encoder、小型BERT式文本encoder，均随机初始化并保留token序列，再进入共享Transformer；每个未来时间步一个action token回归20维动作；数据继续遵循 `LOOM-data/docs/contract/export-contract.md`；多模态统一 token 与共享主干沿用 [Notion LOOM-policy](https://app.notion.com/p/3f90249b500e80999eebe508023adde0) 的 v0.2 草案。

数据契约读取时 SHA256：`85cb65829aa30cbbb48b4f22921c336a92570b73c5e7bbe04122dc378f0a91a0`。上游文件是导出格式唯一事实源，此处只说明消费方式。

借鉴 [OpenVLA-OFT 论文](https://arxiv.org/abs/2502.19645) 的连续动作、并行 chunk 和 L1 目标。其[官方动作头](https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py)对动作输出槽的 hidden states 做 MLP 回归；本实现使用每个时间步一个 token 与共享 MLP，不复制其每维一个 token 再拼接的布局；主干是自己的小型统一 Transformer。

## 输入接口

`PolicyInput(current, time, references=[])`。`current` 是当前连续物理轨迹，包含历史和当前实际到达的 image/state/text 事件；`time` 是当前预测时刻（秒）；`references` 是其他物理轨迹的可见观测。

整理输入时保留当前段 `event.time <= time` 的事件。参考段保留给定可见事件。相同 `source_id` 不能既作为 current 又作为 reference；当前段必须至少有一个可见观测。来源真实性由调用方保证，标准适配器用 resolved episode 路径作为身份。

调用方不传入待预测动作、预测 offsets 或标签。模型内部追加固定 action 占位 token；变长的是观测，输出 H 与 Δt 固定为策略配置。输出槽用于并行读出，不接受可变未来时间查询。

## 异步导出数据

读取共同 clock 下 `timestamp_ns:int64`，不按行配对、不用 FPS 推算时间。四路 state 在原生时间并集上 backward-as-of 合并，只使用已经到达的样本，重复时间取最后一条；四路均到达且未超过 `max_age_ns` 才产生统一 state 事件。

每臂输入为相对前一个完整 state 事件的 `T_prev^-1 @ T_now`（pose12），加当前绝对 openness；双臂26维，首次相对位姿为 identity。未更新的 EEF 为 identity motion。先算完整事件 delta，再做显式数量采样；不会把删帧后的长步运动改写为原生一步。两个 EEF 可以使用不同固定世界系，body-frame delta 分别计算，不推断臂间外参。

pose12 为位置3和列优先完整旋转9：`[x,y,z,R00,R10,R20,R01,R11,R21,R02,R12,R22]`，按 SO(3) 检查。夹爪范围 `[0,1]`。相机按显示顺序解码，采集时间只读逐帧索引；检查 H.264、无音轨与帧数一致。instruction 从 start_ns 到达时作为文本事件，end_ns 不作模型特征。

## 模态内部编码

默认模态encoder与共享主干都使用D=128、4 heads和4D=512的GELU FFN，共用dropout配置；图像和文本深度分别由`vision_layers=2`、`text_layers=2`控制，必须为正，不提供跳过encoder的兼容路径。

**图像：小型ViT。** 每幅图像先以Conv2d做非重叠patch投影，加行/列分开的二维sin/cos位置编码，再过2层pre-LN Transformer与最终LayerNorm。二维位置在模态内attention之前加入，支持可变矩形patch网格。返回所有patch hidden states及对应中心xy坐标；不生成CLS、不做全局池化。每帧独立编码，跨帧、相机及参考/当前段的交互发生在共享主干。

**文本：小型BERT式encoder。** UTF-8字节Embedding(256,D)与可学习位置Embedding相加，经Embedding LayerNorm、dropout和2层双向post-LN/GELU Transformer；保留每个字节的hidden state。每个文本事件独立编码，内部没有padding或因果mask。输入仍使用256项字节词表，不引入WordPiece、CLS/SEP、MLM或NSP头。`max_text_tokens=512`按UTF-8字节计数，超限显式报错。

所有参数随机初始化并可训练，没有预训练模型下载或冻结参数。文本Linear/Embedding与attention投影使用std=0.02正态初始化；ViT及共享Transformer的attention/Linear投影用Xavier初始化。TransformerEncoder克隆出的每一层都重新独立初始化。仅以策略L1损失联合训练，不额外增加预训练任务。

结构参考：[ViT](https://arxiv.org/abs/2010.11929)、[BERT](https://arxiv.org/abs/1810.04805)。这里是适配LOOM的精简架构，不声称等同于官方预训练模型或具备其效果。

## 统一 token 与共享主干

默认 D=128，四层 pre-LN TransformerEncoder，4 heads、FFN=512，层末 LayerNorm。全注意力，batch padding mask 隔离填充；没有未来观测输入，因此不需要额外的时间因果 attention mask。

| 模态 | 内容映射 | token 数 |
|---|---|---|
| RGB | patch投影 + 2D位置 + 2层ViT | 每图的 patch 数，64×64 默认16个 |
| state26 | Linear(26,D) → GELU → Linear(D,D) | 每事件1个 |
| text | 字节/位置Embedding + 2层BERT式encoder | 每字节1个 |
| action output | 空内容 + sin/cos 固定槽位序号 | H个，默认8个 |

所有 token 逐一加同一个 metadata MLP 输出：`[tau,x,y,has_xy,E_segment,E_stream,E_modality]`，默认388→128→128。modality 包含 image/state/text/action output 四类。当前段观测 `tau=event.time-time`，参考段相对其可见末端归零，保留真实秒数和异步间隔。图像 x/y 是 patch 中心归一化坐标，其他模态 has_xy=0。

segment/stream 为局部类别 ID，训练时可随机重编号。参考与当前共用同一个ViT、同一个文本encoder及共享主干；不使用CLS或输入压缩器。action 输出槽绑定当前 segment、使用该段额外保留的 stream ID，tau=0、has_xy=0；槽位编码从0到H−1，与观测数量无关。

## 并行动作读出

每个未来时间步追加一个 action 输出 token。它只含固定的时间步位置编码和 metadata，不输入真实动作值或噪声。所有观测与输出 token 一起经过共享 Transformer，读取 action hidden states `[B,H,D]`；按各样本的实际起点切片，padding 不进入读出。

每个时间步的D维表示经过同一个动作头：`LayerNorm → Linear(D,D) → ReLU`，两个 `LayerNorm → Linear(D,D) → ReLU` 残差块，最后 `LayerNorm → Linear(D,20)`。默认D=128，因此动作头输入128维，直接并行输出 `[B,H,20]`。没有按坐标展开的 token 或跨token拼接，也不池化观测。

固定位置编码区分时间步，避免全注意力下空白输出 token 完全对称。输出物理时间由固定H与Δt决定，不接受可变未来时间查询。全部输出 token 并行读取观测并彼此交互，主干只执行一次；所有 token 均不含目标值，因此输出 token 间全注意力不会泄漏标签。

## 标签、损失与输出时间

模型配置固定 `chunk_size=H` 和 `action_interval_ns=Δt`。当前时刻 t 发起预测，标签是 `t+Δt,...,t+HΔt` 的最近已到达实测状态；不给首位置塞入 `t` 的同一份观测，避免 EEF 第一项恒为 identity 的无效目标。因为数据没有控制指令，这是下一控制步的目标轨迹代理监督。

每臂动作10维：平移3、旋转前两列6、实际夹爪1；双臂20维，左右各相对 t 时各自的观测位姿锚点，整个 chunk 共用。过期或超出该流末端的字段不监督，不越界延拓。loss 只对有效字段计算 L1，训练 batch 和完整验证集都按有效字段数归一化。未观测标签即使含 NaN，也不进入输入或误差计算。

训练和推理调用相同 `model(inputs)`。推理最后将6D旋转正交化为 SO(3)，夹爪裁剪到 `[0,1]`，乘当前 EEF anchor 还原绝对 pose12。这里只生成数学目标，不包含机器人下发或碰撞/限位处理。

## 配置与范围

默认每相机4帧、最多32个state事件，显式均匀取样并保留末端；数量可取消限制。`max_tokens`包含全部模态输出和H个action token，是计算预算，超过报错，不静默截断。图像/文本在模态内attention前检查剩余token预算，始终为H个action token预留位置。时间没有固定最大秒数，但可靠跨度取决于训练覆盖。

当前 checkpoint 格式 v6，固定输出时序随模型配置保存。没有多套动作头、兼容实现或生成式采样参数。尚未验证真实数据、GPU、大规模分布式和机器人任务效果。
