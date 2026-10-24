# 当前设计：观测输入，直接回归动作块

## 设计依据

优先级为用户当前决定、数据契约、已有架构草案：用户指定简单直接回归且不输入动作查询；数据继续遵循 `LOOM-data/docs/contract/export-contract.md`；多模态统一 token 与共享主干沿用 [Notion LOOM-policy](https://app.notion.com/p/3f90249b500e80999eebe508023adde0) 的 v0.2 草案。

数据契约读取时 SHA256：`85cb65829aa30cbbb48b4f22921c336a92570b73c5e7bbe04122dc378f0a91a0`。上游文件是导出格式唯一事实源，此处只说明消费方式。

借鉴 [OpenVLA-OFT 论文](https://arxiv.org/abs/2502.19645) 的连续动作、并行 chunk 和 L1 目标。其[官方动作头](https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py)对动作输出槽的 hidden states 做 MLP 回归；本实现使用观测池化直接读出，不复制 OFT 的输出槽机制。

## 输入接口

`PolicyInput(current, time, references=[])`。`current` 是当前连续物理轨迹，包含历史和当前实际到达的 image/state/text 事件；`time` 是当前预测时刻（秒）；`references` 是其他物理轨迹的可见观测。

整理输入时保留当前段 `event.time <= time` 的事件。参考段保留给定可见事件。相同 `source_id` 不能既作为 current 又作为 reference；当前段必须至少有一个可见观测。来源真实性由调用方保证，标准适配器用 resolved episode 路径作为身份。

输入中不包含待预测动作、动作占位 token、预测 offsets 或标签。变长的是观测，输出 H 与 Δt 固定为策略配置。标识当前段和当前时刻只是输入事实，不会生成可学习查询。

## 异步导出数据

读取共同 clock 下 `timestamp_ns:int64`，不按行配对、不用 FPS 推算时间。四路 state 在原生时间并集上 backward-as-of 合并，只使用已经到达的样本，重复时间取最后一条；四路均到达且未超过 `max_age_ns` 才产生统一 state 事件。

每臂输入为相对前一个完整 state 事件的 `T_prev^-1 @ T_now`（pose12），加当前绝对 openness；双臂26维，首次相对位姿为 identity。未更新的 EEF 为 identity motion。先算完整事件 delta，再做显式数量采样；不会把删帧后的长步运动改写为原生一步。两个 EEF 可以使用不同固定世界系，body-frame delta 分别计算，不推断臂间外参。

pose12 为位置3和列优先完整旋转9：`[x,y,z,R00,R10,R20,R01,R11,R21,R02,R12,R22]`，按 SO(3) 检查。夹爪范围 `[0,1]`。相机按显示顺序解码，采集时间只读逐帧索引；检查 H.264、无音轨与帧数一致。instruction 从 start_ns 到达时作为文本事件，end_ns 不作模型特征。

## 观测 token 与共享主干

默认 D=128，四层 pre-LN TransformerEncoder，4 heads、FFN=512，层末 LayerNorm。全注意力，batch padding mask 隔离填充；没有未来观测输入，因此不需要额外的时间因果 attention mask。

| 模态 | 内容映射 | token 数 |
|---|---|---|
| RGB | Conv2d(3,D,kernel=16,stride=16) | 每图的 patch 数，64×64 默认16个 |
| state26 | Linear(26,D) → GELU → Linear(D,D) | 每事件1个 |
| text | UTF-8 byte embedding + sin/cos 字节序号 | 每字节1个 |

所有 token 逐一加同一个 metadata MLP 输出：`[tau,x,y,has_xy,E_segment,E_stream,E_modality]`，默认388→128→128。modality 只有 image/state/text 三类。当前段 `tau=event.time-time`，参考段相对其可见末端归零，保留真实秒数和异步间隔。图像 x/y 是 patch 中心归一化坐标，其他模态 has_xy=0。

segment/stream 为局部类别 ID，训练时可随机重编号。参考与当前共用权重，不存在独立深视觉塔、CLS 或输入压缩器。

## 直接读出动作块

所有观测经过共享 Transformer 后，只平均池化当前轨迹 token 的 hidden states。参考通过 attention 影响这些 hidden states，但不直接进入池化集合；padding 不进入池化。池化在充分混合之后进行，不在输入端提前丢弃多模态信息。

`pooled[D] → LayerNorm → Linear(D,D) → GELU → Linear(D,H*20) → reshape[H,20]`。

每个输出位置由最后一层的独立参数确定其 horizon 顺序，不需要动作查询或未来时间 embedding。主干只执行一次。固定 H、固定采样周期和池化是明确的简化假设；表达能力及任务效果仍需要训练验证，不把“能拟合”当成成功率结论。

## 标签、损失与输出时间

模型配置固定 `chunk_size=H` 和 `action_interval_ns=Δt`。当前时刻 t 发起预测，标签是 `t+Δt,...,t+HΔt` 的最近已到达实测状态；不给首位置塞入 `t` 的同一份观测，避免 EEF 第一项恒为 identity 的无效目标。因为数据没有控制指令，这是下一控制步的目标轨迹代理监督。

每臂动作10维：平移3、旋转前两列6、实际夹爪1；双臂20维，左右各相对 t 时各自的观测位姿锚点，整个 chunk 共用。过期或超出该流末端的字段不监督，不越界延拓。loss 只对有效字段计算 L1，训练 batch 和完整验证集都按有效字段数归一化。未观测标签即使含 NaN，也不进入输入或误差计算。

训练和推理调用相同 `model(inputs)`。推理最后将6D旋转正交化为 SO(3)，夹爪裁剪到 `[0,1]`，乘当前 EEF anchor 还原绝对 pose12。这里只生成数学目标，不包含机器人下发或碰撞/限位处理。

## 配置与范围

默认每相机4帧、最多32个state事件，显式均匀取样并保留末端；数量可取消限制。`max_tokens` 是计算预算，超过报错，不静默截断。时间没有固定最大秒数，但可靠跨度取决于训练覆盖。

当前 checkpoint 格式 v3，固定输出时序随模型配置保存。没有多套动作头、兼容实现或生成式采样参数。尚未验证真实数据、GPU、大规模分布式和机器人任务效果。
