# 实现合同与设计选择

## 依据与优先级

1. [Notion LOOM-policy](https://app.notion.com/p/3f90249b500e80999eebe508023adde0)，当前架构附件 `loom-policy-design-v2.html`，讨论稿 v0.2 / 2026-10-22。SHA256：`494a3fa57571a7bb7a914dae7f7274c2a5989ae84a29c4c374fabaad9d530e6b`。
2. 相邻 `LOOM-data/docs/contract/export-contract.md`，读取时仓库 HEAD：`fc93c1e030afea952af13d8a97602b763113ab87`；契约文件 SHA256：`85cb65829aa30cbbb48b4f22921c336a92570b73c5e7bbe04122dc378f0a91a0`。此文件是导出格式唯一事实源，本仓库只记录消费方式，不修改上游契约。
3. Notion 的 `relative-pose-v2.html` 解释 body-frame 相对位姿；最终采用架构 v0.2 指定的输入 Step / 输出 Chunk。旧页面中的“手工设计版”已废弃。

## 原生异步数据的消费

严格读取 `timestamp_ns:int64`，原生 pose `fixed_size_list<float64,12>`，gripper `float32`，按流校验排序、非空、非有限值、SO(3)、范围和文件存在性。额外字段忽略。位姿排列是 `[x,y,z,R00,R10,R20,R01,R11,R21,R02,R12,R22]`，不是行优先，不把旋转当 Euler 或 quaternion。

四路 state 的时间戳取并集，在每个实际到达时刻做 **backward-as-of**：每一路只看当时或之前最后一个样本。不插值、不从行号配对、不使用后到的样本。重复时间取流内最后一条。只有四路都已到达且没有超出 `max_age_ns` 时，才产生一个统一 state 事件；初始缺失期间不以零值代替。该合并是下游模型适配器的明确选择，导出文件仍保持原生异步格式。

每次产生 state 事件时，左右 EEF 分别计算 `T_prev^-1 @ T_now`，prev 是前一个已产生的完整 state 事件。第一次是 identity；夹爪为当前实际 openness，保留绝对 `[0,1]`。某一路未更新时其 EEF delta 为 identity。先计算完整事件 delta，再按明确的采样预算选择事件，不把删帧后的跨步运动误写为原生一步运动。采样后不保证历史 delta 可直接连乘还原完整轨迹，模型仍有每个事件的真实时间。

state 共 26 维，顺序为 `[left_delta_pose12,left_gripper,right_delta_pose12,right_gripper]`。只有一个共享 state MLP，没有本体 embedding 或维度 mask。左右臂分别在自己的 body 系编码，不推断两臂绝对外参/间距。

每个相机独立解码 MP4 的显示顺序，按 Parquet 的 `frame_index` 索引，实际时间只用 `timestamp_ns`。不使用 FPS/PTS 推算时刻。视频必须无音轨，帧数与索引一致。图像在数据层按配置 resize 到固定尺寸、RGB 归一化到 `[0,1]`，模型直接做非重叠 patch embedding；不使用视觉塔、CLS 或压缩器。

instruction 在 `start_ns` 到达时成为一个文本事件；目标 query 之后才开始的指令不输入。不把已知指令的 `end_ns` 当作输入特征，不泄漏未来切换时刻。更早已观察的指令仍属于历史。

## 监督与控制端输出

LOOM-data 不记录 action。每个 future offset 使用该时刻已到达的最近实测状态作监督，并记录显式 staleness/episode-end mask；不会读取指定时刻之后的样本或超出流末尾延拓。左右 EEF 都相对 **同一 query 时刻各自的观测锚点** 编码为 `T_query^-1 @ T_future`，不在 horizon 内累乘动作。

每臂 10 维：`[translation3,rotation_column0_3,rotation_column1_3,openness1]`，双臂总 20 维，左臂在前。用 MSE 对有效字段回归；位置单位为米，旋转是前两列的 6D 表示，夹爪是绝对开合程度。无效字段虽然存储成零，loss 从布尔 mask 选择有效字段，零值不参与监督。没有使用未来动作、未来图像/状态或 teacher forcing 输入。

推理以 Gram-Schmidt 把预测 6D 旋转映射到 SO(3)，退化向量有确定性正交基兜底，夹爪裁剪到 `[0,1]`；再乘对应 EEF 的 query anchor 还原成导出格式绝对 pose12。此处是数学反变换，不包含机器人限位、控制周期、碰撞处理或实际下发。

## 统一 token 与可见性

- 图像：一层 patch Conv2d 等价于共享 patch 线性映射，附 patch 中心 `(x,y)`，`has_xy=1`。
- state：共享两层 MLP，一事件一个 token，`has_xy=0`。
- 文本：UTF-8 字节 embedding，每字节一个 token，加显式 sin/cos 序号以保留字节顺序。无需模型下载，是浅层可替换基线，不声称具备预训练语言理解能力。
- 所有 token 都加同一个元数据 MLP 输出，输入 `[tau,x,y,has_xy,E_segment,E_stream,E_modality]`。
- 目标段 `tau = event_time-query.time`；参考段相对最后可见事件归零。单位为秒，不按段长度归一化，没有固定最大秒数。模型不做 KV cache，避免锚点移动引起旧缓存无效。
- ID 是每样本局部类别标签，stream 仅在段内区分来源。训练默认随机重编号 segment/stream，避免模型依赖固定的“参考=0/目标=1”等标签。类别表容量可配，超出报错，不映射到机器人身份。
- 每个 future offset 产生一个动作查询 token，使用目标 segment、独立 query stream、action modality 和相对未来秒数；只有可学习 query 内容，不包含真实未来动作。

数据适配器先裁掉目标未来，模型公开入口也再次在任意混合层之前裁掉 query 之后的目标事件。参考可完整提供，因为它属于另一个物理轨迹。相同 `source_id` 不能以不同 segment 重复出现；但客户端若谎报来源，模型无法自行识别，调用方必须保留真实物理来源。

过滤后所有 token 进入一个标准全注意力 TransformerEncoder；观测和 query 可相互注意，query 之间也可注意，但没有 future ground truth，所以训练和推理可见信息相同。batch 内不同样本有 padding mask，彼此隔离。无 context/current 专用分支，无视觉/时间专用注意力机制。

## 预算、范围与后续研究

采样器默认每相机 4 帧、32 个 state 事件，均匀选取并保留末端；`null` 可取消对应数量预算。是显式采样策略，不是主干固定时间窗。`max_tokens` 仅作为显存保护上限，超过直接报错，不静默截断文本或参考。模型支持不同 segment 数、不同模态、异步时间和不同 horizon；可靠跨度和跨域能力仍依赖训练数据。

待研究项包括回归头与 flow/diffusion 的比较、语言输入映射、query/observation 注意力形式、源 ID 增广、采样策略、长序列计算、在线缓存以及大规模分布式训练。当前代码把这些选择保持为局部实现，不把尚未验证的假设写成实验结论。
