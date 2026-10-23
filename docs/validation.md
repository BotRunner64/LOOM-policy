# Flow matching 验证记录

日期：2026-10-23（Asia/Shanghai）。验证复用了公共科研项目 / CPU资源空间 的 `wbq-dev` 长期实例，没有申请 GPU 或停止已有实例。

实际镜像：`wbq-dev-base:1.0`。Python 环境：`/inspire/hdd/global_user/czxs253130598/cache/conda/envs/loom-prototype/bin/python`；未修改该环境依赖。Python 3.12.15，PyTorch 2.11.0+cu128（本次只用 CPU），NumPy 2.3.1，PyArrow 23.0.1，PyAV 19.0.2，Pillow 12.2.0。

## 自动化测试

在仓库根运行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
LOOM_DATA_REPO=/inspire/hdd/global_user/czxs253130598/projects/my_projects/LOOM-data \
PYTHONPATH=. \
/inspire/hdd/global_user/czxs253130598/cache/conda/envs/loom-prototype/bin/python \
  -m pytest -q \
  --basetemp=/inspire/hdd/global_user/czxs253130598/cache/loom-policy-pytest
```

若未设置 `LOOM_DATA_REPO`，上游写入器集成测试明确 skip，其余测试独立于数据仓库。

本次结果：**26 passed**。

覆盖：

- 直线条件路径的端点、速度 `u=a-z` 和 oracle 零损失；flow time 与带噪动作确实影响速度预测。
- 无效标签中的 NaN 在进入动作条件之前被清除，不改变有效速度损失；无效字段不产生梯度。
- Euler 积分方向、步数和解析常速度解；固定 seed/initial_noise 的可复现采样；非法时间、horizon、步数和空监督拒绝。
- pose12 列优先旋转、body-frame 世界系不变性、双臂独立 anchor、chunk 共享 anchor、6D 旋转退化时有效 SO(3) 还原。
- 异步原生流、因果 as-of join、重复时间、陈旧状态和 episode 尾部 mask，以及先 delta 后采样。
- MP4 显示帧序与非均匀真实时间，不从容器 FPS 推断时间。
- 目标未来原始位姿不影响历史；目标未来注入 NaN state、越界图像和秘密文本不影响速度或最终 flow 采样。
- 物理来源防重标记、Arrow 类型校验、路径安全、LOOM-data 实际写入器产物兼容。
- variable-length batch padding、所有模态与 flow head 的反向梯度、文本次序。
- 连续训练与中途恢复的模型参数逐位一致，覆盖 optimizer、样本游标、ID/dropout、噪声和 flow time RNG；独立验证 generator 不推进训练 RNG。
- 训练/验证来源隔离、checkpoint v2、预测 CLI seed/flow_steps 和 SO(3) 输出。

## 命令行闭环（合成数据）

当前验证产物在共享盘仓库 `runs/smoke-flow/`。4 个合成 episode 均有四路异步 state、H.264 视频和带区间 instruction。训练用 `target` + `demo`，验证用 `validation` + `validation_demo`，来源 ID 隔离；运动模式简单且相同，不能据此推断泛化效果。

```bash
python -m loom_policy.index --dataset runs/smoke-flow/export \
  --output runs/smoke-flow/train.jsonl --episodes target --references demo \
  --stride 2 --horizon 4 --dt-ns 50000000
python -m loom_policy.index --dataset runs/smoke-flow/export \
  --output runs/smoke-flow/validation.jsonl --episodes validation --references validation_demo \
  --stride 4 --horizon 4 --dt-ns 50000000
python -m loom_policy.train --config runs/smoke-flow/config.json --steps 80
python -m loom_policy.train --config runs/smoke-flow/config.json \
  --resume runs/smoke-flow/training/last.pt --steps 120
python -m loom_policy.predict --checkpoint runs/smoke-flow/training/last.pt \
  --manifest runs/smoke-flow/validation.jsonl --flow-steps 10 --seed 7 \
  --output runs/smoke-flow/prediction.json
```

使用上述 Python 环境和 `PYTHONPATH=.`。训练 5 个 query，验证 2 个 query。模型 D=32、2 层、4 heads、patch=8，图像 16×16，每相机最多 2 帧、最多 8 个 state，batch=2、lr=0.001、seed=7。推理使用 10 步 Euler。

训练和验证指标都是 conditional velocity MSE，验证用固定 seed 的噪声与 flow time；不是动作重建 MSE 或任务成功率。每步训练采样不同噪声与时间，损失不保证单调。

本次训练第 1 步 loss `1.244683`，第 80 步 `0.843159`，恢复后第 120 步 `0.631936`；固定随机数的验证速度 MSE 在第 80/120 步为 `0.914929 / 0.856639`。这些数值只用于检查训练链路。

`runs/smoke-flow/` 包含配置、manifest、metrics、v2 checkpoint、训练/恢复日志和 flow 预测。这些是软件验证产物，不是可部署的策略权重。

CLI 采样确认输出为有限 `[4,20]`，使用 10 步 Euler、seed=7，左右绝对位姿均通过 SO(3) 检查。v0.2.0 wheel 构建通过。

## 范围

未运行真实数据训练、GPU 训练、机器人执行或跨域 ICL 评测，没有真实任务收敛或成功率结论。
