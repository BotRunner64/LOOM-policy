# 直接回归验证记录

日期：2026-10-24（Asia/Shanghai）。复用公共科研项目 / CPU资源空间 的长期 `wbq-dev`，不新增 GPU，也不停止已有实例。

实际镜像：`wbq-dev-base:1.0`；复用现有 `loom-prototype` Python 环境运行，没有修改环境依赖。

## 测试

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
LOOM_DATA_REPO=/inspire/hdd/global_user/czxs253130598/projects/my_projects/LOOM-data \
PYTHONPATH=. \
/inspire/hdd/global_user/czxs253130598/cache/conda/envs/loom-prototype/bin/python \
  -m pytest -q \
  --basetemp=/inspire/hdd/global_user/czxs253130598/cache/loom-policy-pytest
```

验证观测-only token 数、固定 chunk 输出、单次主干前向、padding、当前段池化、参考通过 attention 生效、各模态梯度、L1 mask、未来数据隔离、导出契约、相对几何、checkpoint 恢复与离线预测。上游集成测试需要 `LOOM_DATA_REPO`，未设置时明确 skip。

结果：**23 passed**。CPU 连续训练与中途恢复的全部模型参数逐位一致；预测在 `eval()` 下确定，无采样参数。

## 合成数据命令行闭环

4 个合成 episode 保存于仓库共享盘 `runs/smoke-direct/export`。训练为 `target` + `demo`，验证为 `validation` + `validation_demo`，来源 ID 隔离但运动模式相同，仅检查软件链路。

```bash
python -m loom_policy.index --config runs/smoke-direct/config.json \
  --dataset runs/smoke-direct/export --output runs/smoke-direct/train.jsonl \
  --episodes target --references demo --stride 2
python -m loom_policy.index --config runs/smoke-direct/config.json \
  --dataset runs/smoke-direct/export --output runs/smoke-direct/validation.jsonl \
  --episodes validation --references validation_demo --stride 4
python -m loom_policy.train --config runs/smoke-direct/config.json --steps 40
python -m loom_policy.train --config runs/smoke-direct/config.json \
  --resume runs/smoke-direct/training/last.pt --steps 60
python -m loom_policy.predict --checkpoint runs/smoke-direct/training/last.pt \
  --manifest runs/smoke-direct/validation.jsonl --output runs/smoke-direct/prediction.json
```

实际运行使用上述共享环境 Python 和 `PYTHONPATH=.`。训练5个样本、验证2个样本。D=32、2层、4heads、图像16×16、patch=8、H=4、Δt=50ms，每相机最多2帧、最多8个state事件，batch=2、lr=0.001、seed=7。

第1步训练 L1 `0.339536`，第40步 `0.090834`，恢复到第60步 `0.052065`；按有效字段数归一化的验证 L1 第40/60步为 `0.089652 / 0.042246`。不与其他目标函数的损失作比较。

产物包括当前配置、manifest、metrics、v3 checkpoint 和直接回归预测，均在 `runs/smoke-direct/`（Git 忽略）。当前 v0.3 wheel 构建通过。

## 验证范围

仅软件与合成数据验证；没有真实数据训练、GPU 训练、机器人执行或跨域 ICL 效果结论。
