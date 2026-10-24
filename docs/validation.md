# 随机初始化 ViT / BERT 模态编码器验证记录

日期：2026-10-25（Asia/Shanghai）。复用公共科研项目 / CPU资源空间 的长期 `wbq-dev`，不新增 GPU，也不停止已有实例。

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

新增检查：矩形图像patch序列与二维位置、ViT模态内交互、文本右侧上下文影响、完整字节序列保留、随机初始化可复现且各层独立、局部attention前的预算/长度检查；并验证H个输出token及其位置可区分性、每步一个hidden state、固定 chunk 输出、单次主干前向、变长padding、参考和当前观测通过 attention 生效、各模态梯度、L1 mask、未来数据隔离、导出契约、相对几何、checkpoint 恢复与离线预测。上游集成测试需要 `LOOM_DATA_REPO`，未设置时明确 skip。

结果：**28 passed**。CPU 连续训练与中途恢复的全部模型参数逐位一致；预测在 `eval()` 下确定，无采样参数。

## 合成数据命令行闭环

4 个合成 episode 保存于仓库共享盘 `runs/smoke-encoders/export`。训练为 `target` + `demo`，验证为 `validation` + `validation_demo`，来源 ID 隔离但运动模式相同，仅检查软件链路。

```bash
python -m loom_policy.index --config runs/smoke-encoders/config.json \
  --dataset runs/smoke-encoders/export --output runs/smoke-encoders/train.jsonl \
  --episodes target --references demo --stride 2
python -m loom_policy.index --config runs/smoke-encoders/config.json \
  --dataset runs/smoke-encoders/export --output runs/smoke-encoders/validation.jsonl \
  --episodes validation --references validation_demo --stride 4
python -m loom_policy.train --config runs/smoke-encoders/config.json --steps 40
python -m loom_policy.train --config runs/smoke-encoders/config.json \
  --resume runs/smoke-encoders/training/last.pt --steps 60
python -m loom_policy.predict --checkpoint runs/smoke-encoders/training/last.pt \
  --manifest runs/smoke-encoders/validation.jsonl --output runs/smoke-encoders/prediction.json
```

实际运行使用上述共享环境 Python 和 `PYTHONPATH=.`。训练5个样本、验证2个样本。D=32、2层、4heads、图像16×16、patch=8、H=4、Δt=50ms，每相机最多2帧、最多8个state事件，batch=2、lr=0.001、seed=7。H=4对应4个输出token。ViT与BERT式encoder均2层、D=32、4heads，完整保留patch及字节token；单次预测调用各模态encoder和一次共享主干。

第1步训练L1 `0.607862`，第40步 `0.049017`，恢复到第60步 `0.043269`；验证L1第40/60步为 `0.040392 / 0.045857`。这只是合成数据拟合检查，不说明真实任务效果。

产物包括当前配置、manifest、metrics、v6 checkpoint 和 action token 回归预测，均在 `runs/smoke-encoders/`（Git 忽略）。当前v0.6 wheel构建通过。实际预测探针保留3组图像patch序列（每组[4,32]）与3组文本序列（[11,32]、[18,32]、[11,32]），加state与4个action token后共享主干输入为[1,67,32]，只调用一次共享主干。

## 验证范围

仅软件与合成数据验证；没有真实数据训练、GPU 训练、机器人执行或跨域 ICL 效果结论。
