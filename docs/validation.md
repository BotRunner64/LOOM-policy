# 验证记录

日期：2026-10-23（Asia/Shanghai）。本次验证复用了公共科研项目 / CPU资源空间 的 `wbq-dev` 长期实例，没有申请 GPU 或停止已有实例。

实际镜像：`wbq-dev-base:1.0`。复用的 Python 环境：`/inspire/hdd/global_user/czxs253130598/cache/conda/envs/loom-prototype/bin/python`；没有修改该环境中的依赖。Python 3.12.15，PyTorch 2.11.0+cu128（本次只用 CPU），NumPy 2.3.1，PyArrow 23.0.1，PyAV 19.0.2，Pillow 12.2.0。

## 自动化测试

在仓库根运行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
LOOM_DATA_REPO=/inspire/hdd/global_user/czxs253130598/projects/my_projects/LOOM-data \
PYTHONPATH=. \
/inspire/hdd/global_user/czxs253130598/cache/conda/envs/loom-prototype/bin/python \
  -m pytest -q
```

结果：**18 passed**。若未设置 `LOOM_DATA_REPO`，上游集成测试明确 skip，其余测试独立于数据仓库。

覆盖：

- pose12 列优先旋转与 body-frame SE(3) 变换；改变外部世界系不改变 delta。
- 双臂分别使用自己的 anchor；horizon 共享 query anchor，不累乘预测误差；退化 6D 旋转还原为有效 SO(3)。
- 异步原生流、因果 as-of join、重复时间最后到达样本、陈旧状态、episode 尾部监督 mask，以及先 delta 后采样。
- MP4 显示帧序与非均匀真实时间对应，不从 30 FPS 容器时序推断采集时间。
- 修改目标未来原始位姿不会改变历史 state；给目标未来注入 NaN state、越界图像和秘密文本不会改变模型预测。
- 拒绝将同一物理源重新标记成参考，拒绝错误 Arrow 类型和 episode 路径逃逸。
- variable-length batch padding 与 singleton 预测一致；图像、state、text 和 metadata 均有反向梯度；文本次序有效。
- checkpoint 恢复优化器、样本游标、epoch、随机 ID/dropout 的 RNG；CPU 上连续训练和中途恢复的全部参数逐位一致。
- 训练/验证来源重叠拒绝；实际 LOOM-data `write_state` / `write_video_index` 产物可直接读取。

## 命令行闭环（合成数据）

在共享盘仓库 `runs/smoke/` 生成 4 个独立 ID 的合成 episode，每集四路异步 state、一段 H.264 视频和带区间 instruction。训练包含 `target` + `demo`，验证包含 `validation` + `validation_demo`，身份完全隔离；运动模式刻意简单且相同，因此不能据此推断泛化性能。

```bash
python -m loom_policy.index --dataset runs/smoke/export \
  --output runs/smoke/train.jsonl --episodes target --references demo \
  --stride 2 --horizon 4 --dt-ns 50000000
python -m loom_policy.index --dataset runs/smoke/export \
  --output runs/smoke/validation.jsonl --episodes validation --references validation_demo \
  --stride 4 --horizon 4 --dt-ns 50000000
python -m loom_policy.train --config runs/smoke/config.json --steps 20
python -m loom_policy.train --config runs/smoke/config.json \
  --resume runs/smoke/training/last.pt --steps 30
python -m loom_policy.predict --checkpoint runs/smoke/training/last.pt \
  --manifest runs/smoke/validation.jsonl --output runs/smoke/prediction.json
```

使用同一上述 Python 环境和 `PYTHONPATH=.`。训练 5 个 query，验证 2 个 query。模型 D=32、2 层、4 heads、patch=8，图像 16×16，每相机最多 2 帧、最多 8 个 state，batch=2、lr=0.001、seed=7。

记录：第 1 步训练 loss `0.486662`；第 20 步 `0.014646`；恢复后第 30 步 `0.018078`。验证 loss 第 10/20/30 步分别为 `0.063871 / 0.017731 / 0.006317`。不同 step 的样本和随机 ID 不同，训练 loss 不保证单调。

产物保存在共享盘仓库（Git 忽略）：`runs/smoke/config.json`、`train.jsonl`、`validation.jsonl`、`training/metrics.jsonl`、`training/last.pt`、`prediction.json`。这些是软件验证产物，不是可部署的策略权重。

另通过 `python -m pip wheel --no-deps --no-build-isolation --wheel-dir runs/wheels .` 验证打包，无需向现有环境安装或改变依赖。

## 范围

未找到本次共享根下可直接用于正式训练的真实导出 `episodes.jsonl`，因此没有运行真实数据训练、GPU 训练、机器人执行或跨域 ICL 评测。没有完成收敛或成功率结论。正式训练前应提供按上述契约导出的数据目录，并固定 episode 划分、参考构造、采样及评价方案。
