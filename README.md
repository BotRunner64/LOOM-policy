# LOOM-policy

多 domain co-train 的统一策略：部署时参数冻结，只靠参考 segment 适配。设计见 Notion「LOOM / policy」页，数据只读 [LOOM-data 导出契约](https://github.com/BotRunner64/LOOM-data/blob/main/docs/contract/export-contract.md)。

1. 每个 segment（参考 / 当前）过同一个 reader（Perceiver IO），读成 L 个 latent。
2. 所有 latent + 50 个动作槽 → 共享 transformer。
3. 动作槽直接回归未来 50 帧，L1。

## 环境

```bash
conda create -n loom-policy --override-channels -c conda-forge python=3.11 pip
```

```bash
conda activate loom-policy && pip install -e ".[dev]" && pytest
```

## 使用

训练。`--data` 可以是数据集目录、源目录或导出根目录；`--output` 下已有 `last.pt` 时自动续训：

```bash
loom-train --data <数据目录>... --output runs/<name> --config configs/base.json
```

多卡：

```bash
torchrun --nproc_per_node 8 -m loom_policy.train --data <数据目录>... --output runs/<name> --config configs/base.json
```

离线评估，同一个模型只换参考数量：

```bash
loom-eval --checkpoint runs/<name>/last.pt --data <数据目录>... --refs 0 1 2
```

部署用 `loom_policy.policy.Policy`：`set_references` 把参考编码一次，`act` 每步返回未来 50 帧的目标位姿。

配置项都在 `loom_policy/config.py`，用 `--set key=value` 覆盖。
