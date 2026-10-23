# 启智共享资产

## 项目与路径

适用 Project：**公共科研项目**。Workspace 与 Project 分开选择。

| 资产 | 共享路径 | 用途 |
|---|---|---|
| 共享根 | `/inspire/hdd/global_user/czxs253130598` | 所有代码、环境、数据、checkpoint 和日志的存储根 |
| 策略仓库 | `/inspire/hdd/global_user/czxs253130598/projects/my_projects/LOOM-policy` | 本仓库 |
| 数据仓库 | `/inspire/hdd/global_user/czxs253130598/projects/my_projects/LOOM-data` | 正式导出契约 `docs/contract/export-contract.md` |
| 策略运行产物 | 本仓库 `runs/<run-name>/`（Git 忽略） | 配置、manifest、metrics、checkpoint 与离线预测；位于共享盘，可跨实例恢复 |

## 使用约定

- CPU 调试先用 live 查询查找并复用长期 CPU 实例；不要在临时资源清理中停止或删除长期实例。
- 训练输出必须在共享根下，优先低优先级 GPU Job，采用 `--resume` 恢复。调试用普通优先级。
- 创建任何 workload 都显式指定 workspace、project、group、quota、image，使用 CLI 实时资源查询；镜像优先最新可用 `wbq-dev` 系列。实际实验镜像写入该实验记录，不把即时资源状态写在此处。
- CPU、4090 调试区、H200 分布式训练区选择遵循用户工作约定。先检查可用性与实际费用，不依赖历史配额或镜像状态。
- 导出契约必须从上述 LOOM-data 文件读取；源数据集自己的 schema 不能直接输入策略。本仓库不修改数据清理仓库的工作树。
