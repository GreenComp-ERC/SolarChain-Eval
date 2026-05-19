# Linux 论文实验流程

本文件夹提供在 Linux 环境中复现实验、生成论文数据和图片的 bash 流水线。所有命令默认在 `SolarChain-Eval` 仓库根目录执行，并使用 conda 环境 `SolarChain-rl`。

## 0. 目录与产物

实验会写入以下目录：

- `outputs/runs/<timestamp>/`：主实验或消融实验的完整运行目录。
- `outputs/paper_runs/latest_main_run.txt`：最近一次主实验目录路径。
- `outputs/paper_runs/latest_ablation_run.txt`：最近一次 `no_physics_penalty` 消融目录路径。
- `outputs/paper_runs/PAPER_RESULTS.md`：自动生成的结果索引与论文取数说明。
- `figures/main/`：主实验三张论文图。
- `figures/ablation_no_physics_penalty/`：无物理惩罚消融图。

每个 run 目录包含：

- `metrics.csv`：每个 policy/episode 的指标。
- `summary.json`：按 policy 汇总的指标。
- `actions.csv`：每小时动作、流动性、滑点、物理违规等轨迹。
- `city_hour_policy.csv`：城市-小时维度的 liquidity split 和 city reward。
- `config_snapshot.json`：本次运行配置。
- `models/ppo/ppo_model.zip`、`models/sac/sac_model.zip`、`models/dqn/dqn_model.zip`：主实验中训练出的 RL 模型。

## 1. 环境准备

首次运行：

```bash
bash paper_pipeline/00_setup_linux.sh
```

该脚本会：

- 激活 `SolarChain-rl`。
- 安装本仓库为 editable package。
- 打印 Python、Gymnasium、Stable-Baselines3 版本。

如果 Linux 机器还没有 conda 环境，请先手动创建：

```bash
conda create -n SolarChain-rl python=3.10 -y
conda activate SolarChain-rl
pip install -r requirements.txt
```

## 2. 快速检查

正式长实验前运行：

```bash
bash paper_pipeline/01_smoke_check.sh
```

该脚本会执行短评估、短 DQN 训练、图表生成和 Python 编译检查。通过后说明数据路径、CLI、Gymnasium 环境和图表脚本都能正常工作。

## 3. 生成完整论文数据

推荐正式命令：

```bash
bash paper_pipeline/02_run_paper_experiments.sh
```

默认参数：

- `TIMESTEPS=100000`
- `EPISODES=10`
- `CONFIG=configs/default.yaml`

可覆盖参数，例如：

```bash
TIMESTEPS=300000 EPISODES=20 bash paper_pipeline/02_run_paper_experiments.sh
```

该脚本会依次运行：

1. 主实验：PPO、SAC、DQN、Static 1:3、Random、Myopic Greedy。
2. 关键消融：同样六个 baseline，但启用 `--no-physics-penalty`。
3. 主实验图表：learning curves、安全-收益前沿、城市-小时 liquidity split 对比热力图。
4. 消融图表。
5. 结果索引：写入 `outputs/paper_runs/PAPER_RESULTS.md`。

## 4. 单独重新生成图表

如果已有 run 目录，只想重画图：

```bash
bash paper_pipeline/03_make_figures_for_run.sh outputs/runs/<run_id> figures/<figure_dir>
```

例子：

```bash
bash paper_pipeline/03_make_figures_for_run.sh outputs/runs/20260519_120000 figures/main
```

## 5. 论文中使用哪些结果

建议论文表格从主实验的 `summary.json` 和 `metrics.csv` 中取：

- cumulative reward
- physics violation rate
- max drawdown
- action jitter
- slippage reduction vs static
- spatial fairness index
- artificial liquidity MWh

建议论文消融段落对比：

- 主实验 `summary.json`
- 消融实验 `summary.json`
- 两者的 `physics_violation_rate` 与 `artificial_liquidity_MWh`

建议论文图片使用：

- `figures/main/learning_curves.png`
- `figures/main/safety_utility_frontier.png`
- `figures/main/city_hour_liquidity_heatmap.png`
- 如需展示消融，则使用 `figures/ablation_no_physics_penalty/` 中对应图片。

## 6. 复现实验注意事项

- `SolarSave` 目录不参与运行，不应被修改。
- Linux 上所有脚本都使用 `set -euo pipefail`，任一步失败会立即停止。
- 如果 Stable-Baselines3 某个 baseline 在目标机器上不稳定，保留成功运行的结果，并在论文中说明实现限制；benchmark 目标是可信评估，不是生产级 RL 部署。
- 长实验建议保留终端日志：

```bash
TIMESTEPS=300000 EPISODES=20 bash paper_pipeline/02_run_paper_experiments.sh 2>&1 | tee outputs/paper_runs/full_run.log
```

