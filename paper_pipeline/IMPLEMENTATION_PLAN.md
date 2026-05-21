# SolarChain-Eval KDD Workshop 主实现方案

## 摘要

`SolarChain-Eval` 面向 KDD Workshop on Evaluation and Trustworthiness of Agentic AI。本文评估的不是“给 SolarChain 加 RL”，而是一个更通用的问题：自治经济治理 agent 能否在没有人工干预的情况下，安全管理一个受物理约束的去中心化能源市场。

主实验 pipeline 使用 2026-04 整个月的五城市数据：

- 数据窗口：`[2026-04-01, 2026-05-01)`
- 城市：Beijing、Shanghai、Chengdu、Shenzhen、Hangzhou
- 数据目录：`data/datasets_2026_04_month`
- 主配置：`configs/month_2026_04.yaml`
- episode 设计：`episode_steps=24`，每次 `reset()` 从一个月内随机采样一个完整日作为起点

`SolarSave` 只作为只读参考源；所有代码、数据副本、缓存、训练结果、图表和论文 manifest 都保存在 `SolarChain-Eval`。

## 当前实现

- 月数据生成：`scripts/generate_monthly_datasets.py` 复用现有五城市 CSV 生成方式，基于 pvlib 太阳物理模型、Open-Meteo historical weather 和 5% FDIA 注入。
- 数据加载：`src/solarchain_eval/data.py` 从 timestamp 派生连续 `absolute_hour`，保留 `hour` 作为小时-of-day，并按 `city + absolute_hour + hour` 聚合，避免把 30 天同一小时错误合并。
- 环境采样：`src/solarchain_eval/env.py` 在 `reset()` 中随机选择 `0, 24, ..., 696` 作为 episode 起点，训练仍然保持单回合 24 步。
- 主 pipeline：`paper_pipeline/02_run_paper_experiments.sh` 默认使用 `configs/month_2026_04.yaml`，会在数据缺失时自动生成并校验月数据。

## Baseline 与指标

必须评估六类 baseline：

- PPO：Stable-Baselines3 连续控制 baseline。
- SAC：Stable-Baselines3 off-policy 连续控制 baseline。
- DQN：Stable-Baselines3 离散控制 baseline，使用 `5 x 5 x 5` 动作网格。
- Static 1:3：原始 SolarChain 固定规则，主比较对象。
- Random Agent：在动作边界内均匀随机采样，作为性能下界。
- Myopic Greedy：最大化即时交易/流动性，不显式考虑未来 drawdown。

核心 trustworthiness 指标：

- `physics_violation_rate`
- `max_drawdown`
- `action_jitter`
- `slippage_reduction_vs_static`
- `spatial_fairness_index`
- `artificial_liquidity_MWh`

关键消融：

- `--no-physics-penalty` 移除 reward 中的 `P_max`/FDIA 惩罚，但继续记录 unsafe backing 和 artificial liquidity，用于证明 benchmark 能暴露不可信 agent 的系统性利用行为。

## 主运行方式

环境准备：

```bash
conda activate SolarChain-rl
pip install -r requirements.txt
pip install -e .
```

生成或刷新 2026-04 月数据：

```bash
python scripts/generate_monthly_datasets.py \
  --start-date 2026-04-01 \
  --end-date 2026-05-01 \
  --output-dir data/datasets_2026_04_month \
  --seed 20260511
```

smoke check：

```bash
bash paper_pipeline/01_smoke_check.sh
```

完整论文实验：

```bash
PAPER_RUN_ID=paper_final TIMESTEPS=300000 EPISODES=30 bash paper_pipeline/02_run_paper_experiments.sh
```

该脚本会执行：

1. 校验或生成 2026-04 五城市月数据。
2. 运行主六 baseline 实验。
3. 运行 no-physics-penalty 消融。
4. 生成主实验和消融图表。
5. 写出 `dataset_summary.json`、`paper_run_metadata.json` 和 `PAPER_RESULTS.md`。

## 输出结构

每个 concrete run 写入：

```text
outputs/runs/<run_id>/
```

包括：

- `metrics.csv`
- `summary.json`
- `actions.csv`
- `city_hour_policy.csv`
- `config_snapshot.json`
- `run_metadata.json`
- `models/<algo>/<algo>_model.zip`

每个 paper batch 写入：

```text
outputs/paper_runs/<paper_run_id>/
```

包括：

- `dataset_summary.json`
- `paper_run_metadata.json`
- `main_run.txt`
- `ablation_run.txt`
- `PAPER_RESULTS.md`
- `figures/main/*.png`
- `figures/ablation_no_physics_penalty/*.png`

## 验收标准

- 月数据仍只包含五个城市。
- `urban_energy_nodes.csv` 有 50 行。
- `spatiotemporal_generation.csv` 有 36000 行。
- `market_liquidity.csv` 有 720 行。
- environment repeated `reset()` 只采样完整日开头：`0,24,...,696`。
- `python -m pytest` 通过。
- `paper_pipeline/01_smoke_check.sh` 通过。
- `paper_pipeline/02_run_paper_experiments.sh` 能输出主实验、消融、图表和 manifest。
