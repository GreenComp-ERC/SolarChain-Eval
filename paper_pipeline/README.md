# SolarChain-Eval KDD Paper Pipeline

This folder contains the main experiment pipeline for the planned KDD Workshop on Evaluation and Trustworthiness of Agentic AI submission. Run all commands from the `SolarChain-Eval` repository root. The expected conda environment is `SolarChain-rl`.

The paper pipeline now uses the 2026-04 monthly five-city dataset as the primary benchmark data:

- Config: `configs/month_2026_04.yaml`
- Dataset: `data/datasets_2026_04_month`
- Window: `[2026-04-01, 2026-05-01)`, 720 hourly timestamps
- Cities: Beijing, Shanghai, Chengdu, Shenzhen, Hangzhou
- Episode design: `episode_steps=24`; each reset samples one complete day from the month

`SolarSave` remains read-only. Dataset generation, caches, runs, figures, and paper manifests all live in `SolarChain-Eval`.

## 1. Setup

First-time setup:

```bash
bash paper_pipeline/00_setup_linux.sh
```

If the conda environment does not exist yet:

```bash
conda create -n SolarChain-rl python=3.10 -y
conda activate SolarChain-rl
pip install -r requirements.txt
pip install -e .
```

## 2. Monthly Data

The smoke and full paper scripts automatically create the monthly dataset if it is missing:

```bash
python scripts/generate_monthly_datasets.py \
  --start-date 2026-04-01 \
  --end-date 2026-05-01 \
  --output-dir data/datasets_2026_04_month \
  --seed 20260511
```

Expected validation values:

- `urban_energy_nodes.csv`: 50 rows
- `spatiotemporal_generation.csv`: 36000 rows
- `market_liquidity.csv`: 720 rows
- unique generation timestamps: 720
- FDIA rows: about 1800

## 3. Smoke Check

Before any long experiment:

```bash
bash paper_pipeline/01_smoke_check.sh
```

The smoke check ensures the monthly dataset exists, validates the five-city/720-hour shape, compiles Python files, evaluates static/random/myopic policies, trains a tiny DQN model, and generates smoke figures.

## 4. Full Paper Experiment

Default run:

```bash
bash paper_pipeline/02_run_paper_experiments.sh
```

Default parameters:

- `CONFIG=configs/month_2026_04.yaml`
- `DATA_DIR=data/datasets_2026_04_month`
- `DATA_START_DATE=2026-04-01`
- `DATA_END_DATE=2026-05-01`
- `TIMESTEPS=100000`
- `EPISODES=10`
- `PAPER_RUN_ID=<utc_timestamp>_paper`

Recommended final paper run:

```bash
PAPER_RUN_ID=paper_final TIMESTEPS=300000 EPISODES=30 bash paper_pipeline/02_run_paper_experiments.sh
```

The script runs:

1. Ensure and validate the 2026-04 monthly five-city dataset.
2. Main six-baseline benchmark: PPO, SAC, DQN, Static 1:3, Random, Myopic Greedy.
3. No-physics-penalty ablation with the same baseline set.
4. Main and ablation figures.
5. `PAPER_RESULTS.md` and `dataset_summary.json` manifests for paper tables and figures.

## 5. Output Layout

Each concrete run goes to `outputs/runs/<run_id>/`:

- `metrics.csv`
- `summary.json`
- `actions.csv`
- `city_hour_policy.csv`
- `config_snapshot.json`
- `run_metadata.json`
- `models/<algo>/<algo>_model.zip`

Each paper batch goes to `outputs/paper_runs/<paper_run_id>/`:

- `paper_run_metadata.json`
- `dataset_summary.json`
- `main_run.txt`
- `ablation_run.txt`
- `PAPER_RESULTS.md`
- `figures/main/*.png`
- `figures/ablation_no_physics_penalty/*.png`

## 6. Hyperparameter Testing

Use a unique `PAPER_RUN_ID` for every parameter test:

```bash
PAPER_RUN_ID=lr_1e-4 TIMESTEPS=100000 EPISODES=10 bash paper_pipeline/02_run_paper_experiments.sh
PAPER_RUN_ID=lr_3e-4 TIMESTEPS=100000 EPISODES=10 bash paper_pipeline/02_run_paper_experiments.sh
PAPER_RUN_ID=gamma_0p995 TIMESTEPS=100000 EPISODES=10 bash paper_pipeline/02_run_paper_experiments.sh
```

If a parameter lives in YAML, copy the monthly config and edit the copy:

```bash
cp configs/month_2026_04.yaml configs/month_2026_04_physics_penalty_4p0.yaml
# Edit reward.physics_penalty = 4.0
PAPER_RUN_ID=month_2026_04_physics_penalty_4p0 CONFIG=configs/month_2026_04_physics_penalty_4p0.yaml bash paper_pipeline/02_run_paper_experiments.sh
```

Recommended low-cost sensitivity checks:

- `reward.physics_penalty`: `1.0`, `2.0`, `4.0`
- `training.learning_rate`: `0.0001`, `0.0003`, `0.001`
- `training.gamma`: `0.95`, `0.98`, `0.995`

## 7. Regenerate Figures

```bash
bash paper_pipeline/03_make_figures_for_run.sh outputs/runs/<run_id> outputs/paper_runs/<paper_run_id>/figures/custom
```

By default, this uses `configs/month_2026_04.yaml`. Override with `CONFIG=<path>` only when regenerating figures for a run produced with a non-default config.

## 8. Paper Tables And Figures

Use the main run `summary.json` for the main comparison table:

- `cumulative_reward`
- `physics_violation_rate`
- `max_drawdown`
- `action_jitter`
- `slippage_reduction_vs_static`
- `spatial_fairness_index`
- `artificial_liquidity_MWh`

Use the main and ablation `summary.json` files to compare whether removing the `P_max` penalty increases:

- `physics_violation_rate`
- `artificial_liquidity_MWh`

Use these figures from each `paper_run_id`:

- `outputs/paper_runs/<paper_run_id>/figures/main/learning_curves.png`
- `outputs/paper_runs/<paper_run_id>/figures/main/safety_utility_frontier.png`
- `outputs/paper_runs/<paper_run_id>/figures/main/city_hour_liquidity_heatmap.png`
- `outputs/paper_runs/<paper_run_id>/figures/ablation_no_physics_penalty/*.png`

## 9. Logging

For long runs, keep terminal logs inside the same paper run directory:

```bash
mkdir -p outputs/paper_runs/paper_final
PAPER_RUN_ID=paper_final TIMESTEPS=300000 EPISODES=30 bash paper_pipeline/02_run_paper_experiments.sh 2>&1 | tee outputs/paper_runs/paper_final/full_run.log
```

## 10. Notes

- The primary KDD paper evidence should come from the monthly 2026-04 pipeline, not `configs/default.yaml`.
- The full run may need network access once to populate the Open-Meteo cache if `data/cache/` is empty.
- Scripts use `set -euo pipefail`, so any failed step stops the pipeline.
- If an SB3 baseline is unstable on a target machine, keep the successful outputs and document the limitation in the paper.
