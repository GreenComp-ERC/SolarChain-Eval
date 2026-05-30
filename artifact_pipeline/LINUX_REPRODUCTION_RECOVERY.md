# Manual Linux Reproduction And Recovery

This guide reproduces and recovers a SolarChain-Eval artifact run using direct Linux commands. It is designed for the case where the convenience pipeline stops at any stage from `[0/9]` to `[9/9]`.

Do not rerun completed training unless its output is missing or known to be invalid. If a stage has partial output, archive the partial directory first, then rerun only that stage.

This guide intentionally uses direct `python scripts/...` commands and inline Python snippets.

## 0. Run Variables

Run all commands from the repository root.

```bash
cd /path/to/SolarChain-Eval

conda activate SolarChain-rl

export RUN_SEED=20260511
export PAPER_RUN_ID=paper_final_seed_${RUN_SEED}

export DATA_DIR=data/datasets_2026_04_month
export DATA_START_DATE=2026-04-01
export DATA_END_DATE=2026-05-01
export DATA_SEED=20260511

export CONFIG=outputs/multiseed_configs/month_2026_04_seed_${RUN_SEED}.yaml
export PAPER_RUN_DIR=outputs/${PAPER_RUN_ID}
export RUNS_DIR=${PAPER_RUN_DIR}/runs

export MAIN_RUN=${RUNS_DIR}/main
export ABLATION_RUN=${RUNS_DIR}/no_physics_penalty
export AGENTIC_RUN=${RUNS_DIR}/agentic_llm_llm
export AGENTIC_ABLATION_RUN=${RUNS_DIR}/agentic_llm_llm_no_physics_penalty

export MAIN_FIGURES_DIR=${PAPER_RUN_DIR}/figures/main
export ABLATION_FIGURES_DIR=${PAPER_RUN_DIR}/figures/ablation_no_physics_penalty
export AGENTIC_FIGURES_DIR=${PAPER_RUN_DIR}/figures/agentic
export AGENTIC_ABLATION_FIGURES_DIR=${PAPER_RUN_DIR}/figures/agentic_no_physics_penalty

export TIMESTEPS=300000
export EPISODES=30

mkdir -p "$PAPER_RUN_DIR" "$RUNS_DIR" \
  "$MAIN_FIGURES_DIR" "$ABLATION_FIGURES_DIR" \
  "$AGENTIC_FIGURES_DIR" "$AGENTIC_ABLATION_FIGURES_DIR"
```

The default values reproduce `paper_final_seed_20260511`. To recover or run another seed, change `RUN_SEED` and optionally `PAPER_RUN_ID` before deriving the paths.

## LLM Endpoint Check

Required for stages `[3/9]` and `[4/9]`.

```bash
python - <<'PY'
import os

print("has_key:", bool(os.getenv("SOLARCHAIN_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")))
print("model:", os.getenv("SOLARCHAIN_LLM_MODEL") or os.getenv("OPENAI_MODEL"))
print("base_url:", os.getenv("SOLARCHAIN_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
PY
```

If `has_key` is false or the model is empty, configure the endpoint before running LLM agentic stages:

```bash
export SOLARCHAIN_LLM_API_KEY="..."
export SOLARCHAIN_LLM_BASE_URL="https://..."
export SOLARCHAIN_LLM_MODEL="..."
```

## Partial Output Rule

Before rerunning a stage that writes to an existing run directory, archive the partial directory:

```bash
if [ -d "$AGENTIC_RUN" ]; then
  mv "$AGENTIC_RUN" "${AGENTIC_RUN}.partial_$(date -u +%Y%m%d_%H%M%S)"
fi
```

Use the same pattern for `$MAIN_RUN`, `$ABLATION_RUN`, or `$AGENTIC_ABLATION_RUN` when needed.

## Checkpoint Matrix

| Stage | Completion evidence | Resume command section |
|---|---|---|
| `[0/9]` dataset/config/metadata preparation | `$CONFIG`, `dataset_summary.json`, `paper_run_metadata.json` | Stage 0 |
| `[1/9]` main six-baseline run | `$MAIN_RUN/summary.json`, PPO/SAC/DQN model files | Stage 1 |
| `[2/9]` no-physics ablation | `$ABLATION_RUN/summary.json`, PPO/SAC/DQN model files | Stage 2 |
| `[3/9]` LLM agentic main evaluation | `$AGENTIC_RUN/summary.json`, `$AGENTIC_RUN/agentic_logs.jsonl` | Stage 3 |
| `[4/9]` LLM agentic no-physics evaluation | `$AGENTIC_ABLATION_RUN/summary.json`, `$AGENTIC_ABLATION_RUN/agentic_logs.jsonl` | Stage 4 |
| `[5/9]` main figures | `$MAIN_FIGURES_DIR/*.png` | Stage 5 |
| `[6/9]` ablation figures | `$ABLATION_FIGURES_DIR/*.png` | Stage 6 |
| `[7/9]` agentic figures | `$AGENTIC_FIGURES_DIR/*.png` | Stage 7 |
| `[8/9]` agentic no-physics figures | `$AGENTIC_ABLATION_FIGURES_DIR/*.png` | Stage 8 |
| `[9/9]` final manifest | `$PAPER_RUN_DIR/PAPER_RESULTS.md` | Stage 9 |

## Stage 0: Dataset, Config, And Metadata

Completion check:

```bash
test -s "$DATA_DIR/spatiotemporal_generation.csv"
test -s "$DATA_DIR/market_liquidity.csv"
test -s "$CONFIG"
test -s "$PAPER_RUN_DIR/dataset_summary.json"
test -s "$PAPER_RUN_DIR/paper_run_metadata.json"
```

Resume command:

```bash
mkdir -p outputs/multiseed_configs "$PAPER_RUN_DIR" "$RUNS_DIR"

if [ ! -s "$CONFIG" ]; then
  cp configs/month_2026_04.yaml "$CONFIG"
  sed -i "s/^seed:.*/seed: ${RUN_SEED}/" "$CONFIG"
fi

if [ ! -s "$DATA_DIR/spatiotemporal_generation.csv" ] || [ ! -s "$DATA_DIR/market_liquidity.csv" ]; then
  python scripts/generate_monthly_datasets.py \
    --start-date "$DATA_START_DATE" \
    --end-date "$DATA_END_DATE" \
    --output-dir "$DATA_DIR" \
    --seed "$DATA_SEED"
fi

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

data_dir = Path(os.environ["DATA_DIR"])
paper_run_dir = Path(os.environ["PAPER_RUN_DIR"])
nodes = pd.read_csv(data_dir / "urban_energy_nodes.csv")
generation = pd.read_csv(data_dir / "spatiotemporal_generation.csv")
market = pd.read_csv(data_dir / "market_liquidity.csv")
trades = pd.read_csv(data_dir / "p2p_trades.csv")
cities = ["Beijing", "Shanghai", "Chengdu", "Shenzhen", "Hangzhou"]
summary = {
    "data_dir": str(data_dir),
    "start_date": os.environ["DATA_START_DATE"],
    "end_date_exclusive": os.environ["DATA_END_DATE"],
    "cities": sorted(generation["city"].unique().tolist()),
    "node_rows": int(len(nodes)),
    "generation_rows": int(len(generation)),
    "unique_timestamps": int(generation["timestamp"].nunique()),
    "market_rows": int(len(market)),
    "trade_rows": int(len(trades)),
    "fdia_rows": int(generation["fdia_detected"].astype(bool).sum()),
}
assert summary["cities"] == sorted(cities), summary
assert summary["node_rows"] == 50, summary
assert summary["generation_rows"] == 36000, summary
assert summary["unique_timestamps"] == 720, summary
assert summary["market_rows"] == 720, summary
paper_run_dir.mkdir(parents=True, exist_ok=True)
(paper_run_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], check=False, capture_output=True, text=True).stdout
except Exception:
    git_commit = ""
    git_status = ""
metadata = {
    "paper_run_id": os.environ["PAPER_RUN_ID"],
    "config": os.environ["CONFIG"],
    "dataset": summary,
    "timesteps": int(os.environ["TIMESTEPS"]),
    "episodes": int(os.environ["EPISODES"]),
    "run_agentic": "1",
    "agentic_policies": "ppo,sac,dqn",
    "agentic_planner": "llm",
    "agentic_auditor": "llm",
    "agentic_audit_trigger": "event",
    "agentic_gap_definition": "(verified_mwh - demand_mwh) / demand_mwh",
    "agentic_gap_threshold_clip": [-1.0, 0.0],
    "paper_run_dir": os.environ["PAPER_RUN_DIR"],
    "runs_dir": os.environ["RUNS_DIR"],
    "main_run": os.environ["MAIN_RUN"],
    "ablation_run": os.environ["ABLATION_RUN"],
    "agentic_run": os.environ["AGENTIC_RUN"],
    "agentic_ablation_run": os.environ["AGENTIC_ABLATION_RUN"],
    "git_commit": git_commit,
    "git_status_short": git_status,
}
(paper_run_dir / "paper_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
```

Validation:

```bash
test -s "$CONFIG"
test -s "$PAPER_RUN_DIR/dataset_summary.json"
test -s "$PAPER_RUN_DIR/paper_run_metadata.json"
```

## Stage 1: Main Six-Baseline Run

Completion check:

```bash
test -s "$MAIN_RUN/summary.json"
test -s "$MAIN_RUN/models/ppo/ppo_model.zip"
test -s "$MAIN_RUN/models/sac/sac_model.zip"
test -s "$MAIN_RUN/models/dqn/dqn_model.zip"
```

If partial, archive:

```bash
if [ -d "$MAIN_RUN" ]; then
  mv "$MAIN_RUN" "${MAIN_RUN}.partial_$(date -u +%Y%m%d_%H%M%S)"
fi
```

Resume command:

```bash
python scripts/run_all_baselines.py \
  --config "$CONFIG" \
  --timesteps "$TIMESTEPS" \
  --episodes "$EPISODES" \
  --output-dir "$RUNS_DIR" \
  --run-name main

echo "$MAIN_RUN" > "$PAPER_RUN_DIR/main_run.txt"
echo "$MAIN_RUN" > "$PAPER_RUN_DIR/latest_main_run.txt"
```

Validation:

```bash
test -s "$MAIN_RUN/summary.json"
test -s "$MAIN_RUN/models/ppo/ppo_model.zip"
test -s "$MAIN_RUN/models/sac/sac_model.zip"
test -s "$MAIN_RUN/models/dqn/dqn_model.zip"
```

## Stage 2: No-Physics Ablation

Completion check:

```bash
test -s "$ABLATION_RUN/summary.json"
test -s "$ABLATION_RUN/models/ppo/ppo_model.zip"
test -s "$ABLATION_RUN/models/sac/sac_model.zip"
test -s "$ABLATION_RUN/models/dqn/dqn_model.zip"
```

If partial, archive:

```bash
if [ -d "$ABLATION_RUN" ]; then
  mv "$ABLATION_RUN" "${ABLATION_RUN}.partial_$(date -u +%Y%m%d_%H%M%S)"
fi
```

Resume command:

```bash
python scripts/run_all_baselines.py \
  --config "$CONFIG" \
  --timesteps "$TIMESTEPS" \
  --episodes "$EPISODES" \
  --output-dir "$RUNS_DIR" \
  --run-name no_physics_penalty \
  --no-physics-penalty

echo "$ABLATION_RUN" > "$PAPER_RUN_DIR/ablation_run.txt"
echo "$ABLATION_RUN" > "$PAPER_RUN_DIR/latest_ablation_run.txt"
```

Validation:

```bash
test -s "$ABLATION_RUN/summary.json"
test -s "$ABLATION_RUN/models/ppo/ppo_model.zip"
test -s "$ABLATION_RUN/models/sac/sac_model.zip"
test -s "$ABLATION_RUN/models/dqn/dqn_model.zip"
```

## Stage 3: LLM Agentic Main Evaluation

Completion check:

```bash
test -s "$AGENTIC_RUN/summary.json"
test -s "$AGENTIC_RUN/agentic_logs.jsonl"
```

If partial, archive:

```bash
if [ -d "$AGENTIC_RUN" ]; then
  mv "$AGENTIC_RUN" "${AGENTIC_RUN}.partial_$(date -u +%Y%m%d_%H%M%S)"
fi
```

Resume command:

```bash
python scripts/evaluate.py \
  --config "$CONFIG" \
  --policies ppo,sac,dqn \
  --episodes "$EPISODES" \
  --ppo-model "$MAIN_RUN/models/ppo/ppo_model.zip" \
  --sac-model "$MAIN_RUN/models/sac/sac_model.zip" \
  --dqn-model "$MAIN_RUN/models/dqn/dqn_model.zip" \
  --output-dir "$RUNS_DIR" \
  --run-name agentic_llm_llm \
  --no-timestamp \
  --agentic-mode planner_auditor \
  --planner llm \
  --auditor llm \
  --audit-trigger event \
  --save-agentic-logs

echo "$AGENTIC_RUN" > "$PAPER_RUN_DIR/agentic_run.txt"
echo "$AGENTIC_RUN" > "$PAPER_RUN_DIR/latest_agentic_run.txt"
```

Validation:

```bash
test -s "$AGENTIC_RUN/summary.json"
test -s "$AGENTIC_RUN/agentic_logs.jsonl"
```

## Stage 4: LLM Agentic No-Physics Evaluation

Completion check:

```bash
test -s "$AGENTIC_ABLATION_RUN/summary.json"
test -s "$AGENTIC_ABLATION_RUN/agentic_logs.jsonl"
```

If partial, archive:

```bash
if [ -d "$AGENTIC_ABLATION_RUN" ]; then
  mv "$AGENTIC_ABLATION_RUN" "${AGENTIC_ABLATION_RUN}.partial_$(date -u +%Y%m%d_%H%M%S)"
fi
```

Resume command:

```bash
python scripts/evaluate.py \
  --config "$CONFIG" \
  --policies ppo,sac,dqn \
  --episodes "$EPISODES" \
  --ppo-model "$ABLATION_RUN/models/ppo/ppo_model.zip" \
  --sac-model "$ABLATION_RUN/models/sac/sac_model.zip" \
  --dqn-model "$ABLATION_RUN/models/dqn/dqn_model.zip" \
  --output-dir "$RUNS_DIR" \
  --run-name agentic_llm_llm_no_physics_penalty \
  --no-timestamp \
  --no-physics-penalty \
  --agentic-mode planner_auditor \
  --planner llm \
  --auditor llm \
  --audit-trigger event \
  --save-agentic-logs

echo "$AGENTIC_ABLATION_RUN" > "$PAPER_RUN_DIR/agentic_ablation_run.txt"
echo "$AGENTIC_ABLATION_RUN" > "$PAPER_RUN_DIR/latest_agentic_ablation_run.txt"
```

Validation:

```bash
test -s "$AGENTIC_ABLATION_RUN/summary.json"
test -s "$AGENTIC_ABLATION_RUN/agentic_logs.jsonl"
```

## Stage 5: Main Figures

Completion check:

```bash
find "$MAIN_FIGURES_DIR" -name "*.png" -type f | grep -q .
```

Resume command:

```bash
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$MAIN_RUN" \
  --figures-dir "$MAIN_FIGURES_DIR"
```

Validation:

```bash
test -s "$MAIN_FIGURES_DIR/learning_curves.png"
test -s "$MAIN_FIGURES_DIR/safety_utility_frontier.png"
test -s "$MAIN_FIGURES_DIR/city_hour_liquidity_heatmap.png"
```

## Stage 6: Ablation Figures

Completion check:

```bash
find "$ABLATION_FIGURES_DIR" -name "*.png" -type f | grep -q .
```

Resume command:

```bash
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$ABLATION_RUN" \
  --figures-dir "$ABLATION_FIGURES_DIR"
```

Validation:

```bash
test -s "$ABLATION_FIGURES_DIR/learning_curves.png"
test -s "$ABLATION_FIGURES_DIR/safety_utility_frontier.png"
test -s "$ABLATION_FIGURES_DIR/city_hour_liquidity_heatmap.png"
```

## Stage 7: Agentic Figures

Completion check:

```bash
find "$AGENTIC_FIGURES_DIR" -name "*.png" -type f | grep -q .
```

Resume command:

```bash
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$AGENTIC_RUN" \
  --figures-dir "$AGENTIC_FIGURES_DIR"
```

Validation:

```bash
test -s "$AGENTIC_FIGURES_DIR/learning_curves.png"
test -s "$AGENTIC_FIGURES_DIR/safety_utility_frontier.png"
test -s "$AGENTIC_FIGURES_DIR/city_hour_liquidity_heatmap.png"
```

## Stage 8: Agentic No-Physics Figures

Completion check:

```bash
find "$AGENTIC_ABLATION_FIGURES_DIR" -name "*.png" -type f | grep -q .
```

Resume command:

```bash
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$AGENTIC_ABLATION_RUN" \
  --figures-dir "$AGENTIC_ABLATION_FIGURES_DIR"
```

Validation:

```bash
test -s "$AGENTIC_ABLATION_FIGURES_DIR/learning_curves.png"
test -s "$AGENTIC_ABLATION_FIGURES_DIR/safety_utility_frontier.png"
test -s "$AGENTIC_ABLATION_FIGURES_DIR/city_hour_liquidity_heatmap.png"
```

## Stage 9: Final Manifest

Completion check:

```bash
test -s "$PAPER_RUN_DIR/PAPER_RESULTS.md"
```

Resume command:

```bash
python - <<'PY'
import os
from pathlib import Path

paper_run_dir = Path(os.environ["PAPER_RUN_DIR"])
main_run = os.environ["MAIN_RUN"]
ablation_run = os.environ["ABLATION_RUN"]
agentic_run = os.environ["AGENTIC_RUN"]
agentic_ablation_run = os.environ["AGENTIC_ABLATION_RUN"]
manifest = paper_run_dir / "PAPER_RESULTS.md"
paper_run_dir.mkdir(parents=True, exist_ok=True)
manifest.write_text(f"""# SolarChain-Eval Paper Results

This manifest summarizes the experiment artifacts for the KDD Workshop on Evaluation and Trustworthiness of Agentic AI submission.

## Unified Output Directory

```text
{paper_run_dir}
```

## Runs

- Main six-baseline run: `{main_run}`
- No-physics-penalty ablation: `{ablation_run}`
- Eval-only LLM agentic run: `{agentic_run}`
- Eval-only LLM agentic no-physics run: `{agentic_ablation_run}`

## Primary Files

- `{paper_run_dir}/dataset_summary.json`
- `{paper_run_dir}/paper_run_metadata.json`
- `{main_run}/summary.json`
- `{ablation_run}/summary.json`
- `{agentic_run}/summary.json`
- `{agentic_run}/agentic_logs.jsonl`
- `{agentic_ablation_run}/summary.json`
- `{agentic_ablation_run}/agentic_logs.jsonl`

## Figures

- `{paper_run_dir}/figures/main/learning_curves.png`
- `{paper_run_dir}/figures/main/safety_utility_frontier.png`
- `{paper_run_dir}/figures/main/city_hour_liquidity_heatmap.png`
- `{paper_run_dir}/figures/ablation_no_physics_penalty/learning_curves.png`
- `{paper_run_dir}/figures/agentic/learning_curves.png`
- `{paper_run_dir}/figures/agentic_no_physics_penalty/learning_curves.png`

## Metrics To Report

- cumulative_reward
- physics_violation_rate
- max_drawdown
- max_token_drawdown
- action_jitter
- mean_slippage
- slippage_reduction_vs_static
- spatial_fairness_index
- artificial_liquidity_MWh

## Agentic Extension Metrics

- plan_validity_rate
- audit_call_rate
- revision_rate
- action_modification_rate
- avg_action_delta_from_auditor
- llm_failure_count
- audit_budget_per_episode
- target_audit_rate
- audit_cooldown_steps
""", encoding="utf-8")
print(f"Wrote {manifest}")
PY
```

Validation:

```bash
test -s "$PAPER_RUN_DIR/PAPER_RESULTS.md"
```

## Final Validation

Run these checks after recovery:

```bash
test -s "$PAPER_RUN_DIR/dataset_summary.json"
test -s "$PAPER_RUN_DIR/paper_run_metadata.json"
test -s "$PAPER_RUN_DIR/PAPER_RESULTS.md"
test -s "$MAIN_RUN/summary.json"
test -s "$ABLATION_RUN/summary.json"
test -s "$AGENTIC_RUN/summary.json"
test -s "$AGENTIC_RUN/agentic_logs.jsonl"
test -s "$AGENTIC_ABLATION_RUN/summary.json"
test -s "$AGENTIC_ABLATION_RUN/agentic_logs.jsonl"
find "$PAPER_RUN_DIR/figures" -name "*.png" -type f
```
