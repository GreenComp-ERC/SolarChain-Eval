#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SolarChain-rl

CONFIG="${CONFIG:-configs/month_2026_04.yaml}"
DATA_DIR="${DATA_DIR:-data/datasets_2026_04_month}"
DATA_START_DATE="${DATA_START_DATE:-2026-04-01}"
DATA_END_DATE="${DATA_END_DATE:-2026-05-01}"
DATA_SEED="${DATA_SEED:-20260511}"
TIMESTEPS="${TIMESTEPS:-100000}"
EPISODES="${EPISODES:-10}"
PAPER_RUN_ID="${PAPER_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_paper}"
PAPER_RUN_DIR="outputs/paper_runs/$PAPER_RUN_ID"
MAIN_RUN_NAME="${PAPER_RUN_ID}_main"
ABLATION_RUN_NAME="${PAPER_RUN_ID}_no_physics_penalty"
MAIN_FIGURES_DIR="$PAPER_RUN_DIR/figures/main"
ABLATION_FIGURES_DIR="$PAPER_RUN_DIR/figures/ablation_no_physics_penalty"
mkdir -p "$PAPER_RUN_DIR" "$MAIN_FIGURES_DIR" "$ABLATION_FIGURES_DIR"

echo "Running SolarChain-Eval paper experiments"
echo "PAPER_RUN_ID=$PAPER_RUN_ID"
echo "PAPER_RUN_DIR=$PAPER_RUN_DIR"
echo "CONFIG=$CONFIG"
echo "DATA_DIR=$DATA_DIR"
echo "DATA_WINDOW=[$DATA_START_DATE, $DATA_END_DATE)"
echo "TIMESTEPS=$TIMESTEPS"
echo "EPISODES=$EPISODES"

echo "[0/6] Ensure 2026-04 five-city monthly dataset"
if [[ ! -s "$DATA_DIR/spatiotemporal_generation.csv" || ! -s "$DATA_DIR/market_liquidity.csv" ]]; then
  python scripts/generate_monthly_datasets.py \
    --start-date "$DATA_START_DATE" \
    --end-date "$DATA_END_DATE" \
    --output-dir "$DATA_DIR" \
    --seed "$DATA_SEED"
fi

DATASET_SUMMARY="$(python - <<PY
import json
from pathlib import Path
import pandas as pd

data_dir = Path("$DATA_DIR")
nodes = pd.read_csv(data_dir / "urban_energy_nodes.csv")
generation = pd.read_csv(data_dir / "spatiotemporal_generation.csv")
market = pd.read_csv(data_dir / "market_liquidity.csv")
trades = pd.read_csv(data_dir / "p2p_trades.csv")
cities = ["Beijing", "Shanghai", "Chengdu", "Shenzhen", "Hangzhou"]
summary = {
    "data_dir": str(data_dir),
    "start_date": "$DATA_START_DATE",
    "end_date_exclusive": "$DATA_END_DATE",
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
print(json.dumps(summary, indent=2))
PY
)"
echo "$DATASET_SUMMARY" > "$PAPER_RUN_DIR/dataset_summary.json"
echo "$DATASET_SUMMARY"

cat > "$PAPER_RUN_DIR/paper_run_metadata.json" <<EOF
{
  "paper_run_id": "$PAPER_RUN_ID",
  "created_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": "$CONFIG",
  "dataset": $DATASET_SUMMARY,
  "timesteps": $TIMESTEPS,
  "episodes": $EPISODES,
  "main_run_name": "$MAIN_RUN_NAME",
  "ablation_run_name": "$ABLATION_RUN_NAME",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || true)",
  "git_status_short": $(python - <<'PY'
import json
import subprocess

try:
    status = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True).stdout
except Exception:
    status = ""
print(json.dumps(status))
PY
)
}
EOF

echo "[1/6] Main six-baseline experiment"
python scripts/run_all_baselines.py \
  --config "$CONFIG" \
  --timesteps "$TIMESTEPS" \
  --episodes "$EPISODES" \
  --run-name "$MAIN_RUN_NAME"

MAIN_RUN="outputs/runs/$MAIN_RUN_NAME"
echo "$MAIN_RUN" > "$PAPER_RUN_DIR/main_run.txt"
echo "$MAIN_RUN" > outputs/paper_runs/latest_main_run.txt
echo "Main run: $MAIN_RUN"

echo "[2/6] No-physics-penalty ablation"
python scripts/run_all_baselines.py \
  --config "$CONFIG" \
  --timesteps "$TIMESTEPS" \
  --episodes "$EPISODES" \
  --run-name "$ABLATION_RUN_NAME" \
  --no-physics-penalty

ABLATION_RUN="outputs/runs/$ABLATION_RUN_NAME"
echo "$ABLATION_RUN" > "$PAPER_RUN_DIR/ablation_run.txt"
echo "$ABLATION_RUN" > outputs/paper_runs/latest_ablation_run.txt
echo "Ablation run: $ABLATION_RUN"

echo "[3/6] Main figures"
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$MAIN_RUN" \
  --figures-dir "$MAIN_FIGURES_DIR"

echo "[4/6] Ablation figures"
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$ABLATION_RUN" \
  --figures-dir "$ABLATION_FIGURES_DIR"

echo "[5/6] Paper results manifest"
bash paper_pipeline/04_write_results_manifest.sh "$MAIN_RUN" "$ABLATION_RUN" "$PAPER_RUN_DIR"

echo "[6/6] Paper pipeline complete"

echo "Paper experiment pipeline complete."
echo "Paper run directory: $PAPER_RUN_DIR"
echo "Main run: $MAIN_RUN"
echo "Ablation run: $ABLATION_RUN"
echo "Manifest: $PAPER_RUN_DIR/PAPER_RESULTS.md"

