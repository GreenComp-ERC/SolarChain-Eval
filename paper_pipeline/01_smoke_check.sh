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

mkdir -p outputs/paper_runs

echo "[1/5] Ensure 2026-04 five-city monthly dataset"
if [[ ! -s "$DATA_DIR/spatiotemporal_generation.csv" || ! -s "$DATA_DIR/market_liquidity.csv" ]]; then
  python scripts/generate_monthly_datasets.py \
    --start-date "$DATA_START_DATE" \
    --end-date "$DATA_END_DATE" \
    --output-dir "$DATA_DIR" \
    --seed "$DATA_SEED"
fi

python - <<PY
from pathlib import Path
import pandas as pd

data_dir = Path("$DATA_DIR")
nodes = pd.read_csv(data_dir / "urban_energy_nodes.csv")
generation = pd.read_csv(data_dir / "spatiotemporal_generation.csv")
market = pd.read_csv(data_dir / "market_liquidity.csv")
cities = ["Beijing", "Shanghai", "Chengdu", "Shenzhen", "Hangzhou"]
assert sorted(generation["city"].unique()) == sorted(cities), sorted(generation["city"].unique())
assert len(nodes) == 50, len(nodes)
assert generation["timestamp"].nunique() == 720, generation["timestamp"].nunique()
assert len(generation) == 36000, len(generation)
assert len(market) == 720, len(market)
print("Dataset OK:", data_dir)
PY

echo "[2/5] Python compile check"
python -m compileall -q src scripts tests

echo "[3/5] Built-in policy evaluation smoke"
python scripts/evaluate.py \
  --config "$CONFIG" \
  --policies "static,random,myopic" \
  --episodes 1 \
  --run-name smoke_month_2026_04_builtin

SMOKE_RUN="$(ls -td outputs/runs/*_smoke_month_2026_04_builtin | head -n 1)"
echo "$SMOKE_RUN" > outputs/paper_runs/latest_smoke_run.txt

echo "[4/5] DQN training smoke"
python scripts/train.py \
  --config "$CONFIG" \
  --algo dqn \
  --timesteps 5 \
  --run-name smoke_month_2026_04_dqn

echo "[5/5] Figure generation smoke"
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$SMOKE_RUN" \
  --figures-dir outputs/paper_runs/smoke/figures

echo "Smoke check passed."
echo "Config: $CONFIG"
echo "Dataset: $DATA_DIR"
echo "Smoke run: $SMOKE_RUN"
echo "Smoke figures: outputs/paper_runs/smoke/figures"

