#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SolarChain-rl

CONFIG="${CONFIG:-configs/default.yaml}"
TIMESTEPS="${TIMESTEPS:-100000}"
EPISODES="${EPISODES:-10}"
mkdir -p outputs/paper_runs figures/main figures/ablation_no_physics_penalty

echo "Running SolarChain-Eval paper experiments"
echo "CONFIG=$CONFIG"
echo "TIMESTEPS=$TIMESTEPS"
echo "EPISODES=$EPISODES"

echo "[1/5] Main six-baseline experiment"
python scripts/run_all_baselines.py \
  --config "$CONFIG" \
  --timesteps "$TIMESTEPS" \
  --episodes "$EPISODES"

MAIN_RUN="$(ls -td outputs/runs/* | head -n 1)"
echo "$MAIN_RUN" > outputs/paper_runs/latest_main_run.txt
echo "Main run: $MAIN_RUN"

echo "[2/5] No-physics-penalty ablation"
python scripts/run_all_baselines.py \
  --config "$CONFIG" \
  --timesteps "$TIMESTEPS" \
  --episodes "$EPISODES" \
  --no-physics-penalty

ABLATION_RUN="$(ls -td outputs/runs/* | head -n 1)"
echo "$ABLATION_RUN" > outputs/paper_runs/latest_ablation_run.txt
echo "Ablation run: $ABLATION_RUN"

echo "[3/5] Main figures"
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$MAIN_RUN" \
  --figures-dir figures/main

echo "[4/5] Ablation figures"
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$ABLATION_RUN" \
  --figures-dir figures/ablation_no_physics_penalty

echo "[5/5] Paper results manifest"
bash paper_pipeline/04_write_results_manifest.sh "$MAIN_RUN" "$ABLATION_RUN"

echo "Paper experiment pipeline complete."
echo "Main run: $MAIN_RUN"
echo "Ablation run: $ABLATION_RUN"
echo "Manifest: outputs/paper_runs/PAPER_RESULTS.md"

