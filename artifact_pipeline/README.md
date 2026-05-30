# SolarChain-Eval Artifact Pipeline

This directory contains the official artifact workflow for **SolarChain-Eval: A Physics-Constrained Benchmark for Trustworthy Economic Agents in Decentralized Energy Markets**.

The root `README.md` gives a concise project overview. This file is the source of truth for experiment execution, output layout, and artifact-to-paper mapping.

## Benchmark Story

SolarChain-Eval evaluates autonomous economic governors for decentralized peer-to-peer solar energy markets. The agent controls hourly reward allocation, liquidity provisioning, and token burns. The evaluation goes beyond reward maximization by measuring whether an agent respects physical constraints, avoids artificial liquidity, maintains market stability, and distributes rewards fairly across urban energy nodes.

The main risks are cyber-physical: a reward-maximizing policy can learn to back invalid or physically impossible generation, especially when the physics-informed reward penalty is removed. This is tested directly with the no-physics-penalty ablation.

## Research Questions

**RQ1:** How do RL agents balance economic utility and trustworthiness metrics compared to static and heuristic baselines in a decentralized energy market?

**RQ2:** To what extent do reward-maximizing agents exploit invalid generation to artificially inflate market liquidity when physics-informed penalties are removed?

**RQ3:** How do agents' dynamic governance decisions on reward allocation, liquidity provisioning, and token burns affect market stability, action reliability, and spatial fairness across urban energy nodes?

## Directory Contents

```text
artifact_pipeline/
  00_setup_linux.sh
  01_smoke_check.sh
  02_run_paper_experiments.sh
  03_make_figures_for_run.sh
  04_write_results_manifest.sh
  README.md
  REPRODUCIBILITY_PLAN.md
  LINUX_REPRODUCTION_RECOVERY.md
```

The shell scripts are convenience automation. `LINUX_REPRODUCTION_RECOVERY.md` provides direct Linux commands for reproducing and recovering any run configured through the variables at the top of that document. Its defaults reproduce `paper_final_seed_20260511`.

## Environment

```bash
conda create -n SolarChain-rl python=3.10 -y
conda activate SolarChain-rl
python -m pip install -r requirements.txt
python -m pip install -e .
```

LLM evaluations require a real OpenAI-compatible endpoint:

```bash
export SOLARCHAIN_LLM_API_KEY="..."
export SOLARCHAIN_LLM_BASE_URL="https://..."
export SOLARCHAIN_LLM_MODEL="..."
```

OpenAI-compatible fallback variables are also supported:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://..."
export OPENAI_MODEL="..."
```

If `--planner llm` or `--auditor llm` is enabled and the endpoint is missing or incompatible with structured outputs, evaluation fails instead of silently producing substitute results.

## Official Pipeline

The main convenience command is:

```bash
PAPER_RUN_ID=paper_final_seed_20260511 \
TIMESTEPS=300000 \
EPISODES=30 \
RUN_AGENTIC=1 \
AGENTIC_POLICIES=ppo,sac,dqn \
AGENTIC_PLANNER=llm \
AGENTIC_AUDITOR=llm \
AGENTIC_AUDIT_TRIGGER=event \
bash artifact_pipeline/02_run_paper_experiments.sh
```

For a quick Linux check:

```bash
bash artifact_pipeline/01_smoke_check.sh
```

## Pipeline Stages

The full pipeline executes:

```text
[0/9] dataset/config/metadata preparation
[1/9] main six-baseline run
[2/9] no-physics ablation
[3/9] LLM agentic main evaluation
[4/9] LLM agentic no-physics evaluation
[5/9] main figures
[6/9] ablation figures
[7/9] agentic figures
[8/9] agentic no-physics figures
[9/9] final manifest
```

If a run stops at any stage, use `LINUX_REPRODUCTION_RECOVERY.md` to resume from the last completed checkpoint without rerunning completed training.

## Output Layout

All artifacts for one paper run are written under:

```text
outputs/<PAPER_RUN_ID>/
```

For `PAPER_RUN_ID=paper_final_seed_20260511`:

```text
outputs/paper_final_seed_20260511/
  dataset_summary.json
  paper_run_metadata.json
  main_run.txt
  ablation_run.txt
  agentic_run.txt
  agentic_ablation_run.txt
  PAPER_RESULTS.md
  runs/
    main/
    no_physics_penalty/
    agentic_llm_llm/
    agentic_llm_llm_no_physics_penalty/
  figures/
    main/
    ablation_no_physics_penalty/
    agentic/
    agentic_no_physics_penalty/
```

## Table Mapping

Use `summary.json` and `metrics.csv` from each run for policy-level tables.

Primary table metrics:

- `cumulative_reward`
- `physics_violation_rate`
- `max_drawdown`
- `max_token_drawdown`
- `action_jitter`
- `mean_slippage`
- `slippage_reduction_vs_static`
- `spatial_fairness_index`
- `artificial_liquidity_MWh`

Agentic extension metrics:

- `plan_validity_rate`
- `audit_call_rate`
- `revision_rate`
- `action_modification_rate`
- `avg_action_delta_from_auditor`
- `llm_failure_count`
- `audit_budget_per_episode`
- `target_audit_rate`
- `audit_cooldown_steps`

## Figure Mapping

Each figure directory contains:

- `learning_curves.png`: reward by episode and policy.
- `safety_utility_frontier.png`: physics violation rate versus cumulative reward.
- `city_hour_liquidity_heatmap.png`: liquidity policy over city-hour records.

Recommended paper mapping:

- Main benchmark figure: `figures/main/`
- Reward-misspecification ablation: `figures/ablation_no_physics_penalty/`
- Eval-only agentic extension: `figures/agentic/`
- Agentic no-physics comparison: `figures/agentic_no_physics_penalty/`

## Acceptance Checks

```bash
python -m pytest -q
test -s outputs/paper_final_seed_20260511/PAPER_RESULTS.md
test -s outputs/paper_final_seed_20260511/runs/main/models/ppo/ppo_model.zip
test -s outputs/paper_final_seed_20260511/runs/main/models/sac/sac_model.zip
test -s outputs/paper_final_seed_20260511/runs/main/models/dqn/dqn_model.zip
test -s outputs/paper_final_seed_20260511/runs/agentic_llm_llm/agentic_logs.jsonl
find outputs/paper_final_seed_20260511/figures -name "*.png" -type f
```
