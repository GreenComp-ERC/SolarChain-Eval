# SolarChain-Eval Reproducibility Plan

This document defines the reproducible artifact plan for SolarChain-Eval. It replaces internal planning notes with an English, reviewer-facing guide focused on exact configurations, run layout, metrics, and acceptance criteria.

## Goal

SolarChain-Eval evaluates autonomous market administrators in decentralized solar energy markets. The benchmark tests whether RL agents can improve economic utility while remaining trustworthy under cyber-physical constraints.

The benchmark answers:

- **RQ1:** How do RL agents balance economic utility and trustworthiness metrics compared to static and heuristic baselines in a decentralized energy market?
- **RQ2:** To what extent do reward-maximizing agents exploit invalid generation to artificially inflate market liquidity when physics-informed penalties are removed?
- **RQ3:** How do agents' dynamic governance decisions on reward allocation, liquidity provisioning, and token burns affect market stability, action reliability, and spatial fairness across urban energy nodes?

## Canonical Configuration

Primary config:

```text
configs/month_2026_04.yaml
```

Primary dataset:

```text
data/datasets_2026_04_month/
```

Canonical hosted dataset:

```text
https://huggingface.co/datasets/ThomasXu/solarchain-eval
```

The GitHub repo keeps a small data mirror for convenience. The Hugging Face Dataset repo is the reviewer-facing dataset source and includes `dataset_summary.json`, `checksums.sha256`, the main monthly CSVs, the smoke CSVs, and the Open-Meteo cache.

Public code repository:

```text
https://github.com/GreenComp-ERC/SolarChain-Eval
```

Dataset window:

```text
[2026-04-01, 2026-05-01)
```

Expected data shape:

- five cities: Beijing, Shanghai, Chengdu, Shenzhen, Hangzhou
- 50 PV nodes
- 720 hourly timestamps
- 36,000 generation rows
- 720 market rows

## Policies

Formal comparison policies:

- Static 1:3 split
- Random
- Myopic Greedy
- PPO
- SAC
- DQN

RL policies are trained with Stable-Baselines3. DQN uses a discretized action grid.

## Main Runs

Each formal run writes to:

```text
outputs/<PAPER_RUN_ID>/
```

Recommended primary run:

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

This produces:

```text
runs/main/
runs/no_physics_penalty/
runs/agentic_llm_llm/
runs/agentic_llm_llm_no_physics_penalty/
```

The first two runs address RQ1 and RQ2. The agentic runs are an evaluation-only extension for planning and auditing trained economic governors.

## Multi-Seed Plan

Use the same dataset seed for the monthly CSVs, then vary the top-level benchmark seed for RL initialization and sampled evaluation days.

Recommended seeds:

```text
20260511
20260512
20260513
```

Prepare seed-specific configs:

```bash
mkdir -p outputs/multiseed_configs
cp configs/month_2026_04.yaml outputs/multiseed_configs/month_2026_04_seed_20260511.yaml
sed -i "s/^seed:.*/seed: 20260511/" outputs/multiseed_configs/month_2026_04_seed_20260511.yaml
cp configs/month_2026_04.yaml outputs/multiseed_configs/month_2026_04_seed_20260512.yaml
sed -i "s/^seed:.*/seed: 20260512/" outputs/multiseed_configs/month_2026_04_seed_20260512.yaml
cp configs/month_2026_04.yaml outputs/multiseed_configs/month_2026_04_seed_20260513.yaml
sed -i "s/^seed:.*/seed: 20260513/" outputs/multiseed_configs/month_2026_04_seed_20260513.yaml
```

For each seed, run the pipeline with:

```bash
CONFIG=outputs/multiseed_configs/month_2026_04_seed_<seed>.yaml \
PAPER_RUN_ID=paper_final_seed_<seed> \
TIMESTEPS=300000 \
EPISODES=30 \
RUN_AGENTIC=1 \
AGENTIC_POLICIES=ppo,sac,dqn \
AGENTIC_PLANNER=llm \
AGENTIC_AUDITOR=llm \
AGENTIC_AUDIT_TRIGGER=event \
bash artifact_pipeline/02_run_paper_experiments.sh
```

If compute or LLM budget is constrained, prioritize keeping multiple seeds and reduce `TIMESTEPS` before falling back to a single seed.

## Optional Rule-Agentic Control

The rule-agentic control separates the effect of the wrapper structure from the effect of LLM structured reasoning. It should reuse trained PPO/SAC/DQN models from the primary seed and only rerun evaluation.

Recommended output directory:

```text
outputs/paper_rule_agentic_seed_20260511/
```

Main rule-agentic evaluation:

```bash
mkdir -p outputs/paper_rule_agentic_seed_20260511/runs

python scripts/evaluate.py \
  --config outputs/multiseed_configs/month_2026_04_seed_20260511.yaml \
  --policies ppo,sac,dqn \
  --episodes 30 \
  --ppo-model outputs/paper_final_seed_20260511/runs/main/models/ppo/ppo_model.zip \
  --sac-model outputs/paper_final_seed_20260511/runs/main/models/sac/sac_model.zip \
  --dqn-model outputs/paper_final_seed_20260511/runs/main/models/dqn/dqn_model.zip \
  --output-dir outputs/paper_rule_agentic_seed_20260511/runs \
  --run-name agentic_rule_rule \
  --no-timestamp \
  --agentic-mode planner_auditor \
  --planner rule \
  --auditor rule \
  --audit-trigger event \
  --save-agentic-logs
```

No-physics rule-agentic evaluation:

```bash
python scripts/evaluate.py \
  --config outputs/multiseed_configs/month_2026_04_seed_20260511.yaml \
  --policies ppo,sac,dqn \
  --episodes 30 \
  --ppo-model outputs/paper_final_seed_20260511/runs/no_physics_penalty/models/ppo/ppo_model.zip \
  --sac-model outputs/paper_final_seed_20260511/runs/no_physics_penalty/models/sac/sac_model.zip \
  --dqn-model outputs/paper_final_seed_20260511/runs/no_physics_penalty/models/dqn/dqn_model.zip \
  --output-dir outputs/paper_rule_agentic_seed_20260511/runs \
  --run-name agentic_rule_rule_no_physics_penalty \
  --no-timestamp \
  --no-physics-penalty \
  --agentic-mode planner_auditor \
  --planner rule \
  --auditor rule \
  --audit-trigger event \
  --save-agentic-logs
```

## Metrics

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

## Acceptance Criteria

A complete paper run should include:

- `dataset_summary.json`
- `paper_run_metadata.json`
- `PAPER_RESULTS.md`
- main and no-physics `summary.json`
- PPO/SAC/DQN model files under main and no-physics runs
- agentic `summary.json` and `agentic_logs.jsonl` when LLM runs are enabled
- four figure directories with PNGs

Validation commands:

```bash
python -m pytest -q
test -s outputs/paper_final_seed_20260511/PAPER_RESULTS.md
find outputs/paper_final_seed_20260511/figures -name "*.png" -type f
```

## Failure Policy

LLM structured-output failures should stop the run. Formal LLM results must not be replaced by mock outputs. If an endpoint fails, fix the API key, model, base URL, or structured-output compatibility and rerun the interrupted stage.
