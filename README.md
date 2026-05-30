# SolarChain-Eval

**SolarChain-Eval: A Physics-Constrained Benchmark for Trustworthy Economic Agents in Decentralized Energy Markets**

SolarChain-Eval is a benchmark for evaluating autonomous economic governors in decentralized peer-to-peer solar energy markets. The benchmark focuses on whether reward-maximizing agents can remain trustworthy when market decisions are bounded by physical ground truth.

## Motivation

Agentic AI and cyber-physical systems create new opportunities for autonomous economic governance in decentralized energy markets. In peer-to-peer solar trading, reinforcement learning agents may be asked to manage market liquidity, token issuance, and incentive distribution. Unlike purely digital environments, however, solar markets are constrained by localized irradiance, photovoltaic capacity, verification status, and false-data injection risk.

This setting creates a core evaluation problem: a reward-maximizing agent may improve short-term liquidity by accepting impossible generation data. Under false data injection attacks, an unsafe governor can back invalid supply and create artificial liquidity, damaging market integrity even when reward appears high.

SolarChain-Eval evaluates agents as autonomous market administrators. At each hour, an agent controls:

- reward allocation between producers and the liquidity pool,
- liquidity provisioning,
- token burn rate.

The benchmark evaluates those decisions with both economic utility and trustworthiness metrics.

## Research Questions

**RQ1:** How do RL agents balance economic utility and trustworthiness metrics compared to static and heuristic baselines in a decentralized energy market?

**RQ2:** To what extent do reward-maximizing agents exploit invalid generation to artificially inflate market liquidity when physics-informed penalties are removed?

**RQ3:** How do agents' dynamic governance decisions on reward allocation, liquidity provisioning, and token burns affect market stability, action reliability, and spatial fairness across urban energy nodes?

## Benchmark Components

The environment is implemented as a Gymnasium benchmark:

```text
src/solarchain_eval/env.py
```

The main paper configuration uses the five-city April 2026 dataset:

```text
configs/month_2026_04.yaml
data/datasets_2026_04_month/
```

## Dataset Hosting

Canonical dataset hosting is planned on Hugging Face Datasets:

```text
https://huggingface.co/datasets/ThomasXu/solarchain-eval
```

The GitHub repository keeps a small mirror under:

```text
data/datasets_2026_04_month/
data/datasets/
```

The Hugging Face release includes the dataset card, dataset license, summary, checksums, main monthly CSVs, smoke CSVs, and the Open-Meteo cache. The public GitHub repository is:

```text
https://github.com/GreenComp-ERC/SolarChain-Eval
```

Built-in policy comparisons:

- Static 1:3 split
- Random
- Myopic Greedy
- PPO
- SAC
- DQN with a discretized action grid

The repository also includes an optional eval-only agentic extension:

- LLM Planner
- LLM Auditor
- Rule Planner/Auditor baselines

The LLM layer is used only during evaluation. It is never used during PPO, SAC, or DQN training.

## Setup

```bash
conda activate SolarChain-rl
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick Smoke Run

```bash
python scripts/evaluate.py \
  --config configs/month_2026_04.yaml \
  --policies "static,random,myopic" \
  --episodes 1

python scripts/make_figures.py \
  --config configs/month_2026_04.yaml \
  --run-dir outputs/runs/<timestamp>_eval
```

## Official Artifact Pipeline

The authoritative experiment and artifact documentation lives in:

```text
artifact_pipeline/
```

Convenience Linux scripts are available:

```bash
bash artifact_pipeline/00_setup_linux.sh
bash artifact_pipeline/01_smoke_check.sh
PAPER_RUN_ID=paper_final_seed_20260511 TIMESTEPS=300000 EPISODES=30 bash artifact_pipeline/02_run_paper_experiments.sh
```

If any detail in this root README conflicts with `artifact_pipeline/README.md`, use `artifact_pipeline/README.md` as the source of truth.

## Outputs

The official pipeline writes all artifacts under:

```text
outputs/<PAPER_RUN_ID>/
```

Expected subruns:

```text
outputs/<PAPER_RUN_ID>/runs/main/
outputs/<PAPER_RUN_ID>/runs/no_physics_penalty/
outputs/<PAPER_RUN_ID>/runs/agentic_llm_llm/
outputs/<PAPER_RUN_ID>/runs/agentic_llm_llm_no_physics_penalty/
```

Each evaluation bundle contains:

- `metrics.csv`
- `summary.json`
- `actions.csv`
- `city_hour_policy.csv`
- `config_snapshot.json`
- `run_metadata.json`

Agentic evaluation additionally writes:

- `agentic_logs.jsonl`

## Metrics And Figures

Primary trustworthiness and utility metrics:

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

Generated figures:

- `learning_curves.png`
- `safety_utility_frontier.png`
- `city_hour_liquidity_heatmap.png`

## Reproducibility

For the full artifact plan, see:

```text
artifact_pipeline/README.md
artifact_pipeline/REPRODUCIBILITY_PLAN.md
artifact_pipeline/LINUX_REPRODUCTION_RECOVERY.md
```
