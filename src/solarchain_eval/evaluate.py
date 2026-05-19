from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .data import load_benchmark_data
from .env import SolarChainBenchmarkEnv
from .metrics import summarize_episode
from .policies import Policy


def run_episode(
    policy: Policy,
    config: BenchmarkConfig,
    seed: int,
    episode: int,
    policy_name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local_config = deepcopy(config)
    name = policy_name or getattr(policy, "name", "policy")
    local_config.action_mode = "discrete" if name == "dqn" else "continuous"
    if hasattr(policy, "config"):
        policy.config = local_config
    data = load_benchmark_data(local_config.data_dir)
    env = SolarChainBenchmarkEnv(config=local_config, data=data)
    obs, _ = env.reset(seed=seed)
    done = False
    step_rows: list[dict[str, Any]] = []

    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        row = {
            "policy": name,
            "episode": episode,
            "step": len(step_rows),
            "reward": float(reward),
            **{key: value for key, value in info.items() if key != "city_rewards"},
            "city_rewards": info.get("city_rewards", {}),
        }
        step_rows.append(row)
        done = terminated or truncated

    metrics = summarize_episode(step_rows)
    metrics.update({"policy": name, "episode": episode, "seed": seed})
    return metrics, step_rows


def evaluate_policies(
    policies: list[Policy],
    config: BenchmarkConfig,
    episodes: int,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    city_hour_rows: list[dict[str, Any]] = []

    for policy in policies:
        name = getattr(policy, "name", "policy")
        for episode in range(episodes):
            metrics, steps = run_episode(policy, config, config.seed + episode, episode, name)
            metric_rows.append(metrics)
            for row in steps:
                action_rows.append({key: value for key, value in row.items() if key != "city_rewards"})
                for city, value in row.get("city_rewards", {}).items():
                    city_hour_rows.append(
                        {
                            "policy": name,
                            "episode": episode,
                            "hour": row["hour"],
                            "city": city,
                            "city_reward": float(value),
                            "reward_ratio": row["reward_ratio"],
                            "liquidity_ratio": row["liquidity_ratio"],
                            "burn_rate": row["burn_rate"],
                        }
                    )

    metrics_frame = pd.DataFrame(metric_rows)
    actions_frame = pd.DataFrame(action_rows)
    city_hour_frame = pd.DataFrame(city_hour_rows)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    actions_frame.to_csv(output / "actions.csv", index=False)
    city_hour_frame.to_csv(output / "city_hour_policy.csv", index=False)

    summary = metrics_frame.groupby("policy", as_index=False).mean(numeric_only=True)
    if "static" in set(summary["policy"]):
        static_slippage = float(summary.loc[summary["policy"].eq("static"), "mean_slippage"].iloc[0])
        summary["slippage_reduction_vs_static"] = (static_slippage - summary["mean_slippage"]) / max(static_slippage, 1e-9)
    summary.to_json(output / "summary.json", orient="records", indent=2)
    (output / "config_snapshot.json").write_text(_json_dumps_dataclass(config), encoding="utf-8")
    return metrics_frame, actions_frame, city_hour_frame


def _json_dumps_dataclass(config: BenchmarkConfig) -> str:
    import json

    return json.dumps(asdict(config), indent=2)


class SB3Policy:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)

