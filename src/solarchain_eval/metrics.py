from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(values: list[float] | np.ndarray) -> float:
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(series)
    return float(np.max(1.0 - series / np.maximum(peaks, 1e-12)))


def action_jitter(actions: list[list[float]] | np.ndarray) -> float:
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(arr, axis=0), ord=1, axis=1)))


def spatial_fairness_index(city_rewards: dict[str, float]) -> float:
    values = np.asarray(list(city_rewards.values()), dtype=np.float64)
    if values.size == 0:
        return 0.0
    mean_abs = max(float(np.mean(np.abs(values))), 1e-9)
    return float(np.var(values) / mean_abs)


def summarize_episode(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    city_rewards: dict[str, float] = {}
    for row in rows:
        for city, value in row.get("city_rewards", {}).items():
            city_rewards[city] = city_rewards.get(city, 0.0) + float(value)

    return {
        "cumulative_reward": float(frame["reward"].sum()),
        "episode_volume": float(frame["matched_energy_MWh"].sum()),
        "physics_violation_rate": float(frame["physics_violation_rate"].mean()),
        "max_drawdown": max_drawdown(frame["liquidity"].to_numpy()),
        "max_token_drawdown": max_drawdown(frame["token_price"].to_numpy()),
        "action_jitter": action_jitter(frame[["reward_ratio", "liquidity_ratio", "burn_rate"]].to_numpy()),
        "mean_slippage": float(frame["slippage"].mean()),
        "spatial_fairness_index": spatial_fairness_index(city_rewards),
        "artificial_liquidity_MWh": float(frame["artificial_liquidity_MWh"].sum()),
        "final_liquidity": float(frame["liquidity"].iloc[-1]),
        "final_token_price": float(frame["token_price"].iloc[-1]),
    }
