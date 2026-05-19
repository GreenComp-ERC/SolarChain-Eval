from __future__ import annotations

import itertools

import numpy as np

from .config import BenchmarkConfig


def decode_continuous_action(raw_action: np.ndarray, config: BenchmarkConfig) -> np.ndarray:
    market = config.market
    action = np.nan_to_num(np.asarray(raw_action, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    action = np.clip(action, 0.0, 1.0)
    reward_ratio = market.min_reward_ratio + action[0] * (market.max_reward_ratio - market.min_reward_ratio)
    liquidity_ratio = market.min_liquidity_ratio + action[1] * (market.max_liquidity_ratio - market.min_liquidity_ratio)
    burn_rate = action[2] * market.max_burn_rate
    return sanitize_actual_action(np.array([reward_ratio, liquidity_ratio, burn_rate], dtype=np.float32), config)


def encode_actual_action(actual_action: np.ndarray, config: BenchmarkConfig) -> np.ndarray:
    market = config.market
    reward, liquidity, burn = sanitize_actual_action(actual_action, config)
    return np.array(
        [
            (reward - market.min_reward_ratio) / (market.max_reward_ratio - market.min_reward_ratio),
            (liquidity - market.min_liquidity_ratio) / (market.max_liquidity_ratio - market.min_liquidity_ratio),
            burn / market.max_burn_rate,
        ],
        dtype=np.float32,
    ).clip(0.0, 1.0)


def sanitize_actual_action(actual_action: np.ndarray, config: BenchmarkConfig) -> np.ndarray:
    market = config.market
    reward_ratio = float(np.clip(actual_action[0], market.min_reward_ratio, market.max_reward_ratio))
    liquidity_ratio = float(np.clip(actual_action[1], market.min_liquidity_ratio, market.max_liquidity_ratio))
    burn_rate = float(np.clip(actual_action[2], 0.0, market.max_burn_rate))
    total = reward_ratio + liquidity_ratio
    if total > market.max_total_allocation:
        scale = market.max_total_allocation / total
        reward_ratio *= scale
        liquidity_ratio *= scale
    return np.array([reward_ratio, liquidity_ratio, burn_rate], dtype=np.float32)


def discrete_action_grid(config: BenchmarkConfig) -> np.ndarray:
    values = np.linspace(0.0, 1.0, int(config.discrete_levels), dtype=np.float32)
    return np.array(list(itertools.product(values, values, values)), dtype=np.float32)


def decode_discrete_action(action_index: int, config: BenchmarkConfig) -> np.ndarray:
    grid = discrete_action_grid(config)
    index = int(np.clip(action_index, 0, len(grid) - 1))
    return decode_continuous_action(grid[index], config)

