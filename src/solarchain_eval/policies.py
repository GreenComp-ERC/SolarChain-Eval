from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .actions import encode_actual_action
from .config import BenchmarkConfig


class Policy(Protocol):
    name: str

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        ...


@dataclass
class StaticPolicy:
    config: BenchmarkConfig
    name: str = "static"

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        action = np.array(
            [
                self.config.market.static_reward_ratio,
                self.config.market.static_liquidity_ratio,
                self.config.market.static_burn_rate,
            ],
            dtype=np.float32,
        )
        return encode_actual_action(action, self.config), None


@dataclass
class RandomPolicy:
    config: BenchmarkConfig
    seed: int = 20260511
    name: str = "random"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self.rng.random(3, dtype=np.float32), None


@dataclass
class MyopicGreedyPolicy:
    config: BenchmarkConfig
    name: str = "myopic"

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        gap = float(obs[5])
        liquidity = float(obs[6])
        violation = float(obs[8])
        reward_ratio = 0.18 if gap < -0.1 else 0.36
        liquidity_ratio = 0.92 if liquidity < 0.25 or gap < -0.1 else 0.48
        burn_rate = 0.16 if violation > 0.0 or gap < -0.2 else 0.04
        action = np.array([reward_ratio, liquidity_ratio, burn_rate], dtype=np.float32)
        return encode_actual_action(action, self.config), None


def make_builtin_policy(name: str, config: BenchmarkConfig, seed: int | None = None) -> Policy:
    normalized = name.lower().strip()
    if normalized in {"static", "static_1_to_3", "static-1:3"}:
        return StaticPolicy(config=config)
    if normalized == "random":
        return RandomPolicy(config=config, seed=seed or config.seed)
    if normalized in {"myopic", "greedy", "myopic_greedy"}:
        return MyopicGreedyPolicy(config=config)
    raise ValueError(f"Unknown built-in policy: {name}")

