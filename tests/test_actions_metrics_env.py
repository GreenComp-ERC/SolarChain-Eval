from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solarchain_eval.actions import decode_continuous_action, decode_discrete_action, discrete_action_grid
from solarchain_eval.config import load_config
from solarchain_eval.env import SolarChainBenchmarkEnv
from solarchain_eval.metrics import action_jitter, max_drawdown, spatial_fairness_index
from solarchain_eval.policies import make_builtin_policy


def test_action_mapping_and_dqn_grid():
    config = load_config()
    actual = decode_continuous_action(np.array([1.0, 1.0, 1.0]), config)
    assert actual.shape == (3,)
    assert actual[0] + actual[1] <= config.market.max_total_allocation + 1e-6
    assert len(discrete_action_grid(config)) == config.discrete_levels ** 3
    dqn_action = decode_discrete_action(0, config)
    assert dqn_action.shape == (3,)


def test_metrics():
    assert max_drawdown([1.0, 1.2, 0.6]) == 0.5
    assert action_jitter([[0, 0, 0], [1, 0, 0], [1, 1, 0]]) == 1.0
    assert spatial_fairness_index({"a": 1.0, "b": 2.0}) > 0.0


def test_env_static_random_myopic_smoke():
    config = load_config()
    config.episode_steps = 3
    for name in ["static", "random", "myopic"]:
        env = SolarChainBenchmarkEnv(config)
        policy = make_builtin_policy(name, config)
        obs, _ = env.reset(seed=config.seed)
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert "physics_violation_rate" in info
        assert not terminated


def test_physics_violation_is_action_dependent():
    config = load_config()
    env = SolarChainBenchmarkEnv(config)
    obs, _ = env.reset(seed=config.seed)
    _, _, _, _, low_info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    env = SolarChainBenchmarkEnv(config)
    obs, _ = env.reset(seed=config.seed)
    _, _, _, _, high_info = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))

    assert high_info["physics_violation_rate"] >= low_info["physics_violation_rate"]
    assert "artificial_liquidity_MWh" in high_info
