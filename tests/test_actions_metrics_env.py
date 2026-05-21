from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solarchain_eval.actions import decode_continuous_action, decode_discrete_action, discrete_action_grid
from solarchain_eval.config import load_config
from solarchain_eval.data import BenchmarkData, load_benchmark_data
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


def test_default_dataset_gets_absolute_hour_index():
    data = load_benchmark_data("data/datasets")
    assert data.hour_count == 24
    assert set(data.city_hour["absolute_hour"]) == set(range(24))
    assert data.generation["hour"].between(0, 23).all()


def test_env_randomizes_whole_day_episode_starts():
    config = load_config()
    config.episode_steps = 24
    env = SolarChainBenchmarkEnv(config=config, data=_make_multiday_test_data(hours=72))

    starts = set()
    for seed in range(20):
        env.reset(seed=seed)
        starts.add(env._episode_start_hour)

    assert starts.issubset({0, 24, 48})
    assert starts


def test_env_uses_absolute_hour_within_selected_day():
    config = load_config()
    config.episode_steps = 24
    env = SolarChainBenchmarkEnv(config=config, data=_make_multiday_test_data(hours=72))
    obs, _ = env.reset(seed=config.seed, options={"start_hour": 24})
    assert obs.shape == env.observation_space.shape

    seen = []
    for _ in range(24):
        _, _, _, truncated, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        seen.append(info["absolute_hour"])

    assert seen == list(range(24, 48))
    assert truncated


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


def _make_multiday_test_data(hours: int) -> BenchmarkData:
    cities = ["Beijing", "Shanghai", "Chengdu", "Shenzhen", "Hangzhou"]
    city_rows = []
    market_rows = []
    timestamps = pd.date_range("2026-04-01", periods=hours, freq="60min", tz="Asia/Shanghai")
    for absolute_hour, timestamp in enumerate(timestamps):
        hour = int(timestamp.hour)
        market_rows.append(
            {
                "timestamp": timestamp,
                "absolute_hour": absolute_hour,
                "hour": hour,
                "total_verified_MW": 0.01,
                "slippage_solarchain_pct": 0.5,
            }
        )
        for city in cities:
            city_rows.append(
                {
                    "city": city,
                    "absolute_hour": absolute_hour,
                    "hour": hour,
                    "verified_W": 1000.0,
                    "reported_W": 1000.0,
                    "pmax_W": 1200.0,
                    "physics_excess_W": 0.0,
                    "rejected_reported_W": 0.0,
                    "violation_count": 0,
                    "record_count": 1,
                    "violation_rate": 0.0,
                }
            )

    trades = pd.DataFrame(
        columns=[
            "timestamp",
            "absolute_hour",
            "hour",
            "energy_purchased_MW",
        ]
    )
    return BenchmarkData(
        nodes=pd.DataFrame(),
        generation=pd.DataFrame(),
        market=pd.DataFrame(market_rows),
        trades=trades,
        city_hour=pd.DataFrame(city_rows),
        cities=cities,
        hour_count=hours,
    )
