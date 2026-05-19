from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor

from .config import BenchmarkConfig
from .data import load_benchmark_data
from .env import SolarChainBenchmarkEnv


def make_env(config: BenchmarkConfig):
    data = load_benchmark_data(config.data_dir)

    def _factory():
        return Monitor(SolarChainBenchmarkEnv(config=config, data=data))

    return _factory


def train_model(algo: str, config: BenchmarkConfig, timesteps: int, output_dir: str | Path):
    normalized = algo.lower().strip()
    local_config = deepcopy(config)
    local_config.action_mode = "discrete" if normalized == "dqn" else "continuous"
    env = make_env(local_config)()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if normalized == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            seed=local_config.seed,
            learning_rate=local_config.training.learning_rate,
            batch_size=local_config.training.batch_size,
            gamma=local_config.training.gamma,
            verbose=1,
        )
    elif normalized == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            seed=local_config.seed,
            learning_rate=local_config.training.learning_rate,
            batch_size=local_config.training.batch_size,
            gamma=local_config.training.gamma,
            verbose=1,
        )
    elif normalized == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            seed=local_config.seed,
            learning_rate=local_config.training.learning_rate,
            batch_size=local_config.training.batch_size,
            gamma=local_config.training.gamma,
            learning_starts=min(100, max(1, timesteps // 10)),
            verbose=1,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    model.learn(total_timesteps=timesteps)
    model_path = output / f"{normalized}_model"
    model.save(model_path)
    return model_path.with_suffix(".zip")

