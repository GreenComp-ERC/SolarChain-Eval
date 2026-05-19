from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent.
    yaml = None


@dataclass
class MarketConfig:
    initial_liquidity: float = 1.0
    initial_token_price: float = 1.0
    min_reward_ratio: float = 0.05
    max_reward_ratio: float = 0.65
    min_liquidity_ratio: float = 0.20
    max_liquidity_ratio: float = 0.95
    max_burn_rate: float = 0.20
    max_total_allocation: float = 0.98
    static_reward_ratio: float = 0.25
    static_liquidity_ratio: float = 0.75
    static_burn_rate: float = 0.02


@dataclass
class RewardConfig:
    drawdown_penalty: float = 1.50
    action_jitter_penalty: float = 0.25
    unmet_demand_penalty: float = 0.40
    physics_penalty: float = 2.00
    fairness_penalty: float = 0.20


@dataclass
class TrainingConfig:
    timesteps: int = 10_000
    n_envs: int = 1
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.98


@dataclass
class EvaluationConfig:
    episodes: int = 5
    policies: list[str] = field(default_factory=lambda: ["static", "random", "myopic"])


@dataclass
class BenchmarkConfig:
    seed: int = 20260511
    episode_steps: int = 24
    data_dir: str = "data/datasets"
    output_dir: str = "outputs/runs"
    figures_dir: str = "figures"
    action_mode: str = "continuous"
    discrete_levels: int = 5
    no_physics_penalty: bool = False
    market: MarketConfig = field(default_factory=MarketConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge_dataclass(cls, payload: dict[str, Any]):
    fields = {name for name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in payload.items() if key in fields})


def load_config(path: str | Path = "configs/default.yaml", overrides: dict[str, Any] | None = None) -> BenchmarkConfig:
    config_path = Path(path)
    payload: dict[str, Any] = {}
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if yaml else _parse_simple_yaml(text)
        payload = payload or {}
    if overrides:
        payload = _deep_update(payload, overrides)

    return BenchmarkConfig(
        seed=payload.get("seed", BenchmarkConfig.seed),
        episode_steps=payload.get("episode_steps", BenchmarkConfig.episode_steps),
        data_dir=payload.get("data_dir", BenchmarkConfig.data_dir),
        output_dir=payload.get("output_dir", BenchmarkConfig.output_dir),
        figures_dir=payload.get("figures_dir", BenchmarkConfig.figures_dir),
        action_mode=payload.get("action_mode", BenchmarkConfig.action_mode),
        discrete_levels=payload.get("discrete_levels", BenchmarkConfig.discrete_levels),
        no_physics_penalty=payload.get("no_physics_penalty", BenchmarkConfig.no_physics_penalty),
        market=_merge_dataclass(MarketConfig, payload.get("market", {})),
        reward=_merge_dataclass(RewardConfig, payload.get("reward", {})),
        training=_merge_dataclass(TrainingConfig, payload.get("training", {})),
        evaluation=_merge_dataclass(EvaluationConfig, payload.get("evaluation", {})),
    )


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.startswith("- ") and current_section and current_list_key:
            payload[current_section].setdefault(current_list_key, []).append(_coerce_value(line[2:].strip()))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0 and not value:
            payload[key] = {}
            current_section = key
            current_list_key = None
        elif indent == 0:
            payload[key] = _coerce_value(value)
            current_section = None
            current_list_key = None
        elif current_section:
            if value:
                payload[current_section][key] = _coerce_value(value)
                current_list_key = None
            else:
                payload[current_section][key] = []
                current_list_key = key
    return payload


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")
