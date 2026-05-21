from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .actions import decode_continuous_action, decode_discrete_action, discrete_action_grid
from .config import BenchmarkConfig, load_config
from .data import BenchmarkData, load_benchmark_data


class SolarChainBenchmarkEnv(gym.Env):
    """Gymnasium environment for physics-constrained SolarChain market governance."""

    metadata = {"render_modes": []}

    def __init__(self, config: BenchmarkConfig | None = None, data: BenchmarkData | None = None):
        super().__init__()
        self.config = config or load_config()
        self.data = data or load_benchmark_data(self.config.data_dir)
        self._rng = np.random.default_rng(self.config.seed)
        if self.config.action_mode == "discrete":
            self.action_space = spaces.Discrete(len(discrete_action_grid(self.config)))
        else:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=np.float32)
        self._step = 0
        self._episode_start_hour = 0
        self._liquidity = self.config.market.initial_liquidity
        self._token_price = self.config.market.initial_token_price
        self._peak_price = self._token_price
        self._peak_liquidity = self._liquidity
        self._prev_action = self._static_actual_action()
        self._city_rewards = {city: 0.0 for city in self.data.cities}
        self._last_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._episode_start_hour = self._sample_episode_start_hour(options)
        self._liquidity = self.config.market.initial_liquidity
        self._token_price = self.config.market.initial_token_price
        self._peak_price = self._token_price
        self._peak_liquidity = self._liquidity
        self._prev_action = self._static_actual_action()
        self._city_rewards = {city: 0.0 for city in self.data.cities}
        self._last_info = {}
        return self._observation(), {}

    def step(self, action):
        actual = self._decode_action(action)
        reward_ratio, liquidity_ratio, burn_rate = map(float, actual)
        absolute_hour = self._current_absolute_hour()
        hour = absolute_hour % 24
        rows = self.data.city_hour[self.data.city_hour["absolute_hour"].eq(absolute_hour)]
        verified_mwh = float(rows["verified_W"].sum() / 1_000_000.0)
        reported_mwh = float(rows["reported_W"].sum() / 1_000_000.0)
        pmax_mwh = float(rows["pmax_W"].sum() / 1_000_000.0)
        excess_mwh = float(rows["physics_excess_W"].sum() / 1_000_000.0)
        rejected_reported_mwh = float(rows["rejected_reported_W"].sum() / 1_000_000.0)
        raw_physics_record_rate = float(rows["violation_count"].sum() / max(rows["record_count"].sum(), 1))

        trade_rows = self.data.trades[self.data.trades["absolute_hour"].eq(absolute_hour)]
        demand_mwh = float(trade_rows["energy_purchased_MW"].sum())
        if demand_mwh <= 0:
            market_hour = self.data.market[self.data.market["absolute_hour"].eq(absolute_hour)]
            demand_mwh = max(float(market_hour["total_verified_MW"].sum()), 0.001)

        unsafe_supply = max(rejected_reported_mwh, excess_mwh, max(reported_mwh - pmax_mwh, 0.0))
        backing_pressure = min(
            reward_ratio + liquidity_ratio,
            self.config.market.max_total_allocation,
        ) / max(self.config.market.max_total_allocation, 1e-9)
        unsafe_backed_mwh = unsafe_supply * backing_pressure
        backed_supply_mwh = verified_mwh + unsafe_backed_mwh
        physics_rate = unsafe_backed_mwh / max(backed_supply_mwh, 1e-9)
        liquidity_added = backed_supply_mwh * liquidity_ratio
        reward_tokens = backed_supply_mwh * reward_ratio
        effective_demand = demand_mwh * max(0.75, 1.0 - 0.80 * burn_rate)
        available = self._liquidity + liquidity_added
        matched = min(available, effective_demand)
        unmet = max(effective_demand - available, 0.0)
        self._liquidity = max(available - matched, 0.0)
        self._peak_liquidity = max(self._peak_liquidity, self._liquidity)
        liquidity_drawdown = 1.0 - self._liquidity / max(self._peak_liquidity, 1e-12)

        slippage = float(effective_demand / max(available + 0.05, 0.05))
        action_delta = float(np.linalg.norm(actual - self._prev_action, ord=1))
        supply_gap = (verified_mwh - effective_demand) / max(effective_demand, 1e-6)
        price_return = (
            0.04 * supply_gap
            + 0.03 * matched
            + 0.10 * burn_rate
            - 0.08 * slippage
            - 0.02 * reward_tokens
            + float(self._rng.normal(0.0, 0.004))
        )
        price_return = float(np.clip(price_return, -0.35, 0.35))
        self._token_price = max(0.05, self._token_price * (1.0 + price_return))
        self._peak_price = max(self._peak_price, self._token_price)
        token_drawdown = 1.0 - self._token_price / max(self._peak_price, 1e-12)

        city_rewards = self._allocate_city_rewards(rows, reward_tokens, liquidity_ratio)
        for city, value in city_rewards.items():
            self._city_rewards[city] += value
        fairness_variance = float(np.var(list(city_rewards.values()))) if city_rewards else 0.0
        physics_penalty = 0.0 if self.config.no_physics_penalty else self.config.reward.physics_penalty * (physics_rate + unsafe_backed_mwh)
        reward = (
            matched
            - self.config.reward.drawdown_penalty * liquidity_drawdown
            - self.config.reward.action_jitter_penalty * action_delta
            - self.config.reward.unmet_demand_penalty * unmet
            - self.config.reward.fairness_penalty * fairness_variance
            - physics_penalty
        )

        self._prev_action = actual
        self._step += 1
        truncated = self._step >= self.config.episode_steps
        terminated = False
        self._last_info = {
            "absolute_hour": absolute_hour,
            "episode_start_hour": self._episode_start_hour,
            "hour": hour,
            "reward_ratio": reward_ratio,
            "liquidity_ratio": liquidity_ratio,
            "burn_rate": burn_rate,
            "verified_supply_MWh": verified_mwh,
            "reported_supply_MWh": reported_mwh,
            "physics_excess_MWh": unsafe_supply,
            "unsafe_backed_MWh": unsafe_backed_mwh,
            "artificial_liquidity_MWh": unsafe_backed_mwh * liquidity_ratio,
            "physics_violation_rate": physics_rate,
            "raw_physics_record_rate": raw_physics_record_rate,
            "demand_MWh": demand_mwh,
            "matched_energy_MWh": matched,
            "unmet_demand_MWh": unmet,
            "liquidity": self._liquidity,
            "liquidity_depth": self._liquidity / max(self.config.market.initial_liquidity, 1e-9),
            "token_price": self._token_price,
            "max_drawdown": liquidity_drawdown,
            "token_drawdown": token_drawdown,
            "action_jitter": action_delta,
            "slippage": slippage,
            "city_rewards": city_rewards,
        }
        return self._observation(), float(reward), terminated, truncated, dict(self._last_info)

    def _decode_action(self, action) -> np.ndarray:
        if self.config.action_mode == "discrete":
            return decode_discrete_action(int(action), self.config)
        return decode_continuous_action(np.asarray(action, dtype=np.float32), self.config)

    def _sample_episode_start_hour(self, options: dict[str, Any] | None = None) -> int:
        if options and "start_hour" in options:
            start_hour = int(options["start_hour"])
            upper_bound = max(self.data.hour_count - self.config.episode_steps, 0)
            return int(np.clip(start_hour, 0, upper_bound))
        if self.data.hour_count <= self.config.episode_steps:
            return 0
        latest_start = self.data.hour_count - self.config.episode_steps
        daily_starts = np.arange(0, latest_start + 1, 24, dtype=np.int64)
        if len(daily_starts) == 0:
            return 0
        return int(self._rng.choice(daily_starts))

    def _current_absolute_hour(self) -> int:
        if self.data.hour_count <= 0:
            return self._episode_start_hour + self._step
        return min(self._episode_start_hour + self._step, self.data.hour_count - 1)

    def _allocate_city_rewards(self, rows, reward_tokens: float, liquidity_ratio: float) -> dict[str, float]:
        total_verified = max(float(rows["verified_W"].sum()), 1e-9)
        city_rewards: dict[str, float] = {}
        for row in rows.to_dict("records"):
            city = str(row["city"])
            share = float(row["verified_W"]) / total_verified
            trust_discount = 1.0 - float(row["violation_rate"])
            city_rewards[city] = reward_tokens * share * trust_discount + liquidity_ratio * share * 0.01
        return city_rewards

    def _observation(self) -> np.ndarray:
        absolute_hour = self._current_absolute_hour()
        hour = absolute_hour % 24
        rows = self.data.city_hour[self.data.city_hour["absolute_hour"].eq(absolute_hour)]
        market = self.data.market[self.data.market["absolute_hour"].eq(absolute_hour)]
        verified_mwh = float(rows["verified_W"].sum() / 1_000_000.0)
        reported_mwh = float(rows["reported_W"].sum() / 1_000_000.0)
        pmax_mwh = float(rows["pmax_W"].sum() / 1_000_000.0)
        violation_rate = float(rows["violation_count"].sum() / max(rows["record_count"].sum(), 1))
        demand_mwh = float(self.data.trades[self.data.trades["absolute_hour"].eq(absolute_hour)]["energy_purchased_MW"].sum())
        static_slippage = float(market["slippage_solarchain_pct"].mean()) if not market.empty else 0.0
        gap = (verified_mwh - demand_mwh) / max(demand_mwh, 1e-6)
        obs = np.array(
            [
                float(np.sin(2 * np.pi * hour / 24)),
                float(np.cos(2 * np.pi * hour / 24)),
                verified_mwh,
                reported_mwh,
                pmax_mwh,
                gap,
                self._liquidity,
                self._token_price,
                violation_rate,
                static_slippage,
                float(self._prev_action[0]),
                float(self._prev_action[1]),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -10.0, 10.0)

    def _static_actual_action(self) -> np.ndarray:
        return np.array(
            [
                self.config.market.static_reward_ratio,
                self.config.market.static_liquidity_ratio,
                self.config.market.static_burn_rate,
            ],
            dtype=np.float32,
        )

    def latest_info(self) -> dict[str, Any]:
        return dict(self._last_info)

    def city_rewards(self) -> dict[str, float]:
        return dict(self._city_rewards)
