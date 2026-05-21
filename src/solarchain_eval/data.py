from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CITIES = ["Beijing", "Shanghai", "Chengdu", "Shenzhen", "Hangzhou"]


@dataclass(frozen=True)
class BenchmarkData:
    nodes: pd.DataFrame
    generation: pd.DataFrame
    market: pd.DataFrame
    trades: pd.DataFrame
    city_hour: pd.DataFrame
    cities: list[str]
    hour_count: int


def load_benchmark_data(data_dir: str | Path) -> BenchmarkData:
    root = Path(data_dir)
    nodes = pd.read_csv(root / "urban_energy_nodes.csv")
    generation = pd.read_csv(root / "spatiotemporal_generation.csv")
    market = pd.read_csv(root / "market_liquidity.csv")
    trades = pd.read_csv(root / "p2p_trades.csv")

    generation["timestamp"] = pd.to_datetime(generation["timestamp"])
    market["timestamp"] = pd.to_datetime(market["timestamp"])
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    _add_absolute_hour_columns(generation, market, trades)
    generation["fdia_detected"] = generation["fdia_detected"].astype(bool)
    cities = [city for city in DEFAULT_CITIES if city in set(generation["city"])]

    generation["verified_W"] = np.where(generation["verification_status"].eq("verified"), generation["P_reported_W"], 0.0)
    generation["physics_excess_W"] = np.maximum(generation["P_reported_W"] - generation["P_max_W"], 0.0)
    generation["physics_violation"] = generation["physics_excess_W"].gt(1e-6) | generation["fdia_detected"]
    generation["rejected_reported_W"] = np.where(generation["physics_violation"], generation["P_reported_W"], 0.0)

    city_hour = (
        generation.groupby(["city", "absolute_hour", "hour"], as_index=False)
        .agg(
            verified_W=("verified_W", "sum"),
            reported_W=("P_reported_W", "sum"),
            pmax_W=("P_max_W", "sum"),
            physics_excess_W=("physics_excess_W", "sum"),
            rejected_reported_W=("rejected_reported_W", "sum"),
            violation_count=("physics_violation", "sum"),
            record_count=("node_id", "count"),
        )
        .sort_values(["absolute_hour", "city"])
        .reset_index(drop=True)
    )
    city_hour["violation_rate"] = city_hour["violation_count"] / city_hour["record_count"].clip(lower=1)

    hour_count = int(generation["absolute_hour"].max()) + 1 if not generation.empty else 0
    return BenchmarkData(nodes, generation, market, trades, city_hour, cities, hour_count)


def _add_absolute_hour_columns(*frames: pd.DataFrame) -> None:
    timestamp_frames = [frame for frame in frames if "timestamp" in frame and not frame.empty]
    if not timestamp_frames:
        return

    canonical = timestamp_frames[0]
    start = canonical["timestamp"].min()
    for frame in frames:
        if "timestamp" not in frame or frame.empty:
            continue
        elapsed = frame["timestamp"] - start
        frame["absolute_hour"] = (elapsed / pd.Timedelta(hours=1)).round().astype(int)
        frame["hour"] = frame["timestamp"].dt.hour.astype(int)
