from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def make_figures(run_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    setup_style()
    run = Path(run_dir)
    output = Path(figures_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(run / "metrics.csv")
    city_hour = pd.read_csv(run / "city_hour_policy.csv")
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    for policy, frame in metrics.groupby("policy"):
        ax.plot(frame["episode"], frame["cumulative_reward"], marker="o", label=policy)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    path = output / "learning_curves.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    for policy, frame in metrics.groupby("policy"):
        ax.scatter(frame["physics_violation_rate"], frame["cumulative_reward"], s=70, label=policy)
    ax.set_xlabel("Physics Violation Rate")
    ax.set_ylabel("Cumulative Profit / Reward")
    ax.legend()
    path = output / "safety_utility_frontier.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    heat_source = city_hour.groupby(["policy", "city", "hour"], as_index=False)["liquidity_ratio"].mean()
    preferred_policy = "ppo" if "ppo" in set(heat_source["policy"]) else sorted(heat_source["policy"].unique())[0]
    static_heat = _policy_heat(heat_source, "static")
    agent_heat = _policy_heat(heat_source, preferred_policy)
    if static_heat is None:
        static_heat = agent_heat.copy()
    static_heat, agent_heat = static_heat.align(agent_heat, join="outer", axis=None, fill_value=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    vmin = min(float(static_heat.min().min()), float(agent_heat.min().min()))
    vmax = max(float(static_heat.max().max()), float(agent_heat.max().max()))
    for ax, heat, title in [
        (axes[0], static_heat, "Static 1:3"),
        (axes[1], agent_heat, preferred_policy),
    ]:
        image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels([str(col) for col in heat.columns], rotation=0)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_xlabel("Hour of Day")
        ax.set_title(title)
    axes[0].set_ylabel("City")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist())
    cbar.set_label("Liquidity Split")
    path = output / "city_hour_liquidity_heatmap.png"
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    return written


def _policy_heat(heat_source: pd.DataFrame, policy: str) -> pd.DataFrame | None:
    frame = heat_source[heat_source["policy"].eq(policy)]
    if frame.empty:
        return None
    return frame.pivot_table(index="city", columns="hour", values="liquidity_ratio", fill_value=0.0)
