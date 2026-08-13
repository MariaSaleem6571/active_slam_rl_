"""
Plotting utilities. Every function here saves a PNG to `out_dir` and
returns the path, so scripts can chain plot -> present_files without
holding figures open.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(csv_path: str, out_dir: str) -> str:
    """Reward, map completeness, ATE, and collisions over training episodes
    -- read straight from the MetricsLoggingCallback's CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No episodes logged yet in {csv_path}")

    df = df.sort_values("timestep")
    window = max(1, len(df) // 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(df["timestep"], df["episode_reward"], alpha=0.3, color="tab:blue")
    ax.plot(df["timestep"], df["episode_reward"].rolling(window, min_periods=1).mean(),
            color="tab:blue", linewidth=2)
    ax.set_title("Episode reward")
    ax.set_xlabel("training timestep")
    ax.set_ylabel("total reward")

    ax = axes[0, 1]
    ax.plot(df["timestep"], df["map_completeness"], alpha=0.3, color="tab:green")
    ax.plot(df["timestep"], df["map_completeness"].rolling(window, min_periods=1).mean(),
            color="tab:green", linewidth=2)
    ax.set_title("Map completeness (%)")
    ax.set_xlabel("training timestep")
    ax.set_ylabel("completeness")

    ax = axes[1, 0]
    ax.plot(df["timestep"], df["mean_ate"], alpha=0.3, color="tab:red")
    ax.plot(df["timestep"], df["mean_ate"].rolling(window, min_periods=1).mean(),
            color="tab:red", linewidth=2)
    ax.set_title("Mean Absolute Trajectory Error (drift)")
    ax.set_xlabel("training timestep")
    ax.set_ylabel("ATE (m)")

    ax = axes[1, 1]
    ax.plot(df["timestep"], df["collision_count"], alpha=0.3, color="tab:orange", label="collisions")
    ax2 = ax.twinx()
    ax2.plot(df["timestep"], df["loop_closures_validated"], alpha=0.6, color="tab:purple",
             label="loop closures")
    ax.set_title("Collisions & validated loop closures / episode")
    ax.set_xlabel("training timestep")
    ax.set_ylabel("collisions", color="tab:orange")
    ax2.set_ylabel("loop closures", color="tab:purple")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_episode_dashboard(metrics, world_occ: np.ndarray, out_dir: str, name: str = "episode_dashboard") -> str:
    """Single-episode dashboard: reconstructed map + true/estimated
    trajectories, plus ATE, reward, and completeness traces over time."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.imshow(world_occ, cmap="Greys", origin="upper")
    if metrics.true_traj:
        ty, tx = zip(*[(p[0], p[1]) for p in metrics.true_traj])
        ey, ex = zip(*[(p[0], p[1]) for p in metrics.est_traj])
        ax.plot(tx, ty, color="tab:blue", label="true trajectory", linewidth=1.5)
        ax.plot(ex, ey, color="tab:red", linestyle="--", label="estimated (drifted) trajectory", linewidth=1.5)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Ground-truth world + true vs. estimated trajectory")

    ax = axes[0, 1]
    ax.plot(metrics.ate_trace, color="tab:red")
    ax.set_title("Absolute Trajectory Error over time")
    ax.set_xlabel("step")
    ax.set_ylabel("ATE (m)")

    ax = axes[0, 2]
    ax.plot(metrics.completeness_trace, color="tab:green")
    ax.set_title("Map completeness over time")
    ax.set_xlabel("step")
    ax.set_ylabel("completeness (%)")

    ax = axes[1, 0]
    ax.plot(np.cumsum(metrics.reward_trace), color="tab:blue")
    ax.set_title("Cumulative reward")
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative reward")

    ax = axes[1, 1]
    ax.plot(metrics.entropy_trace, color="tab:purple")
    ax.set_title("Map entropy H(M_t) over time")
    ax.set_xlabel("step")
    ax.set_ylabel("entropy (nats)")

    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        f"Mission time: {metrics.mission_time_steps} steps\n"
        f"Path length: {metrics.path_length:.1f} m\n"
        f"Final completeness: {metrics.completeness:.1f}%\n"
        f"Mean ATE: {metrics.ate_mean:.2f} m\n"
        f"Mean RPE: {metrics.rpe_mean:.3f} m/step\n"
        f"Info gain / meter: {metrics.info_gain_per_meter:.4f}\n"
        f"Collisions: {metrics.collision_count}\n"
        f"Validated loop closures: {metrics.loop_closures_validated}\n"
        f"Total reward: {metrics.total_reward:.1f}"
    )
    ax.text(0.0, 0.5, summary, fontsize=12, va="center", family="monospace")
    ax.set_title("Episode summary")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_baseline_comparison(summaries: dict, out_dir: str) -> str:
    """Bar-chart comparison of RL policy vs. Frontier / NBV / Random Walk
    baselines across the thesis's headline metrics (section 8.5)."""
    metrics_to_plot = [
        ("completeness", "Map completeness (%)", False),
        ("ate_mean", "Mean ATE (m, lower better)", True),
        ("info_gain_per_meter", "Info gain / meter", False),
        ("collision_count", "Collisions (lower better)", True),
    ]
    policies = list(summaries.keys())
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))

    for ax, (key, title, invert) in zip(axes, metrics_to_plot):
        means = [summaries[p][key][0] for p in policies]
        stds = [summaries[p][key][1] for p in policies]
        ax.bar(policies, means, yerr=stds, color=colors, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("RL policy vs. heuristic baselines")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "baseline_comparison.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
