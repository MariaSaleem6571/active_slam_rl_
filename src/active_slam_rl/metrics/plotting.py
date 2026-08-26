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


def plot_training_diagnostics_curves(csv_path: str, out_dir: str) -> str:
    """Registration/fusion diagnostics over *training*, not just a single
    eval episode -- the training-time counterpart to
    plot_registration_and_fusion_diagnostics. Reads the extra columns
    MetricsLoggingCallback now logs per episode: mean_q_t split by sonar
    modality, the FS2D NIS-gate rejection rate, and FS2D usage rate. Lets
    you watch, e.g., whether the imaging/scanning q_t gap (see
    registration/fs2d.py's fold-ambiguity discussion) narrows or widens
    over training, and whether the policy learns to lean on `dwell` more
    when imaging-mode registration is unreliable.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No episodes logged yet in {csv_path}")

    df = df.sort_values("timestep")
    window = max(1, len(df) // 40)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for col, color, label in [
        ("mean_q_t_imaging", "tab:orange", "imaging"),
        ("mean_q_t_scanning", "tab:cyan", "scanning"),
    ]:
        if col in df.columns:
            ax.plot(df["timestep"], df[col].rolling(window, min_periods=1).mean(),
                    color=color, linewidth=2, label=label)
    ax.set_title("Mean registration quality (q_t) by modality")
    ax.set_xlabel("training timestep")
    ax.set_ylabel("q_t (rolling mean)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    for col, color, label in [
        ("fs2d_rejected_outlier_rate", "tab:red", "NIS-gate rejection rate"),
        ("used_fs2d_rate", "tab:green", "FS2D usage rate"),
    ]:
        if col in df.columns:
            ax.plot(df["timestep"], df[col].rolling(window, min_periods=1).mean(),
                    color=color, linewidth=2, label=label)
    ax.set_title("Fusion gating over training")
    ax.set_xlabel("training timestep")
    ax.set_ylabel("fraction (rolling mean)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "training_diagnostics_curves.png")
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


def plot_registration_and_fusion_diagnostics(metrics, out_dir: str, name: str = "registration_diagnostics") -> str:
    """Per-step registration quality (q_t) broken down by sonar modality,
    plus StateFusionModule internals (gyro-bias estimate, NIS-gate
    rejection rate, FS2D usage fraction) over one episode.

    This is the visibility that was missing for diagnosing
    registration/fs2d.py's fold-ambiguity issue and env/sim_env.py's
    cross-modality registration fix: `plot_training_curves` and
    `plot_episode_dashboard` only ever showed aggregate reward/ATE/
    completeness, with nothing that would show *why* those numbers moved
    -- e.g. that q_t is structurally worse on imaging-mode steps than
    scanning-mode ones, or that the NIS gate is rejecting an unusual
    fraction of FS2D registrations as outliers.

    Call with an `EpisodeMetrics` from `rollout_episode` (its
    `q_t_trace`/`frame_mode_trace`/`bias_estimate_deg_trace`/
    `fs2d_rejected_outlier_trace`/`used_fs2d_trace` fields).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    modes = np.array(metrics.frame_mode_trace)
    q_t = np.array(metrics.q_t_trace)
    steps = np.arange(len(q_t))
    for mode, color in [("imaging", "tab:orange"), ("scanning", "tab:cyan")]:
        sel = modes == mode
        if sel.any():
            ax.scatter(steps[sel], q_t[sel], s=10, alpha=0.6, color=color, label=mode)
    ax.set_title("Registration quality (q_t) by sonar modality")
    ax.set_xlabel("step")
    ax.set_ylabel("q_t")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    labels = ["imaging", "scanning"]
    box_data = [q_t[modes == m] for m in labels]
    box_data = [d for d in box_data if len(d) > 0]
    box_labels = [m for m, d in zip(labels, [q_t[modes == m] for m in labels]) if len(d) > 0]
    if box_data:
        ax.boxplot(box_data, tick_labels=box_labels)
    ax.set_title("q_t distribution by modality")
    ax.set_ylabel("q_t")

    ax = axes[1, 0]
    if metrics.bias_estimate_deg_trace:
        bsteps = np.arange(len(metrics.bias_estimate_deg_trace))
        ax.plot(bsteps, metrics.bias_estimate_deg_trace, color="tab:purple")
        ax.set_title("SfM gyro-bias estimate over episode")
        ax.set_xlabel("step (fusion steps only)")
        ax.set_ylabel("bias estimate (deg)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "cfg.use_sfm_fusion=False\n(no fusion diagnostics this episode)",
                ha="center", va="center", fontsize=10)

    ax = axes[1, 1]
    if metrics.used_fs2d_trace:
        fsteps = np.arange(len(metrics.used_fs2d_trace))
        window = max(1, len(fsteps) // 20)
        rejected = pd.Series(metrics.fs2d_rejected_outlier_trace, dtype=float)
        used = pd.Series(metrics.used_fs2d_trace, dtype=float)
        ax.plot(fsteps, rejected.rolling(window, min_periods=1).mean(),
                color="tab:red", label="NIS-gate rejection rate")
        ax.plot(fsteps, used.rolling(window, min_periods=1).mean(),
                color="tab:green", label="FS2D usage rate")
        ax.set_title("Fusion gating (rolling)")
        ax.set_xlabel("step (fusion steps only)")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.axis("off")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_sonar_frame_comparison(env, out_dir: str, name: str = "sonar_frame_comparison") -> str:
    """Side-by-side raw imaging-sonar vs. scanning/360-sonar frames at the
    environment's current pose, plus the FS2D phase-correlation surface
    between two consecutive imaging frames -- gives a direct look at *why*
    registration behaves differently on the two modalities (see
    env/sonar_model.py's "MODALITY TAGGING" section and
    registration/fs2d.py's fold-ambiguity discussion) instead of only
    seeing the downstream quality numbers.

    `env` must already be reset (post-`env.reset()`). Doesn't advance or
    mutate the environment's own state beyond drawing extra sonar
    readings (sensing is read-only -- see env/sonar_model.py).
    """
    y, x, theta = env.true_pose
    _, _, _, frame_imaging_1 = env.sonar.sense_imaging(y, x, theta)
    _, _, _, frame_scanning = env.sonar.sense_scanning_360(y, x, theta)
    # a second imaging frame after a small simulated rotation, purely for
    # the correlation-surface panel -- doesn't touch env.true_pose/est_pose
    _, _, _, frame_imaging_2 = env.sonar.sense_imaging(y, x, theta + np.deg2rad(10.0))

    from active_slam_rl.registration.fs2d import _hann_window_2d
    a = frame_imaging_1.astype(np.float64) * _hann_window_2d(frame_imaging_1.shape)
    b = frame_imaging_2.astype(np.float64) * _hann_window_2d(frame_imaging_2.shape)
    Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    R /= (np.abs(R) + 1e-8)
    corr_surface = np.fft.fftshift(np.fft.ifft2(R).real)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.imshow(frame_imaging_1, cmap="viridis", origin="upper")
    ax.set_title(f"Imaging sonar frame\n({(frame_imaging_1 > 0).mean()*100:.1f}% nonzero, "
                 f"{env.cfg.sonar.n_beams} beams / {env.cfg.sonar.fov_deg:.0f} deg FOV)")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(frame_scanning, cmap="viridis", origin="upper")
    ax.set_title(f"Scanning (360 deg) sonar frame\n({(frame_scanning > 0).mean()*100:.1f}% nonzero)")
    ax.axis("off")

    ax = axes[2]
    im = ax.imshow(corr_surface, cmap="inferno", origin="upper")
    ax.set_title("FS2D translation phase-correlation\nsurface (two imaging frames, 10 deg apart)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
