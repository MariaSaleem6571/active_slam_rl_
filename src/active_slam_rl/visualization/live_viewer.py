"""
Animated visualization of one rollout: ground-truth world, sonar beams,
occupancy map being built up live, and the true vs. estimated (drifting)
trajectory -- this is a "watch the algorithm work" view, complementing the
thesis's static Figure 5/6 architecture diagrams with the live simulation.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def record_rollout_frames(env, policy, max_steps: int = 250, deterministic: bool = True):
    """Runs one episode and returns a list of per-step render state dicts
    (kept lightweight: only what's needed for the animation, not full
    high-res frames) plus the world occupancy grid.

    `deterministic` controls whether SB3 models take their argmax action
    or sample from the distribution -- see
    metrics/evaluation.py::rollout_episode's docstring for why this
    matters for an early-training checkpoint (the argmax can be a
    degenerate fixed point even when the policy behaves reasonably when
    sampled).
    """
    obs, info = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    frames = []
    for _ in range(max_steps):
        action, _ = policy.predict(obs, deterministic=deterministic)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append({
            "true_pose": info["true_pose"],
            "est_pose": info["est_pose"],
            "belief": env.map.prob.copy(),
            "change_mask": env._last_change_mask.copy(),
            "loop_closure": info["loop_closure_validated"],
            "ate": info["ate"],
            "completeness": info["map_completeness"],
        })
        if terminated or truncated:
            break
    return frames, env.world.occ.copy()


def render_gif(frames, world_occ, out_path: str, fps: int = 8, stride: int = 1):
    """Renders the recorded rollout into an animated GIF: left panel shows
    the live occupancy-grid reconstruction with true/estimated trajectory
    trails, right panel shows a small live readout of drift and
    completeness so you can watch the RL policy's behavior (dwelling to
    rescan, revisiting for loop closure, exploring frontiers) alongside its
    effect on map quality.
    """
    frames = frames[::stride]
    fig, (ax_map, ax_stats) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]})

    im = ax_map.imshow(frames[0]["belief"], cmap="viridis", vmin=0, vmax=1, origin="upper")
    ax_map.contour(world_occ, levels=[0.5], colors="white", linewidths=0.8)
    true_line, = ax_map.plot([], [], color="cyan", linewidth=2, label="true")
    est_line, = ax_map.plot([], [], color="orangered", linewidth=2, linestyle="--", label="estimated (drift)")
    marker_true, = ax_map.plot([], [], "o", color="cyan", markersize=6)
    marker_est, = ax_map.plot([], [], "o", color="orangered", markersize=6)
    ax_map.legend(loc="upper right", fontsize=8)
    ax_map.set_title("Live occupancy belief p(o_v) + trajectories")

    ax_stats.axis("off")
    text = ax_stats.text(0.02, 0.7, "", fontsize=11, family="monospace", va="top")

    true_xs, true_ys, est_xs, est_ys = [], [], [], []

    def update(i):
        f = frames[i]
        im.set_data(f["belief"])
        ty, tx, _ = f["true_pose"]
        ey, ex, _ = f["est_pose"]
        true_xs.append(tx); true_ys.append(ty)
        est_xs.append(ex); est_ys.append(ey)
        true_line.set_data(true_xs, true_ys)
        est_line.set_data(est_xs, est_ys)
        marker_true.set_data([tx], [ty])
        marker_est.set_data([ex], [ey])
        lc_flag = "LOOP CLOSURE!" if f["loop_closure"] else ""
        text.set_text(
            f"step: {i * stride}\n"
            f"drift (ATE): {f['ate']:.2f} m\n"
            f"completeness: {f['completeness']:.1f}%\n\n"
            f"{lc_flag}"
        )
        return [im, true_line, est_line, marker_true, marker_est, text]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
