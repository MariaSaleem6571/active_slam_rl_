"""
Evaluation metrics (thesis section 8), computed by rolling a policy out in
`ActiveSlamEnv` and aggregating the per-step `info` dict. Every formula
here corresponds 1:1 to a named metric in the proposal:

  8.1 Mapping Quality      -- completeness (%), RMSE, entropy reduction
  8.2 Localization         -- ATE, RPE, loop-closure precision/recall
  8.3 Exploration Efficiency -- mission time, path length, info gain/meter
  8.4 Safety               -- collision rate, min obstacle distance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class EpisodeMetrics:
    completeness: float = 0.0
    entropy_reduction: float = 0.0
    ate_mean: float = 0.0
    ate_final: float = 0.0
    rpe_mean: float = 0.0
    mission_time_steps: int = 0
    path_length: float = 0.0
    info_gain_per_meter: float = 0.0
    collision_count: int = 0
    min_obstacle_distance: float = 0.0
    total_reward: float = 0.0
    loop_closures_validated: int = 0

    # raw traces, kept for plotting
    ate_trace: List[float] = field(default_factory=list)
    reward_trace: List[float] = field(default_factory=list)
    completeness_trace: List[float] = field(default_factory=list)
    entropy_trace: List[float] = field(default_factory=list)
    true_traj: List[tuple] = field(default_factory=list)
    est_traj: List[tuple] = field(default_factory=list)

    # registration / sensor-fusion diagnostics, kept for plotting (see
    # metrics/plotting.py's plot_registration_and_fusion_diagnostics).
    # frame_mode_trace entries are "imaging"/"scanning"/None (None only
    # for the very first step of the episode, before any registration has
    # a previous frame to compare against).
    q_t_trace: List[float] = field(default_factory=list)
    frame_mode_trace: List[str] = field(default_factory=list)
    bias_estimate_deg_trace: List[float] = field(default_factory=list)
    fs2d_rejected_outlier_trace: List[bool] = field(default_factory=list)
    used_fs2d_trace: List[bool] = field(default_factory=list)


def relative_pose_error(true_traj: List[tuple], est_traj: List[tuple]) -> float:
    """RPE (m/frame): mean drift accumulated *per step* rather than total
    accumulated ATE -- i.e. how much the estimate diverges from truth on
    each individual transition."""
    if len(true_traj) < 2:
        return 0.0
    errs = []
    for i in range(1, len(true_traj)):
        t_delta = np.array(true_traj[i][:2]) - np.array(true_traj[i - 1][:2])
        e_delta = np.array(est_traj[i][:2]) - np.array(est_traj[i - 1][:2])
        errs.append(np.linalg.norm(t_delta - e_delta))
    return float(np.mean(errs))


def rollout_episode(env, policy, max_steps: int | None = None, deterministic: bool = True) -> EpisodeMetrics:
    """Runs one episode with `policy` (anything exposing `.predict(obs)`,
    i.e. an SB3 model or one of rl/baselines.py's heuristics) and returns
    the full metric bundle.

    `deterministic` controls whether SB3 models pick their argmax action
    or sample from the action distribution. Early in training the policy's
    argmax can be a degenerate fixed point (e.g. "always turn" -- which,
    if the observation only depends on position, never changes the
    observation and so never changes the argmax either) even though the
    stochastic policy behaves reasonably; set `deterministic=False` to see
    the policy's actual learned behavior in that regime. Once training has
    converged (low entropy, confident argmax) the two should agree.
    """
    obs, info = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    metrics = EpisodeMetrics()
    initial_entropy = env.map.entropy()
    min_obs_dist = np.inf
    steps = 0

    while True:
        action, _ = policy.predict(obs, deterministic=deterministic)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        metrics.total_reward += reward
        metrics.ate_trace.append(info["ate"])
        metrics.reward_trace.append(reward)
        metrics.completeness_trace.append(info["map_completeness"])
        metrics.entropy_trace.append(info["map_entropy"])
        metrics.true_traj.append(info["true_pose"])
        metrics.est_traj.append(info["est_pose"])
        metrics.q_t_trace.append(info["q_t"])
        metrics.frame_mode_trace.append(info["frame_mode"])
        if info["sfm"] is not None:
            metrics.bias_estimate_deg_trace.append(info["sfm"]["bias_estimate_deg"])
            metrics.fs2d_rejected_outlier_trace.append(info["sfm"]["fs2d_rejected_outlier"])
            metrics.used_fs2d_trace.append(info["sfm"]["used_fs2d"])
        if info["collided"]:
            min_obs_dist = min(min_obs_dist, 0.0)
        metrics.loop_closures_validated += int(info["loop_closure_validated"])

        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break

    metrics.completeness = info["map_completeness"]
    metrics.entropy_reduction = initial_entropy - info["map_entropy"]
    metrics.ate_mean = float(np.mean(metrics.ate_trace)) if metrics.ate_trace else 0.0
    metrics.ate_final = metrics.ate_trace[-1] if metrics.ate_trace else 0.0
    metrics.rpe_mean = relative_pose_error(metrics.true_traj, metrics.est_traj)
    metrics.mission_time_steps = steps
    metrics.path_length = info["path_length"]
    metrics.info_gain_per_meter = (
        metrics.entropy_reduction / metrics.path_length if metrics.path_length > 1e-6 else 0.0
    )
    metrics.collision_count = info["collision_count"]
    return metrics


def summarize_runs(runs: List[EpisodeMetrics]) -> dict:
    """Mean +/- std across a list of episode rollouts (e.g. N seeds), for
    a compact table comparing policies."""
    def agg(attr):
        vals = [getattr(r, attr) for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    fields = ["completeness", "ate_mean", "rpe_mean", "path_length",
              "info_gain_per_meter", "collision_count", "total_reward",
              "loop_closures_validated"]
    return {f: agg(f) for f in fields}
