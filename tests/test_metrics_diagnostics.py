"""Smoke tests for the registration/fusion diagnostics added to
metrics/evaluation.py, metrics/plotting.py, and rl/callbacks.py.

These don't check exact pixel content (plots are for humans), just that
the full data path -- env.step() -> info dict -> EpisodeMetrics /
MetricsLoggingCallback -> plotting -- runs end to end and produces a file,
so a future refactor of any one piece can't silently break the others.
"""
import csv
import os

import numpy as np

from active_slam_rl.env.sim_env import ActiveSlamEnv, EnvConfig
from active_slam_rl.env.world_generator import WorldConfig
from active_slam_rl.metrics.evaluation import rollout_episode
from active_slam_rl.metrics.plotting import (
    plot_registration_and_fusion_diagnostics,
    plot_sonar_frame_comparison,
)


def _make_env(seed=0):
    cfg = EnvConfig(world=WorldConfig(height=80, width=80, n_steps=250, seed=seed),
                     max_steps=40, seed=seed)
    return ActiveSlamEnv(cfg)


class _RandomPolicy:
    def __init__(self, n_actions, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def predict(self, obs, deterministic=True):
        return self.rng.integers(0, self.n_actions), None


def test_info_dict_carries_frame_mode():
    env = _make_env()
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["frame_mode"] in ("imaging", "scanning")


def test_episode_metrics_traces_populated_by_rollout():
    env = _make_env()
    metrics = rollout_episode(env, _RandomPolicy(env.action_space.n), max_steps=30)
    assert len(metrics.q_t_trace) == 30
    assert len(metrics.frame_mode_trace) == 30
    assert set(metrics.frame_mode_trace) <= {"imaging", "scanning", None}
    # use_sfm_fusion defaults to True (see configs/default.yaml) -> fusion
    # diagnostics should be populated every step.
    if env.cfg.use_sfm_fusion:
        assert len(metrics.bias_estimate_deg_trace) == 30
        assert len(metrics.fs2d_rejected_outlier_trace) == 30


def test_plot_registration_and_fusion_diagnostics(tmp_path):
    env = _make_env()
    metrics = rollout_episode(env, _RandomPolicy(env.action_space.n), max_steps=30)
    out_path = plot_registration_and_fusion_diagnostics(metrics, str(tmp_path))
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_sonar_frame_comparison(tmp_path):
    env = _make_env()
    env.reset()
    out_path = plot_sonar_frame_comparison(env, str(tmp_path))
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_metrics_logging_callback_writes_modality_columns(tmp_path):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from active_slam_rl.rl.callbacks import MetricsLoggingCallback

    log_dir = str(tmp_path / "logs")
    vec_env = DummyVecEnv([lambda: Monitor(_make_env())])
    callback = MetricsLoggingCallback(log_dir=log_dir, live_plot=False)
    callback.init_callback(_DummyModel(vec_env))

    obs = vec_env.reset()
    for _ in range(80):  # a couple of short (max_steps=40) episodes' worth
        actions = np.array([vec_env.action_space.sample()])
        obs, rewards, dones, infos = vec_env.step(actions)
        callback.locals = {"infos": infos, "dones": dones, "rewards": rewards}
        callback.num_timesteps += 1
        callback._on_step()

    with open(callback.csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    for col in ("mean_q_t_imaging", "mean_q_t_scanning",
                "fs2d_rejected_outlier_rate", "used_fs2d_rate"):
        assert col in rows[0]


class _DummyModel:
    """Just enough of SB3's model interface for BaseCallback.init_callback."""
    def __init__(self, env):
        self.env = env
        self.num_timesteps = 0


def test_plot_ablation_comparison_smoke(tmp_path):
    from active_slam_rl.metrics.plotting import plot_ablation_comparison

    table = {
        "full": {"completeness": (85.0, 3.0), "ate_mean": (1.2, 0.2), "total_reward": (150.0, 20.0),
                 "collision_count": (2.0, 1.0), "loop_closures_validated": (3.0, 1.0)},
        "no_loop_closure": {"completeness": (80.0, 4.0), "ate_mean": (2.5, 0.5), "total_reward": (90.0, 15.0),
                             "collision_count": (3.0, 1.5), "loop_closures_validated": (0.0, 0.0)},
    }
    out_path = plot_ablation_comparison(table, str(tmp_path), reference="full")
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
