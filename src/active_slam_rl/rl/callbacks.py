"""
Metrics-logging callback.

SB3's own Monitor wrapper logs reward/length; this callback additionally
pulls the SLAM-specific metrics out of `info` at the end of every episode
(map completeness, ATE, collision count, loop closures, ...) and writes
them to a CSV so metrics/plotting.py can produce the dashboards the thesis
asks for (section 8: Mapping Quality / Localization / Exploration /
Safety) without needing to re-run anything.
"""

from __future__ import annotations

import csv
import os

from stable_baselines3.common.callbacks import BaseCallback


class MetricsLoggingCallback(BaseCallback):
    FIELDNAMES = [
        "timestep", "episode", "env_idx", "episode_reward", "episode_length",
        "map_completeness", "final_ate", "mean_ate", "path_length",
        "collision_count", "map_entropy", "loop_closures_validated",
        # Registration / fusion diagnostics -- see
        # metrics/plotting.py's plot_registration_and_fusion_diagnostics
        # and env/sonar_model.py's "MODALITY TAGGING" section for why
        # imaging vs scanning are broken out separately rather than a
        # single pooled q_t mean.
        "mean_q_t_imaging", "mean_q_t_scanning",
        "fs2d_rejected_outlier_rate", "used_fs2d_rate",
    ]

    def __init__(self, log_dir: str, verbose: int = 0,
                 live_plot: bool = True, plot_every_episodes: int = 5,
                 plot_out_dir: str | None = None):
        """
        live_plot: if True (default), regenerate `training_curves.png`
            every `plot_every_episodes` completed episodes, overwriting the
            same file in place. Point an image viewer at it (most viewers,
            and VS Code's built-in one, auto-refresh on file change) to
            watch training progress live without waiting for it to finish.
            Whatever the plot looks like when training ends *is* the final
            saved PNG -- no separate "save at the end" step needed.
        plot_every_episodes: how often to refresh the PNG. Plotting is
            cheap relative to an RL step, but this still avoids re-plotting
            on every single episode when episodes are very short.
        plot_out_dir: where `training_curves.png` is written; defaults to
            the parent of `log_dir` (matching scripts/run_training.py's
            layout: log_dir="results/train_logs", plot -> "results/").
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "episode_metrics.csv")
        self._episode_counts = {}
        self._ate_running = {}
        self._loop_closure_counts = {}
        self._q_t_by_mode = {}      # env_idx -> {"imaging": [...], "scanning": [...]}
        self._fs2d_rejected = {}    # env_idx -> [bool, ...]
        self._fs2d_used = {}        # env_idx -> [bool, ...]
        os.makedirs(log_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

        self.live_plot = live_plot
        self.plot_every_episodes = max(1, plot_every_episodes)
        self.plot_out_dir = plot_out_dir or os.path.dirname(os.path.normpath(log_dir)) or "."
        self._total_episodes_logged = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            self._ate_running.setdefault(i, []).append(info.get("ate", 0.0))
            if info.get("loop_closure_validated"):
                self._loop_closure_counts[i] = self._loop_closure_counts.get(i, 0) + 1

            frame_mode = info.get("frame_mode")
            if frame_mode in ("imaging", "scanning"):
                self._q_t_by_mode.setdefault(i, {"imaging": [], "scanning": []})
                self._q_t_by_mode[i][frame_mode].append(info.get("q_t", 0.0))
            sfm_info = info.get("sfm")
            if sfm_info is not None:
                self._fs2d_rejected.setdefault(i, []).append(bool(sfm_info["fs2d_rejected_outlier"]))
                self._fs2d_used.setdefault(i, []).append(bool(sfm_info["used_fs2d"]))

            done = dones[i] if i < len(dones) else False
            if done and "episode" in info:  # Monitor injects this on episode end
                self._episode_counts[i] = self._episode_counts.get(i, 0) + 1
                ate_hist = self._ate_running.get(i, [0.0])
                q_t_hist = self._q_t_by_mode.get(i, {"imaging": [], "scanning": []})
                rejected_hist = self._fs2d_rejected.get(i, [])
                used_hist = self._fs2d_used.get(i, [])
                row = {
                    "timestep": self.num_timesteps,
                    "episode": self._episode_counts[i],
                    "env_idx": i,
                    "episode_reward": info["episode"]["r"],
                    "episode_length": info["episode"]["l"],
                    "map_completeness": info.get("map_completeness", 0.0),
                    "final_ate": ate_hist[-1],
                    "mean_ate": sum(ate_hist) / len(ate_hist),
                    "path_length": info.get("path_length", 0.0),
                    "collision_count": info.get("collision_count", 0),
                    "map_entropy": info.get("map_entropy", 0.0),
                    "loop_closures_validated": self._loop_closure_counts.get(i, 0),
                    "mean_q_t_imaging": (sum(q_t_hist["imaging"]) / len(q_t_hist["imaging"])
                                         if q_t_hist["imaging"] else float("nan")),
                    "mean_q_t_scanning": (sum(q_t_hist["scanning"]) / len(q_t_hist["scanning"])
                                          if q_t_hist["scanning"] else float("nan")),
                    "fs2d_rejected_outlier_rate": (sum(rejected_hist) / len(rejected_hist)
                                                   if rejected_hist else float("nan")),
                    "used_fs2d_rate": (sum(used_hist) / len(used_hist)
                                       if used_hist else float("nan")),
                }
                with open(self.csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)
                self._ate_running[i] = []
                self._loop_closure_counts[i] = 0
                self._q_t_by_mode[i] = {"imaging": [], "scanning": []}
                self._fs2d_rejected[i] = []
                self._fs2d_used[i] = []

                self._total_episodes_logged += 1
                if self.live_plot and self._total_episodes_logged % self.plot_every_episodes == 0:
                    self._refresh_plot()
        return True

    def _refresh_plot(self):
        # Imported lazily so a training run that never plots doesn't pay
        # matplotlib's import cost, and so a plotting failure (e.g. too few
        # rows for a rolling window edge case) can't crash training itself.
        try:
            from active_slam_rl.metrics.plotting import plot_training_curves, plot_training_diagnostics_curves
            plot_training_curves(self.csv_path, self.plot_out_dir)
            plot_training_diagnostics_curves(self.csv_path, self.plot_out_dir)
        except Exception as e:
            if self.verbose:
                print(f"[MetricsLoggingCallback] live plot refresh skipped: {e}")

    def _on_training_end(self) -> None:
        # Guarantee the very last episodes are reflected in the saved PNG
        # even if the run didn't end on a plot_every_episodes boundary.
        if self.live_plot:
            self._refresh_plot()
