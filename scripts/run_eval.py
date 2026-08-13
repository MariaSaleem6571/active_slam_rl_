#!/usr/bin/env python3
"""
Evaluate a trained policy (or the untrained baselines) and produce the
metric plots from thesis section 8.

Usage:
    python scripts/run_eval.py --model results/ppo_active_slam.zip --episodes 5
    python scripts/run_eval.py --baselines-only   # skip PPO, just compare heuristics
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from stable_baselines3 import PPO

from active_slam_rl.utils.config import load_config
from active_slam_rl.env.sim_env import ActiveSlamEnv
from active_slam_rl.rl.baselines import RandomWalkPolicy, FrontierBasedPolicy, NextBestViewPolicy
from active_slam_rl.metrics.evaluation import rollout_episode, summarize_runs
from active_slam_rl.metrics.plotting import plot_baseline_comparison, plot_episode_dashboard


class SB3PolicyAdapter:
    """Wraps an SB3 model so it exposes the same .predict(obs) contract as
    the heuristic baselines (SB3 already matches this, this class exists
    purely for a uniform `.reset()` no-op)."""
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

    def reset(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="path to a trained PPO .zip; omit to skip RL policy")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out_dir", default="results/eval")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions instead of taking the argmax -- use this for "
                              "early-training checkpoints, where the argmax can be a degenerate "
                              "fixed point even though the policy behaves reasonably when sampled")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)["env"]

    env = ActiveSlamEnv(cfg)

    policies = {
        "Random Walk": RandomWalkPolicy(env.action_space),
        "Frontier-Based": FrontierBasedPolicy(env.action_space),
        "Next-Best-View": NextBestViewPolicy(env.action_space),
    }
    if args.model and not args.baselines_only:
        model = PPO.load(args.model)
        policies["RL (PPO)"] = SB3PolicyAdapter(model)

    all_runs = {}
    for name, policy in policies.items():
        print(f"Evaluating {name} ...")
        runs = []
        for ep in range(args.episodes):
            env.cfg = cfg.__class__(**{**cfg.__dict__, "seed": 1000 + ep})
            m = rollout_episode(env, policy, deterministic=not args.stochastic)
            runs.append(m)
            print(f"  episode {ep}: completeness={m.completeness:.1f}% "
                  f"ATE={m.ate_mean:.2f}m reward={m.total_reward:.1f}")
        all_runs[name] = runs
        # Save a dashboard for this policy's last episode.
        plot_episode_dashboard(runs[-1], env.world.occ,
                                out_dir=args.out_dir, name=f"dashboard_{name.replace(' ', '_')}")

    summaries = {name: summarize_runs(runs) for name, runs in all_runs.items()}
    print("\n=== Summary (mean +/- std over episodes) ===")
    for name, summary in summaries.items():
        print(f"\n{name}:")
        for k, (mean, std) in summary.items():
            print(f"  {k:24s} {mean:8.3f} +/- {std:.3f}")

    plot_path = plot_baseline_comparison(summaries, args.out_dir)
    print(f"\nComparison plot saved to {plot_path}")


if __name__ == "__main__":
    main()
