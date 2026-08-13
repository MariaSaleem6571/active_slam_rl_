#!/usr/bin/env python3
"""
Render an animated GIF of one rollout so you can *watch* the algorithm
build the map, drift, and correct itself via loop closure.

Usage:
    python scripts/visualize_demo.py --model results/ppo_active_slam.zip
    python scripts/visualize_demo.py --policy frontier   # no trained model needed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from active_slam_rl.utils.config import load_config
from active_slam_rl.env.sim_env import ActiveSlamEnv
from active_slam_rl.rl.baselines import RandomWalkPolicy, FrontierBasedPolicy, NextBestViewPolicy
from active_slam_rl.visualization.live_viewer import record_rollout_frames, render_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--policy", default="frontier", choices=["random", "frontier", "nbv"])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--out", default="results/eval/rollout_demo.gif")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions instead of taking the argmax -- use this for "
                              "early-training checkpoints, where the argmax can be a degenerate "
                              "fixed point (e.g. stuck spinning in place) even though the policy "
                              "behaves reasonably when sampled")
    args = parser.parse_args()

    cfg = load_config(args.config)["env"]
    cfg = cfg.__class__(**{**cfg.__dict__, "seed": args.seed})
    env = ActiveSlamEnv(cfg)

    if args.model:
        from stable_baselines3 import PPO

        class Adapter:
            def __init__(self, model):
                self.model = model
            def predict(self, obs, deterministic=True):
                return self.model.predict(obs, deterministic=deterministic)
            def reset(self):
                pass

        policy = Adapter(PPO.load(args.model))
    else:
        policy = {
            "random": RandomWalkPolicy(env.action_space),
            "frontier": FrontierBasedPolicy(env.action_space),
            "nbv": NextBestViewPolicy(env.action_space),
        }[args.policy]

    frames, world_occ = record_rollout_frames(env, policy, max_steps=args.steps,
                                                deterministic=not args.stochastic)
    out_path = render_gif(frames, world_occ, args.out)
    print(f"Saved rollout animation to {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
