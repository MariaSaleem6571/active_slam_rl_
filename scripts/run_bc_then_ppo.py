#!/usr/bin/env python3
"""
Full thesis section 5.9 Stage 1 -> PPO pipeline:
  1. Collect demonstrations from the Frontier-Based heuristic.
  2. Behavioral-clone the policy (with its joint E_phi encoder) onto them.
  3. Continue training with PPO from that bootstrapped starting point.

Usage:
    python scripts/run_bc_then_ppo.py --bc_episodes 25 --ppo_timesteps 20000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from active_slam_rl.utils.config import load_config
from active_slam_rl.env.sim_env import ActiveSlamEnv
from active_slam_rl.rl.baselines import FrontierBasedPolicy
from active_slam_rl.rl.behavioral_cloning import collect_demonstrations, behavioral_clone, bc_action_accuracy
from active_slam_rl.rl.train import build_vec_env
from active_slam_rl.state.encoder import SB3StateEncoderExtractor
from active_slam_rl.rl.callbacks import MetricsLoggingCallback
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bc_episodes", type=int, default=25)
    parser.add_argument("--bc_epochs", type=int, default=8)
    parser.add_argument("--ppo_timesteps", type=int, default=20000)
    parser.add_argument("--log_dir", default="results/train_logs_bc")
    parser.add_argument("--model_out", default="results/ppo_active_slam_bc.zip")
    args = parser.parse_args()

    cfg = load_config(args.config)["env"]
    os.makedirs(args.log_dir, exist_ok=True)

    print("=== Stage 1: collecting frontier-based demonstrations ===")
    demo_env = ActiveSlamEnv(cfg)
    expert = FrontierBasedPolicy(demo_env.action_space)
    dataset = collect_demonstrations(demo_env, expert, n_episodes=args.bc_episodes)
    print(f"Collected {dataset['action'].shape[0]} (obs, action) pairs.")

    print("=== Stage 1: behavioral cloning onto E_phi + policy ===")
    vec_env = build_vec_env(cfg, n_envs=1, log_dir=args.log_dir, base_seed=0)
    policy_kwargs = dict(
        features_extractor_class=SB3StateEncoderExtractor,
        features_extractor_kwargs=dict(latent_dim=128),
        net_arch=dict(pi=[64], vf=[64]),
    )
    model = PPO(policy="MultiInputPolicy", env=vec_env, learning_rate=3e-4,
                n_steps=256, batch_size=64, policy_kwargs=policy_kwargs, verbose=0, seed=0)

    acc_before = bc_action_accuracy(model, dataset)
    behavioral_clone(model, dataset, n_epochs=args.bc_epochs)
    acc_after = bc_action_accuracy(model, dataset)
    print(f"Greedy action match vs. expert: {acc_before:.2%} -> {acc_after:.2%}")

    print("=== Stage 2: PPO fine-tuning from the BC-bootstrapped policy ===")
    callback = MetricsLoggingCallback(log_dir=args.log_dir)
    model.learn(total_timesteps=args.ppo_timesteps, callback=callback, reset_num_timesteps=False)
    model.save(args.model_out)
    vec_env.close()
    print(f"Saved BC+PPO model to {args.model_out}")


if __name__ == "__main__":
    main()
