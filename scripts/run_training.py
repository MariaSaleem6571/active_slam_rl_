#!/usr/bin/env python3
"""
Train the PPO active-SLAM policy.

Usage:
    python scripts/run_training.py --config configs/default.yaml --timesteps 200000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from active_slam_rl.utils.config import load_config
from active_slam_rl.rl.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None,
                         help="override training.total_timesteps from the config")
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--no_live_plot", action="store_true",
                         help="disable the auto-refreshing training_curves.png during training")
    parser.add_argument("--plot_every_episodes", type=int, default=5,
                         help="how often (in completed episodes) to refresh training_curves.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training_cfg = cfg["training"]
    if args.timesteps is not None:
        training_cfg["total_timesteps"] = args.timesteps
    if args.n_envs is not None:
        training_cfg["n_envs"] = args.n_envs

    model, callback = train(
        total_timesteps=training_cfg["total_timesteps"],
        n_envs=training_cfg["n_envs"],
        latent_dim=training_cfg["latent_dim"],
        learning_rate=training_cfg["learning_rate"],
        n_steps=training_cfg["n_steps"],
        batch_size=training_cfg["batch_size"],
        log_dir=training_cfg["log_dir"],
        model_out=training_cfg["model_out"],
        env_config=cfg["env"],
        seed=training_cfg["seed"],
        live_plot=not args.no_live_plot,
        plot_every_episodes=args.plot_every_episodes,
    )
    print(f"Training complete. Model saved to {training_cfg['model_out']}")
    print(f"Episode metrics logged to {os.path.join(training_cfg['log_dir'], 'episode_metrics.csv')}")
    if not args.no_live_plot:
        plot_path = os.path.join(os.path.dirname(os.path.normpath(training_cfg["log_dir"])) or ".",
                                  "training_curves.png")
        print(f"Live-updated training plot (final state) saved to {plot_path}")


if __name__ == "__main__":
    main()
