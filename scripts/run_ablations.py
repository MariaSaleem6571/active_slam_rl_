#!/usr/bin/env python3
"""
Run a standard active-SLAM ablation study: train + evaluate each config in
configs/ablations/ (or a chosen subset) across multiple seeds, and produce
the mean +/- std comparison table and bar-chart figure papers in this area
typically report (e.g. Chaplot et al.'s "Learning to Explore using Active
Neural SLAM", or Placed et al.'s active-SLAM survey's own comparison
tables) -- one row per ablated component, one column per headline metric.

configs/ablations/full.yaml is the "everything on" reference every other
config in that directory differs from in exactly one respect -- see each
config's own header comment for which one. This script doesn't assume
that naming convention beyond using "full" as the label for whichever
config is literally named full.yaml; add a new ablation by dropping
another single-factor-changed YAML file into configs/ablations/, no code
changes needed.

Usage:
    # Everything in configs/ablations/, 3 seeds each (default) -- this is
    # a genuine but fast smoke-scale study (quick_demo.yaml scale), good
    # for confirming the pipeline and getting a first read on directionality.
    python scripts/run_ablations.py

    # A real, paper-scale study: more seeds, longer training. Swap in
    # default.yaml-scale settings by pointing --configs at copies of
    # configs/ablations/*.yaml with default.yaml's world/sonar sizes
    # (see quick_demo.yaml's own header comment on why a model's
    # world/sonar/patch_size can't be mixed across configs).
    python scripts/run_ablations.py --seeds 5 --timesteps 100000 --episodes 10

    # Just a couple of variants while iterating
    python scripts/run_ablations.py --configs full no_loop_closure

Writes:
    results/ablations/<name>/seed_<k>/ppo_active_slam.zip   (per seed model)
    results/ablations/<name>/seed_<k>/train_logs/...         (per seed logs)
    results/ablations/ablation_results.csv                   (the table)
    results/ablations/ablation_comparison.png                (the figure)
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from stable_baselines3 import PPO

from active_slam_rl.utils.config import load_config
from active_slam_rl.env.sim_env import ActiveSlamEnv
from active_slam_rl.rl.train import train
from active_slam_rl.metrics.evaluation import rollout_episode, summarize_runs
from active_slam_rl.metrics.plotting import plot_ablation_comparison

METRICS = ["completeness", "ate_mean", "path_length", "info_gain_per_meter",
           "collision_count", "total_reward", "loop_closures_validated"]


class SB3PolicyAdapter:
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

    def reset(self):
        pass


def run_one(config_path: str, name: str, seed: int, timesteps: int, episodes: int,
            out_root: str, stochastic_eval: bool):
    seed_dir = os.path.join(out_root, name, f"seed_{seed}")
    log_dir = os.path.join(seed_dir, "train_logs")
    model_out = os.path.join(seed_dir, "ppo_active_slam.zip")
    os.makedirs(log_dir, exist_ok=True)

    cfg = load_config(config_path)
    training_cfg = cfg["training"]

    print(f"[{name}] seed {seed}: training {timesteps} timesteps ...")
    train(
        total_timesteps=timesteps,
        n_envs=training_cfg["n_envs"],
        latent_dim=training_cfg["latent_dim"],
        learning_rate=training_cfg["learning_rate"],
        n_steps=training_cfg["n_steps"],
        batch_size=training_cfg["batch_size"],
        log_dir=log_dir,
        model_out=model_out,
        env_config=cfg["env"],
        seed=seed,
        live_plot=False,
    )

    print(f"[{name}] seed {seed}: evaluating over {episodes} episodes ...")
    model = PPO.load(model_out)
    policy = SB3PolicyAdapter(model)
    env = ActiveSlamEnv(cfg["env"])
    runs = []
    for ep in range(episodes):
        env.cfg = cfg["env"].__class__(**{**cfg["env"].__dict__, "seed": 5000 + seed * 100 + ep})
        m = rollout_episode(env, policy, deterministic=not stochastic_eval)
        runs.append(m)
    return summarize_runs(runs)   # dict[metric] -> (mean, std) over episodes, this seed


def aggregate_across_seeds(per_seed_summaries):
    """per_seed_summaries: list of dicts (one per seed), each
    metric -> (mean, std) over episodes. Returns metric -> (mean, std)
    over seeds' means -- the between-seed variance is what a paper-style
    ablation table reports, since within-episode variance is a different
    (smaller, less interesting) source of noise than between-seed
    variance from different random initializations/training runs."""
    result = {}
    for metric in METRICS:
        seed_means = [s[metric][0] for s in per_seed_summaries]
        result[metric] = (float(np.mean(seed_means)), float(np.std(seed_means)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None,
                         help="ablation names to run, e.g. 'full no_loop_closure' "
                              "(without .yaml) -- default: everything in configs/ablations/")
    parser.add_argument("--ablations_dir", default="configs/ablations")
    parser.add_argument("--seeds", type=int, default=3,
                         help="number of training seeds per ablation (default 3 -- "
                              "a quick smoke-scale study; use 5+ for a paper-scale study)")
    parser.add_argument("--timesteps", type=int, default=None,
                         help="override each config's training.total_timesteps")
    parser.add_argument("--episodes", type=int, default=5,
                         help="evaluation episodes per trained model")
    parser.add_argument("--out_root", default="results/ablations")
    parser.add_argument("--stochastic_eval", action="store_true",
                         help="sample actions during eval instead of argmax -- see "
                              "run_eval.py's --stochastic for why this matters for "
                              "undertrained policies")
    args = parser.parse_args()

    if args.configs:
        names = args.configs
    else:
        paths = sorted(glob.glob(os.path.join(args.ablations_dir, "*.yaml")))
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if not names:
        raise SystemExit(f"No ablation configs found in {args.ablations_dir}")

    os.makedirs(args.out_root, exist_ok=True)
    table = {}
    for name in names:
        config_path = os.path.join(args.ablations_dir, f"{name}.yaml")
        if not os.path.exists(config_path):
            raise SystemExit(f"No such ablation config: {config_path}")
        timesteps = args.timesteps or load_config(config_path)["training"]["total_timesteps"]

        per_seed_summaries = []
        for seed in range(args.seeds):
            summary = run_one(config_path, name, seed, timesteps, args.episodes,
                               args.out_root, args.stochastic_eval)
            per_seed_summaries.append(summary)
        table[name] = aggregate_across_seeds(per_seed_summaries)

    csv_path = os.path.join(args.out_root, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ablation"] + [f"{m}_mean" for m in METRICS] + [f"{m}_std" for m in METRICS])
        for name in names:
            row = [name]
            row += [table[name][m][0] for m in METRICS]
            row += [table[name][m][1] for m in METRICS]
            writer.writerow(row)

    print(f"\n=== Ablation results (mean +/- std over {args.seeds} seeds) ===")
    for name in names:
        print(f"\n{name}:")
        for m in METRICS:
            mean, std = table[name][m]
            print(f"  {m:24s} {mean:8.3f} +/- {std:.3f}")

    plot_path = plot_ablation_comparison(table, args.out_root, reference=names[0])
    print(f"\nResults table: {csv_path}")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
