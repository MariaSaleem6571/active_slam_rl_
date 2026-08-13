"""
Train the PPO policy (thesis section 5.6) with the joint state encoder
(thesis section 5.5, feedback loop G: policy gradients refine E_phi).

Usage:
    python scripts/run_training.py --config configs/default.yaml

This script is intentionally simulator-agnostic: it builds a
`gymnasium.Env` from `active_slam_rl.env.sim_env.ActiveSlamEnv` (the
lightweight environment that runs anywhere, right now) but the exact same
PPO/SB3 code works unmodified against `active_slam_rl.env.marinegym_env`
once MarineGym/Isaac Sim is available -- see that file's docstring for the
one-line swap.
"""

from __future__ import annotations

import os
from dataclasses import replace

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from active_slam_rl.env.sim_env import ActiveSlamEnv, EnvConfig
from active_slam_rl.rl.callbacks import MetricsLoggingCallback
from active_slam_rl.state.encoder import SB3StateEncoderExtractor


def make_env(env_config: EnvConfig, seed: int, log_dir: str):
    def _init():
        cfg = replace(env_config, seed=seed)
        env = ActiveSlamEnv(cfg)
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{seed}"))
        return env
    return _init


def build_vec_env(env_config: EnvConfig, n_envs: int, log_dir: str, base_seed: int = 0):
    os.makedirs(log_dir, exist_ok=True)
    env_fns = [make_env(env_config, base_seed + i, log_dir) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def train(
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    latent_dim: int = 128,
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 256,
    log_dir: str = "results/train_logs",
    model_out: str = "results/ppo_active_slam.zip",
    env_config: EnvConfig = EnvConfig(),
    verbose: int = 1,
    seed: int = 0,
    live_plot: bool = True,
    plot_every_episodes: int = 5,
):
    vec_env = build_vec_env(env_config, n_envs, log_dir, base_seed=seed)

    policy_kwargs = dict(
        features_extractor_class=SB3StateEncoderExtractor,
        features_extractor_kwargs=dict(latent_dim=latent_dim),
        net_arch=dict(pi=[64], vf=[64]),  # small heads on top of the shared E_phi features
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
    )

    callback = MetricsLoggingCallback(log_dir=log_dir, live_plot=live_plot,
                                       plot_every_episodes=plot_every_episodes)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)
    vec_env.close()
    return model, callback
