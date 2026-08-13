import numpy as np

from active_slam_rl.env.sim_env import ActiveSlamEnv, EnvConfig
from active_slam_rl.env.world_generator import WorldConfig


def _make_env(seed=0):
    cfg = EnvConfig(world=WorldConfig(height=80, width=80, n_steps=250, seed=seed),
                     max_steps=40, seed=seed)
    return ActiveSlamEnv(cfg)


def test_reset_returns_valid_observation():
    env = _make_env()
    obs, info = env.reset()
    assert env.observation_space.contains(obs)


def test_step_returns_valid_observation_and_scalar_reward():
    env = _make_env()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) or np.isscalar(reward)
    assert "map_completeness" in info
    assert "ate" in info


def test_episode_terminates_within_max_steps():
    env = _make_env()
    obs, info = env.reset()
    for i in range(60):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert (terminated or truncated)
    assert i <= env.cfg.max_steps


def test_deterministic_with_seed():
    env1 = _make_env(seed=42)
    env2 = _make_env(seed=42)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert np.allclose(obs1["patch"], obs2["patch"])
