"""Loads configs/*.yaml into the dataclasses used across the codebase, so
there is exactly one place ("configs/default.yaml") to change
hyperparameters without touching any Python file."""

from __future__ import annotations

import yaml

from active_slam_rl.env.world_generator import WorldConfig
from active_slam_rl.env.sonar_model import SonarConfig
from active_slam_rl.env.imu_dvl_model import IMUConfig, DVLConfig
from active_slam_rl.env.reward import RewardWeights
from active_slam_rl.env.sim_env import EnvConfig
from active_slam_rl.fusion.sfm import SfMConfig
from active_slam_rl.perception.sfm2d import SfM2DConfig


def load_config(path: str):
    with open(path) as f:
        raw = yaml.safe_load(f)

    world_cfg = WorldConfig(**{**WorldConfig().__dict__, **raw.get("world", {}),
                                "corridor_width_range": tuple(raw["world"]["corridor_width_range"])})
    sonar_cfg = SonarConfig(**{**SonarConfig().__dict__, **raw.get("sonar", {})})
    reward_cfg = RewardWeights(**{**RewardWeights().__dict__, **raw.get("reward_weights", {})})
    # IMU/DVL sensing + the SfM fusion EKF (see env/imu_dvl_model.py,
    # fusion/sfm.py) -- all three sections are optional in the YAML file;
    # anything not specified falls back to the dataclass defaults, same
    # as every other section here.
    imu_cfg = IMUConfig(**{**IMUConfig().__dict__, **raw.get("imu", {})})
    dvl_cfg = DVLConfig(**{**DVLConfig().__dict__, **raw.get("dvl", {})})
    sfm_cfg = SfMConfig(**{**SfMConfig().__dict__, **raw.get("sfm", {})})
    # perception/sfm2d.py's StructureFromMotion2D -- the actual
    # Structure-from-Motion, NOT the same thing as "sfm" above (see that
    # module's docstring for the naming disambiguation). Also optional;
    # falls back to SfM2DConfig()'s defaults. Whether it's used at all is
    # governed separately by EnvConfig.use_sfm2d (an `env:` section key,
    # picked up by the generic passthrough below) -- this section only
    # tunes its internals (match_gate_m etc.) for when it *is* enabled.
    sfm2d_cfg = SfM2DConfig(**{**SfM2DConfig().__dict__, **raw.get("sfm2d", {})})
    env_raw = raw.get("env", {})
    env_cfg = EnvConfig(
        world=world_cfg, sonar=sonar_cfg, reward_weights=reward_cfg,
        imu=imu_cfg, dvl=dvl_cfg, sfm=sfm_cfg, sfm2d=sfm2d_cfg,
        **{k: v for k, v in env_raw.items()},
    )
    training_cfg = raw.get("training", {})
    return {"env": env_cfg, "training": training_cfg}
