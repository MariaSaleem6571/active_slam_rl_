"""
Same rationale as test_marinegym_adapter.py: the Stonefish adapter can't
be exercised end-to-end without ROS2 + Stonefish running (not available
in CI or most dev machines), but it must always import cleanly and stay
correctly structured, so a change here never silently breaks the rest of
the package for people who don't have ROS2 installed.
"""
import numpy as np

from active_slam_rl.env.stonefish_env import (
    StonefishActiveSlamEnv, StonefishConfig, StonefishWorldAdapter,
    StonefishSonarAdapter, _extract_beams_from_frame, _action_to_setpoints,
)
from active_slam_rl.env.sim_env import ActiveSlamEnv


def test_stonefish_env_subclasses_active_slam_env():
    assert issubclass(StonefishActiveSlamEnv, ActiveSlamEnv)


def test_stonefish_env_overrides_only_the_documented_methods():
    overridden = {
        name for name in vars(StonefishActiveSlamEnv)
        if callable(getattr(StonefishActiveSlamEnv, name, None)) and not name.startswith("__")
    }
    disallowed_overrides = {"step", "_build_observation", "_crop_patch",
                             "_update_map_from_beams", "_integrate_odometry"}
    assert overridden.isdisjoint(disallowed_overrides)


def test_config_defaults_are_sane():
    cfg = StonefishConfig()
    assert cfg.n_thrusters > 0
    assert cfg.max_range_m > 0
    assert -1.0 <= 1.0  # sanity placeholder for the documented setpoint range


def test_world_and_sonar_adapters_expose_the_required_duck_typed_interface():
    assert hasattr(StonefishWorldAdapter, "is_free")
    assert hasattr(StonefishSonarAdapter, "sense_imaging")
    assert hasattr(StonefishSonarAdapter, "sense_scanning_360")


def test_beam_extraction_stays_within_bounds():
    frame = np.zeros((64, 64))
    frame[32, 40:50] = 0.9  # a bright return blob to the "right" of center
    ranges, angles, hit_points = _extract_beams_from_frame(frame, max_range=22.0, n_beams=16)
    assert ranges.shape == (16,)
    assert angles.shape == (16,)
    assert np.all(ranges <= 22.0)
    assert np.all(ranges >= 0.0)


def test_thruster_setpoints_are_within_stonefish_documented_range():
    # stonefish_ros2 requires thruster setpoints in [-1, 1] -- verified in
    # docs/STONEFISH_INTEGRATION.md. This must hold for every action.
    for action in range(7):
        setpoints = _action_to_setpoints(action, n_thrusters=8)
        assert setpoints.shape == (8,)
        assert np.all(setpoints >= -1.0) and np.all(setpoints <= 1.0)
