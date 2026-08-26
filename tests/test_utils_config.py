"""Regression tests for utils/config.py's load_config -- specifically that
an optional `sfm2d:` YAML section and env.use_sfm2d/sfm2d_apply_correction_from/
sfm2d_correction_gain round-trip into EnvConfig correctly. Added alongside
perception/sfm2d.py's env integration since load_config previously had no
idea SfM2DConfig existed (it would have silently ignored a `sfm2d:`
section), unlike every other nested config (world/sonar/imu/dvl/sfm) which
was already wired through.
"""
import os

from active_slam_rl.perception.sfm2d import SfM2DConfig
from active_slam_rl.utils.config import load_config

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_default_yaml_sfm2d_defaults_match_dataclass():
    cfg = load_config(os.path.join(_CONFIGS_DIR, "default.yaml"))
    env_cfg = cfg["env"]
    assert env_cfg.use_sfm2d is False   # off by default -- see EnvConfig's docstring
    assert isinstance(env_cfg.sfm2d, SfM2DConfig)
    assert env_cfg.sfm2d.match_gate_m == SfM2DConfig().match_gate_m
    assert env_cfg.sfm2d.min_matched_for_correction == SfM2DConfig().min_matched_for_correction


def test_quick_demo_sfm2d_yaml_enables_sfm2d():
    cfg = load_config(os.path.join(_CONFIGS_DIR, "quick_demo_sfm2d.yaml"))
    env_cfg = cfg["env"]
    assert env_cfg.use_sfm2d is True
    assert env_cfg.sfm2d_apply_correction_from == "both"


def test_sfm2d_yaml_section_overrides_are_applied():
    """A minimal in-memory config exercising the same load_config path
    with a custom sfm2d: override, without depending on the checked-in
    YAML files' exact contents staying the same forever."""
    import tempfile
    import yaml as pyyaml

    base = load_config(os.path.join(_CONFIGS_DIR, "quick_demo.yaml"))
    with open(os.path.join(_CONFIGS_DIR, "quick_demo.yaml")) as f:
        raw = pyyaml.safe_load(f)
    raw["sfm2d"] = {"match_gate_m": 0.99, "min_matched_for_correction": 12}
    raw["env"]["use_sfm2d"] = True

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        pyyaml.safe_dump(raw, f)
        tmp_path = f.name
    try:
        cfg = load_config(tmp_path)
        assert cfg["env"].use_sfm2d is True
        assert cfg["env"].sfm2d.match_gate_m == 0.99
        assert cfg["env"].sfm2d.min_matched_for_correction == 12
    finally:
        os.unlink(tmp_path)
