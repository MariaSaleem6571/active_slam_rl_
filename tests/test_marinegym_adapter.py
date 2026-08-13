"""
The MarineGym adapter can't be exercised end-to-end without Isaac Sim and
a GPU (not available in CI or most dev machines), but it must always
*import* cleanly and correctly subclass ActiveSlamEnv, so a change here
never silently breaks the rest of the package for people who don't have
MarineGym installed.
"""
import inspect

from active_slam_rl.env.marinegym_env import (
    MarineGymActiveSlamEnv, MarineGymTaskConfig, MarineGymWorldAdapter,
    MarineGymSonarAdapter,
)
from active_slam_rl.env.sim_env import ActiveSlamEnv


def test_marinegym_env_subclasses_active_slam_env():
    assert issubclass(MarineGymActiveSlamEnv, ActiveSlamEnv)


def test_marinegym_env_overrides_only_the_documented_methods():
    overridden = {
        name for name in vars(MarineGymActiveSlamEnv)
        if callable(getattr(MarineGymActiveSlamEnv, name, None)) and not name.startswith("__")
    }
    # The adapter is only supposed to override these (plus its own new
    # helpers) -- if a future edit accidentally overrides something else
    # (e.g. step() itself), that would break the "everything else is
    # inherited unmodified" guarantee this design depends on.
    disallowed_overrides = {"step", "_build_observation", "_crop_patch",
                             "_update_map_from_beams", "_integrate_odometry"}
    assert overridden.isdisjoint(disallowed_overrides)


def test_task_config_defaults_are_sane():
    cfg = MarineGymTaskConfig()
    assert cfg.num_envs == 1
    assert cfg.drone_model in {"BlueROV", "BlueROVHeavy", "iAUV", "LAUV", "HAUV"}


def test_world_and_sonar_adapters_expose_the_required_duck_typed_interface():
    # ActiveSlamEnv.step() only ever calls these methods on self.world /
    # self.sonar -- if the adapters stop exposing any of them, the swap
    # silently breaks at runtime instead of at import time.
    assert hasattr(MarineGymWorldAdapter, "is_free")
    assert "occ" in inspect.getsource(MarineGymWorldAdapter.__init__)
    assert hasattr(MarineGymSonarAdapter, "sense_imaging")
    assert hasattr(MarineGymSonarAdapter, "sense_scanning_360")
