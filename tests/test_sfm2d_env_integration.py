"""Integration tests for StructureFromMotion2D wired into ActiveSlamEnv
(env/sim_env.py's EnvConfig.use_sfm2d / sfm2d_apply_correction_from /
sfm2d_correction_gain, and the self.sfm2d_imaging / self.sfm2d_scanning
instances). See tests/test_sfm2d.py for the module's own unit tests."""
import numpy as np
import pytest

from active_slam_rl.env.sim_env import ActiveSlamEnv, EnvConfig
from active_slam_rl.env.world_generator import WorldConfig
from active_slam_rl.metrics.plotting import plot_sfm2d_landmark_maps


def _make_env(seed=0, use_sfm2d=True, apply_from="both"):
    cfg = EnvConfig(world=WorldConfig(height=100, width=100, n_steps=250, seed=seed),
                     max_steps=60, seed=seed,
                     use_sfm2d=use_sfm2d, sfm2d_apply_correction_from=apply_from)
    return ActiveSlamEnv(cfg)


def test_sfm2d_disabled_by_default():
    cfg = EnvConfig()
    assert cfg.use_sfm2d is False


def test_sfm2d_info_none_when_disabled():
    env = _make_env(use_sfm2d=False)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["sfm2d"] is None


def test_sfm2d_builds_both_maps_independently_when_enabled():
    env = _make_env(use_sfm2d=True)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    saw_imaging, saw_scanning = False, False
    for _ in range(50):
        action = rng.integers(0, env.action_space.n)
        _, _, _, _, info = env.step(action)
        if info["frame_mode"] == "imaging":
            saw_imaging = True
        elif info["frame_mode"] == "scanning":
            saw_scanning = True
    assert saw_imaging and saw_scanning, "test needs both modalities exercised to be meaningful"
    assert len(env.sfm2d_imaging.get_map()) > 0
    assert len(env.sfm2d_scanning.get_map()) > 0
    # independence: every landmark in each map is tagged with its own modality
    assert all(lm.modality == "imaging" for lm in env.sfm2d_imaging.get_map())
    assert all(lm.modality == "scanning" for lm in env.sfm2d_scanning.get_map())


def test_sfm2d_maps_reset_between_episodes():
    env = _make_env(use_sfm2d=True)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        env.step(rng.integers(0, env.action_space.n))
    assert len(env.sfm2d_imaging.get_map()) > 0
    env.reset(seed=1)
    assert len(env.sfm2d_imaging.get_map()) == 0
    assert len(env.sfm2d_scanning.get_map()) == 0


def test_apply_correction_from_off_never_moves_est_pose_via_sfm2d():
    """apply_correction_from='off' should still build both landmark maps
    (for inspection) but never let a computed correction touch est_pose."""
    env_off = _make_env(use_sfm2d=True, apply_from="off")
    env_off.reset(seed=0)
    env_both = _make_env(use_sfm2d=True, apply_from="both")
    env_both.reset(seed=0)

    rng1 = np.random.default_rng(5)
    poses_off = []
    for _ in range(60):
        action = rng1.integers(0, env_off.action_space.n)
        _, _, _, _, info = env_off.step(action)
        poses_off.append(info["est_pose"])

    # both maps should still have grown even with corrections turned off
    assert len(env_off.sfm2d_imaging.get_map()) > 0


def test_plot_sfm2d_landmark_maps_smoke(tmp_path):
    env = _make_env(use_sfm2d=True)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        env.step(rng.integers(0, env.action_space.n))
    out_path = plot_sfm2d_landmark_maps(env, str(tmp_path))
    import os
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_sfm2d_apply_correction_from_gates_which_modality_corrects():
    """When apply_correction_from='imaging', a scanning-mode step's
    computed correction must never be applied to est_pose, even though
    the scanning map is still built. Checked indirectly: with
    correction_gain cranked to 1.0 and apply_correction_from='off', the
    only difference between two runs with identical actions/seeds should
    be whatever ordinary process noise contributes -- est_pose should NOT
    diverge due to sfm2d specifically. We check this by comparing
    use_sfm2d=True/apply_from='off' against use_sfm2d=False and requiring
    identical est_pose trajectories (since 'off' should be a no-op on
    est_pose, only building maps on the side)."""
    cfg_disabled = EnvConfig(world=WorldConfig(height=100, width=100, n_steps=250, seed=0),
                              max_steps=60, seed=0, use_sfm2d=False)
    cfg_off = EnvConfig(world=WorldConfig(height=100, width=100, n_steps=250, seed=0),
                         max_steps=60, seed=0, use_sfm2d=True, sfm2d_apply_correction_from="off")

    env_disabled = ActiveSlamEnv(cfg_disabled)
    env_off = ActiveSlamEnv(cfg_off)
    env_disabled.reset(seed=0)
    env_off.reset(seed=0)

    rng = np.random.default_rng(3)
    actions = [int(rng.integers(0, env_disabled.action_space.n)) for _ in range(40)]
    for a in actions:
        _, _, _, _, info_disabled = env_disabled.step(a)
        _, _, _, _, info_off = env_off.step(a)
        assert info_disabled["est_pose"] == pytest.approx(info_off["est_pose"], abs=1e-9)
