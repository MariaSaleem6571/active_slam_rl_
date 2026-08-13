"""
Tests for the SfM fusion module (fusion/sfm.py) and its IMU/DVL sensing
front-end (env/imu_dvl_model.py). Mirrors test_registration.py's style:
mostly synthetic, hand-constructed scenarios with a known ground truth,
checked against a numeric tolerance -- plus one integration-level test
against the full `ActiveSlamEnv` confirming the two pieces are actually
wired together correctly (env/sim_env.py) and that existing behavior is
preserved when the feature is switched off.
"""

import numpy as np
import pytest

from active_slam_rl.fusion.sfm import StateFusionModule, SfMConfig, FusionResult
from active_slam_rl.env.imu_dvl_model import (
    IMUModel, IMUConfig, DVLModel, DVLConfig, true_relative_motion,
)
from active_slam_rl.env.sim_env import ActiveSlamEnv, EnvConfig
from active_slam_rl.env.world_generator import WorldConfig


# ---------------------------------------------------------------------- #
# true_relative_motion (env/imu_dvl_model.py)
# ---------------------------------------------------------------------- #

def test_true_relative_motion_pure_forward_translation():
    # Heading 0 (facing +x): moving 2.0 forward is a pure +x displacement,
    # which should show up entirely as "forward" (body dx), zero lateral.
    dtheta, body_disp = true_relative_motion((0.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    assert dtheta == pytest.approx(0.0)
    assert body_disp[0] == pytest.approx(2.0)   # forward
    assert body_disp[1] == pytest.approx(0.0)   # lateral


def test_true_relative_motion_pure_rotation():
    dtheta, body_disp = true_relative_motion((0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 4))
    assert dtheta == pytest.approx(np.pi / 4)
    assert body_disp[0] == pytest.approx(0.0)
    assert body_disp[1] == pytest.approx(0.0)


def test_true_relative_motion_wraps_across_pi_boundary():
    # Heading goes from +170 deg to -170 deg -- a genuine +20 deg turn,
    # not a -340 deg one.
    dtheta, _ = true_relative_motion(
        (0.0, 0.0, np.deg2rad(170)), (0.0, 0.0, np.deg2rad(-170)))
    assert dtheta == pytest.approx(np.deg2rad(20), abs=1e-6)


def test_true_relative_motion_sideways_translation_is_lateral():
    # Heading 90 deg (facing +y): moving purely in +x is now "lateral" in
    # the body frame, not "forward".
    dtheta, body_disp = true_relative_motion(
        (0.0, 0.0, np.pi / 2), (0.0, 1.0, np.pi / 2))
    assert body_disp[0] == pytest.approx(0.0, abs=1e-9)     # forward
    assert body_disp[1] == pytest.approx(-1.0, abs=1e-9)    # lateral


# ---------------------------------------------------------------------- #
# IMUModel / DVLModel (env/imu_dvl_model.py)
# ---------------------------------------------------------------------- #

def test_imu_model_reports_true_bias_when_noiseless():
    cfg = IMUConfig(gyro_noise_std_deg=0.0, gyro_bias_walk_std_deg=0.0)
    rng = np.random.default_rng(0)
    imu = IMUModel(cfg, rng=rng)
    imu.true_bias = np.deg2rad(5.0)   # force a known bias directly
    reading = imu.sense(true_dtheta=np.deg2rad(10.0))
    assert np.rad2deg(reading) == pytest.approx(15.0, abs=1e-6)


def test_imu_model_reset_samples_a_fresh_bias():
    cfg = IMUConfig(gyro_bias_init_std_deg=5.0)
    rng = np.random.default_rng(1)
    imu = IMUModel(cfg, rng=rng)
    biases = []
    for _ in range(20):
        imu.reset()
        biases.append(imu.true_bias)
    assert np.std(biases) > 0.0   # actually varies across episodes


def test_dvl_model_dropout_more_likely_near_walls():
    cfg = DVLConfig(dropout_prob_base=0.0, dropout_prob_near_wall=1.0, near_wall_range=1.5)
    rng = np.random.default_rng(2)
    dvl = DVLModel(cfg, rng=rng)
    # Far from any wall: dropout_prob_base=0 -> never drops out.
    far_readings = [dvl.sense(np.array([1.0, 0.0]), min_obstacle_range=10.0) for _ in range(20)]
    assert all(r is not None for r in far_readings)
    # Very close to a wall: dropout_prob_near_wall=1 -> always drops out.
    near_readings = [dvl.sense(np.array([1.0, 0.0]), min_obstacle_range=0.1) for _ in range(20)]
    assert all(r is None for r in near_readings)


# ---------------------------------------------------------------------- #
# StateFusionModule (fusion/sfm.py)
# ---------------------------------------------------------------------- #

def test_fusion_falls_back_to_dead_reckoning_with_no_fs2d():
    """First step of an episode: no previous frame, so reg_delta is None.
    The fused output should be exactly the IMU/DVL prediction."""
    sfm = StateFusionModule()
    result = sfm.step(imu_dtheta=np.deg2rad(3.0), dvl_disp=np.array([0.5, 0.1]),
                       reg_delta=None, reg_cov=None)
    assert result.used_fs2d is False
    assert result.used_dvl is True
    assert result.fs2d_rejected_outlier is False
    assert result.dx == pytest.approx(0.5)
    assert result.dy == pytest.approx(0.1)
    assert np.rad2deg(result.dtheta) == pytest.approx(3.0)


def test_fusion_falls_back_to_fs2d_plus_imu_on_dvl_dropout():
    """A DVL dropout (None) should be handled by inflating that channel's
    uncertainty, not by pretending zero velocity is a real reading -- so
    with a very confident FS2D measurement, the fused translation should
    end up close to what FS2D reported, not close to zero."""
    sfm = StateFusionModule()
    reg_delta = np.array([0.9, 0.15, np.deg2rad(2.0)])
    reg_cov = np.diag([0.02 ** 2, 0.02 ** 2, np.deg2rad(0.5) ** 2])
    result = sfm.step(imu_dtheta=np.deg2rad(2.1), dvl_disp=None,
                       reg_delta=reg_delta, reg_cov=reg_cov)
    assert result.used_dvl is False
    assert result.used_fs2d is True
    assert result.dx == pytest.approx(0.9, abs=0.05)
    assert result.dy == pytest.approx(0.15, abs=0.05)


def test_fusion_reduces_uncertainty_below_either_source_alone():
    """The whole point of fusing two independent estimates: the fused
    covariance should be no worse than either source's own covariance."""
    sfm = StateFusionModule()
    reg_delta = np.array([1.0, 0.2, np.deg2rad(5.0)])
    reg_cov = np.diag([0.3 ** 2, 0.3 ** 2, np.deg2rad(3.0) ** 2])
    result = sfm.step(imu_dtheta=np.deg2rad(4.5), dvl_disp=np.array([0.95, 0.25]),
                       reg_delta=reg_delta, reg_cov=reg_cov)
    fused_theta_var = result.covariance[2, 2]
    pred_only_theta_var = sfm.bias_variance + np.deg2rad(sfm.cfg.assumed_gyro_noise_std_deg) ** 2
    assert fused_theta_var < reg_cov[2, 2]
    assert fused_theta_var < pred_only_theta_var
    assert result.fs2d_rejected_outlier is False


def test_bias_tracking_converges_toward_true_bias():
    """Feed the filter many steps where the (synthetic) IMU reading has a
    known constant bias and FS2D gives an independent, unbiased estimate
    of the same true rotation -- the tracked bias estimate should move
    substantially toward the true value and its uncertainty should shrink
    well below the filter's starting prior."""
    rng = np.random.default_rng(0)
    true_bias = np.deg2rad(3.0)
    sfm = StateFusionModule(SfMConfig(bias_init_std_deg=1.5))
    initial_std = sfm.bias_std_deg if hasattr(sfm, "bias_std_deg") else np.rad2deg(
        np.sqrt(sfm.bias_variance))

    result = None
    for _ in range(300):
        true_dtheta = rng.normal(0, np.deg2rad(5))
        imu_dtheta = true_dtheta + true_bias + rng.normal(0, np.deg2rad(1.0))
        reg_dtheta = true_dtheta + rng.normal(0, np.deg2rad(2.0))
        reg_delta = np.array([rng.normal(0, 0.05), rng.normal(0, 0.05), reg_dtheta])
        reg_cov = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(2.0) ** 2])
        dvl = np.array([rng.normal(0, 0.04), rng.normal(0, 0.04)])
        result = sfm.step(imu_dtheta=imu_dtheta, dvl_disp=dvl, reg_delta=reg_delta, reg_cov=reg_cov)

    assert abs(result.bias_estimate_deg - 3.0) < 1.0   # converged close to the true 3.0 deg
    assert result.bias_std_deg < initial_std             # uncertainty genuinely shrank


def test_bias_tracking_handles_the_plus_minus_180_wraparound():
    """A predicted dtheta of ~+179 deg and an FS2D reading of ~-179 deg
    disagree by only ~2 deg once wrapped -- not by ~358 deg. This should
    fuse normally (not get gated as an outlier) and should not corrupt
    the bias estimate."""
    sfm = StateFusionModule()
    reg_delta = np.array([0.0, 0.0, np.deg2rad(-179.0)])
    reg_cov = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(2.0) ** 2])
    result = sfm.step(imu_dtheta=np.deg2rad(179.0), dvl_disp=np.array([0.0, 0.0]),
                       reg_delta=reg_delta, reg_cov=reg_cov)
    assert result.fs2d_rejected_outlier is False
    assert result.used_fs2d is True
    # Fused heading should land near +-180, not somewhere wildly different.
    fused_deg = np.rad2deg(result.dtheta)
    assert min(abs(fused_deg - 180), abs(fused_deg + 180)) < 5.0


def test_nis_gate_rejects_a_grossly_inconsistent_fs2d_reading():
    """The concrete failure mode this gate exists for (see fusion/sfm.py's
    module docstring): FS2D confidently reports a rotation about 180 deg
    off from what IMU/DVL predict (e.g. a Fourier-Mellin fold-ambiguity
    misfire) -- this must be rejected, not fused in, and must not corrupt
    the persisted bias estimate."""
    sfm = StateFusionModule()
    bias_before = sfm.bias_estimate
    # IMU/DVL predict ~0 deg, 1 m forward; FS2D confidently claims ~180 deg
    # rotation with a *tight* covariance (the dangerous case: confidently wrong).
    reg_delta = np.array([1.0, 0.0, np.deg2rad(179.0)])
    reg_cov = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(1.0) ** 2])
    result = sfm.step(imu_dtheta=np.deg2rad(0.5), dvl_disp=np.array([1.0, 0.0]),
                       reg_delta=reg_delta, reg_cov=reg_cov)
    assert result.fs2d_rejected_outlier is True
    assert result.used_fs2d is False
    # Bias belief must be untouched by a rejected reading.
    assert sfm.bias_estimate == pytest.approx(bias_before)
    # Fused output should be the plain IMU/DVL dead-reckoning prediction.
    assert np.rad2deg(result.dtheta) == pytest.approx(0.5, abs=1e-6)
    assert result.dx == pytest.approx(1.0, abs=1e-6)


def test_nis_gate_accepts_a_moderately_noisy_but_honest_reading():
    """A reading that's noisier than the prediction but well within its
    *stated* covariance should NOT be gated out -- the gate is for
    egregious inconsistency, not ordinary disagreement."""
    sfm = StateFusionModule()
    reg_delta = np.array([1.05, -0.05, np.deg2rad(3.0)])
    reg_cov = np.diag([0.3 ** 2, 0.3 ** 2, np.deg2rad(4.0) ** 2])
    result = sfm.step(imu_dtheta=np.deg2rad(2.0), dvl_disp=np.array([1.0, 0.0]),
                       reg_delta=reg_delta, reg_cov=reg_cov)
    assert result.fs2d_rejected_outlier is False
    assert result.used_fs2d is True


def test_reset_clears_bias_belief_back_to_the_prior():
    # A modest, gate-passing 2 deg discrepancy (not the kind of ~180 deg
    # outlier the NIS gate above is meant to catch) -- chosen so this test
    # is actually exercising _update_bias, not the gate.
    cfg = SfMConfig(bias_init_std_deg=2.0)
    sfm = StateFusionModule(cfg)
    result = sfm.step(imu_dtheta=np.deg2rad(1.0), dvl_disp=np.array([0.0, 0.0]),
                       reg_delta=np.array([0.0, 0.0, np.deg2rad(-1.0)]),
                       reg_cov=np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(1.0) ** 2]))
    assert result.fs2d_rejected_outlier is False   # sanity: this test needs the update to fire
    assert sfm.bias_estimate != 0.0   # moved away from the prior
    sfm.reset()
    assert sfm.bias_estimate == 0.0
    assert np.rad2deg(np.sqrt(sfm.bias_variance)) == pytest.approx(2.0)


# ---------------------------------------------------------------------- #
# Integration: ActiveSlamEnv wiring (env/sim_env.py)
# ---------------------------------------------------------------------- #

def _small_env(**overrides):
    world = WorldConfig(height=40, width=40, seed=3)
    cfg = EnvConfig(world=world, max_steps=40, **overrides)
    return ActiveSlamEnv(cfg)


def test_env_runs_end_to_end_with_fusion_enabled():
    env = _small_env(use_sfm_fusion=True)
    obs, info = env.reset(seed=5)
    assert info["sfm"] is None   # nothing fused yet before the first step
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
    assert info["sfm"] is not None
    assert "bias_estimate_deg" in info["sfm"]


def test_env_disabled_fusion_matches_pre_sfm_behavior_exactly():
    """With use_sfm_fusion=False, IMU/DVL models must not even be
    constructed, and two runs with the same seed/actions must be
    identical -- i.e. this really is a no-op path, not fusion-with-tiny-
    effect."""
    env = _small_env(use_sfm_fusion=False)
    obs, info = env.reset(seed=5)
    assert env.imu_model is None
    assert env.dvl_model is None

    env2 = _small_env(use_sfm_fusion=False)
    env2.reset(seed=5)

    for i in range(15):
        a = i % 7
        obs1, r1, t1, tr1, info1 = env.step(a)
        obs2, r2, t2, tr2, info2 = env2.step(a)
        assert r1 == pytest.approx(r2)
        assert info1["est_pose"] == info2["est_pose"]
        assert info1["sfm"] is None and info2["sfm"] is None


def test_env_use_sfm_fusion_toggle_changes_est_pose_trajectory():
    """Sanity check that the toggle actually does something -- fusion on
    vs. off should not coincidentally produce the same trajectory."""
    env_on = _small_env(use_sfm_fusion=True)
    env_on.reset(seed=5)
    env_off = _small_env(use_sfm_fusion=False)
    env_off.reset(seed=5)

    for i in range(15):
        a = i % 7
        env_on.step(a)
        env_off.step(a)

    assert env_on.est_pose != env_off.est_pose
