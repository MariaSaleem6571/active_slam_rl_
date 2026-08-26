"""Tests for perception/sfm2d.py -- see that module's docstring for why
this is the actual "SfM" (as opposed to fusion/sfm.py's StateFusionModule,
an EKF that predates this file and happens to share the "SfM" name)."""
import numpy as np
import pytest

from active_slam_rl.perception.sfm2d import StructureFromMotion2D, SfM2DConfig, _wrap


def _observe(pose, landmarks, rng, max_range=22.0, noise_r=0.02, noise_b_deg=0.2, n_pad=5):
    """Simulate this modality's beam returns to a set of ground-truth
    landmarks from `pose`, in the same (ranges, angles) convention
    SonarModel.sense_imaging/sense_scanning_360 return: angles relative
    to heading, out-of-range beams read exactly `max_range`."""
    Y, X, Theta = pose
    obs_r, obs_a = [], []
    for ly, lx in landmarks:
        dy, dx = ly - Y, lx - X
        r = np.hypot(dy, dx)
        if r > max_range:
            continue
        bearing = np.arctan2(dy, dx) - Theta
        obs_r.append(r + rng.normal(0, noise_r))
        obs_a.append(bearing + np.deg2rad(rng.normal(0, noise_b_deg)))
    ranges = np.array(obs_r + [max_range] * n_pad)
    angles = np.array(obs_a + list(rng.uniform(-np.pi, np.pi, n_pad)))
    return ranges, angles


def test_wrap_keeps_angles_in_range():
    vals = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi, 0.1, -0.1])
    wrapped = _wrap(vals)
    assert np.all(wrapped >= -np.pi - 1e-9) and np.all(wrapped <= np.pi + 1e-9)


def test_no_valid_returns_yields_no_correction():
    sfm = StructureFromMotion2D("imaging")
    ranges = np.full(20, 22.0)   # every beam maxed out -> no landmark observations
    angles = np.linspace(-1.0, 1.0, 20)
    result = sfm.process_frame(ranges, angles, (0.0, 0.0, 0.0), timestep=0, max_range=22.0)
    assert result.pose_correction is None
    assert result.n_matched == 0
    assert len(sfm.get_map()) == 0


def test_landmarks_accumulate_from_repeated_observation():
    rng = np.random.default_rng(1)
    landmarks_true = rng.uniform(-10, 10, size=(15, 2))
    sfm = StructureFromMotion2D("imaging", SfM2DConfig(min_matched_for_correction=4))
    pose = (0.0, 0.0, 0.0)
    for t in range(3):
        ranges, angles = _observe(pose, landmarks_true, rng)
        sfm.process_frame(ranges, angles, pose, timestep=t, max_range=22.0)
    # every ground-truth landmark within range should have produced (roughly)
    # one persistent tracked landmark, not a fresh one every step
    assert 5 <= len(sfm.get_map()) <= len(landmarks_true)
    assert all(lm.n_obs >= 2 for lm in sfm.get_map())


def test_pose_correction_reduces_injected_error_over_iterations():
    """Regression test for the landmark-update contamination fix in
    process_frame (see its 'Pass 2' comment): landmark positions must be
    refined using the *corrected* pose, not the pre-correction estimate,
    or the map slowly absorbs the very bias it's supposed to correct and
    the pose estimate stops improving. This drives a fixed injected pose
    error through several correction steps and checks it shrinks well
    below its starting magnitude and doesn't diverge or plateau high.
    """
    rng = np.random.default_rng(0)
    landmarks_true = rng.uniform(-15, 15, size=(30, 2))
    sfm = StructureFromMotion2D("imaging", SfM2DConfig(
        range_noise_std=0.02, bearing_noise_std_deg=0.2, min_matched_for_correction=4))

    true_pose_warmup = (0.0, 0.0, 0.0)
    for _ in range(3):
        ranges, angles = _observe(true_pose_warmup, landmarks_true, rng)
        sfm.process_frame(ranges, angles, true_pose_warmup, timestep=0, max_range=22.0)

    true_pose = (2.0, 1.0, 0.15)
    injected_error = np.array([0.6, -0.4, 0.05])
    estimate = np.array(true_pose) + injected_error

    initial_err_norm = np.linalg.norm(injected_error)
    for it in range(8):
        ranges, angles = _observe(true_pose, landmarks_true, rng)
        result = sfm.process_frame(ranges, angles, tuple(estimate), timestep=it + 1, max_range=22.0)
        if result.pose_correction is not None:
            estimate = estimate + 0.7 * np.array(result.pose_correction)

    final_err = estimate - np.array(true_pose)
    final_err_norm = np.linalg.norm(final_err)
    assert final_err_norm < 0.3 * initial_err_norm, (
        f"pose correction didn't converge: started at {initial_err_norm:.3f}, "
        f"ended at {final_err_norm:.3f}"
    )


def test_imaging_and_scanning_maps_are_independent():
    rng = np.random.default_rng(2)
    landmarks_true = rng.uniform(-10, 10, size=(10, 2))
    sfm_imaging = StructureFromMotion2D("imaging")
    sfm_scanning = StructureFromMotion2D("scanning")
    pose = (0.0, 0.0, 0.0)
    ranges, angles = _observe(pose, landmarks_true, rng)
    sfm_imaging.process_frame(ranges, angles, pose, timestep=0, max_range=22.0)
    assert len(sfm_imaging.get_map()) > 0
    assert len(sfm_scanning.get_map()) == 0   # untouched -- fully independent instance
    for lm in sfm_imaging.get_map():
        assert lm.modality == "imaging"
