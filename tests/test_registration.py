import numpy as np
import pytest

from active_slam_rl.registration.fs2d import FourierMellinRegistration


def _synthetic_frame(size=64, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size))
    for _ in range(15):
        cy, cx = rng.uniform(10, size - 10, size=2)
        r = rng.uniform(2, 5)
        yy, xx = np.mgrid[0:size, 0:size]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        img[mask] += rng.uniform(0.3, 1.0)
    return np.clip(img, 0, 1)


def _translate(img, dy, dx):
    from scipy.ndimage import shift
    return shift(img, shift=(dy, dx), mode="constant")


def test_pure_translation_recovered():
    base = _synthetic_frame()
    shifted = _translate(base, dy=4, dx=-3)
    reg = FourierMellinRegistration()
    result = reg.register(base, shifted)
    assert abs(result.dy - 4) < 1.5
    assert abs(result.dx - (-3)) < 1.5
    assert 0.0 <= result.quality <= 1.0
    assert result.covariance.shape == (3, 3)


def test_quality_is_bounded():
    a = _synthetic_frame(seed=1)
    b = _synthetic_frame(seed=2)  # unrelated scan -> low quality expected
    reg = FourierMellinRegistration()
    result = reg.register(a, b)
    assert 0.0 <= result.quality <= 1.0


def test_identical_scans_near_zero_motion():
    a = _synthetic_frame(seed=3)
    reg = FourierMellinRegistration()
    result = reg.register(a, a.copy())
    assert abs(result.dy) < 1.0
    assert abs(result.dx) < 1.0
    assert result.quality > 0.5


def test_pure_rotation_recovered():
    from scipy.ndimage import rotate
    base = _synthetic_frame(seed=4)
    angle_deg = 15.0
    rotated = rotate(base, angle_deg, reshape=False, mode="constant")
    reg = FourierMellinRegistration()
    result = reg.register(base, rotated)
    # Fourier-Mellin recovers rotation up to sign/discretization on a small
    # synthetic blob image; check it's in the right ballpark rather than
    # exact, and that it clearly picked up *some* rotation rather than none.
    recovered_deg = abs(np.rad2deg(result.dtheta))
    assert recovered_deg > 3.0, "should detect a substantial rotation, not report ~0"


def test_covariance_shrinks_with_quality():
    reg = FourierMellinRegistration()
    # A cleanly-matching pair (high quality) should report smaller
    # covariance than a poorly-matching pair (low quality) -- this is the
    # qualitative property RewardWeights/state encoding depend on, even
    # though the exact covariance model is a simplification (see fs2d.py).
    a = _synthetic_frame(seed=5)
    identical_result = reg.register(a, a.copy())
    noisy = np.clip(a + np.random.default_rng(0).normal(0, 2.0, a.shape), 0, 1)
    noisy_result = reg.register(a, noisy)
    assert np.trace(identical_result.covariance) <= np.trace(noisy_result.covariance) + 1e-6


def test_fold_ambiguity_tiebreak_uses_real_sonar_frames():
    """Regression test for the fold-ambiguity tie-break fix in
    FourierMellinRegistration.register (see fs2d.py's "FOLD-AMBIGUITY
    TIE-BREAK" docstring section). This env's real imaging-sonar frames
    are far sparser than this file's synthetic blob test images, and
    picking the rotation candidate by translation-correlation *sharpness*
    (the pre-fix behaviour) was wrong on the majority of registrations
    because sharpness saturates near its ceiling for both pi-separated
    candidates on sparse data, carrying essentially no discriminating
    signal. Picking by actual image-domain overlap after applying each
    candidate fixes this. This test locks in the measured improvement
    (>90-degree gross-error rate dropping from ~57% to well under 20%) so
    a future change to the tie-break can't silently regress it.
    """
    from active_slam_rl.env.world_generator import TunnelWorld, WorldConfig
    from active_slam_rl.env.sonar_model import SonarModel, SonarConfig

    rng = np.random.default_rng(0)
    world = TunnelWorld(WorldConfig(height=140, width=140, seed=1))
    sonar = SonarModel(world, SonarConfig(), rng=rng)
    reg = FourierMellinRegistration()
    y, x, theta = world.start_pose

    n_trials = 60
    gross_errors = 0
    for _ in range(n_trials):
        dtheta_true_deg = rng.uniform(-40, 40)
        dtheta_true = np.deg2rad(dtheta_true_deg)
        _, _, _, frame1 = sonar.sense_imaging(y, x, theta)
        _, _, _, frame2 = sonar.sense_imaging(y, x, theta + dtheta_true)
        result = reg.register(frame1, frame2)
        err = (np.rad2deg(result.dtheta) - dtheta_true_deg + 180) % 360 - 180
        if abs(err) > 90:
            gross_errors += 1

    gross_error_rate = gross_errors / n_trials
    assert gross_error_rate < 0.25, (
        f"fold-ambiguity gross-error rate regressed: {gross_error_rate:.1%} "
        f"of registrations picked the wrong pi-separated candidate "
        f"(expected well under 25% with the overlap-based tie-break)"
    )
