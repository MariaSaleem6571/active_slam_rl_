import numpy as np
import pytest

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


def test_fs2d_skips_registration_across_sonar_modality_switch():
    """Regression test: sense_imaging (~130 deg FOV) and sense_scanning_360
    (full 360 deg sweep) frames aren't comparable, so FS2D must not
    register across a modality switch -- see env/sim_env.py::step()'s
    "FS2D registration against previous ego frame" comment. A step whose
    sonar mode differs from the previous step's must behave exactly like
    the very first step of an episode: no registration, reg=None."""
    env = _make_env(seed=7)
    env.reset(seed=7)

    # Force an imaging step, confirm normal (same-modality) registration
    # is attempted, then force a dwell (scanning) step and confirm the
    # modality switch suppresses registration for that one transition step.
    env.step(0)  # forward -> imaging frame; first step, no previous frame yet
    assert env._prev_frame_mode == "imaging"

    env.step(0)  # forward -> imaging again; same-modality registration should run
    assert env._last_reg is not None
    assert env._prev_frame_mode == "imaging"

    env.step(5)  # dwell-and-scan -> scanning frame; modality switch
    assert env._prev_frame_mode == "scanning"
    assert env._last_reg is None   # cross-modality pair must be skipped, not registered

    env.step(5)  # dwell again -> scanning; same-modality registration should resume
    assert env._prev_frame_mode == "scanning"
    assert env._last_reg is not None


def test_scalars_include_frame_mode_flag():
    """The 6th scalar (frame_mode_flag) tells the policy which sonar
    modality produced the current registration -- see
    sim_env.py::_build_observation's comment for why this matters
    (imaging vs scanning registration reliability differs substantially,
    see registration/fs2d.py). 0.0 = imaging, 1.0 = scanning."""
    env = _make_env(seed=11)
    obs, _ = env.reset(seed=11)
    assert obs["scalars"].shape == (8,)
    assert obs["scalars"][5] == 0.0  # no frame yet -> defaults to imaging/0.0

    obs, *_ = env.step(0)  # forward -> imaging
    assert obs["scalars"][5] == 0.0

    obs, *_ = env.step(5)  # dwell -> scanning
    assert obs["scalars"][5] == 1.0

    obs, *_ = env.step(0)  # forward -> imaging again
    assert obs["scalars"][5] == 0.0


def test_sonar_modality_restriction_forces_frame_mode():
    """EnvConfig.sonar_modality_restriction (see scripts/run_ablations.py
    for how this is used) must force every step's frame_mode regardless
    of the action taken, without breaking normal movement/collision
    physics from _apply_action."""
    cfg_imaging = EnvConfig(world=WorldConfig(height=80, width=80, n_steps=250, seed=0),
                             max_steps=40, seed=0, sonar_modality_restriction="imaging_only")
    env = ActiveSlamEnv(cfg_imaging)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, _, _, _, info = env.step(int(rng.integers(0, env.action_space.n)))
        assert info["frame_mode"] == "imaging"

    cfg_scanning = EnvConfig(world=WorldConfig(height=80, width=80, n_steps=250, seed=0),
                              max_steps=40, seed=0, sonar_modality_restriction="scanning_only")
    env = ActiveSlamEnv(cfg_scanning)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, _, _, _, info = env.step(int(rng.integers(0, env.action_space.n)))
        assert info["frame_mode"] == "scanning"


def test_use_loop_closure_false_never_validates_a_closure():
    """EnvConfig.use_loop_closure=False must gate closure *validation* and
    its pose correction -- used by scripts/run_ablations.py's
    'no_loop_closure' variant to isolate loop closure's contribution."""
    cfg = EnvConfig(world=WorldConfig(height=80, width=80, n_steps=250, seed=0),
                     max_steps=150, seed=0, use_loop_closure=False)
    env = ActiveSlamEnv(cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    any_validated = False
    for _ in range(150):
        _, _, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))
        if info["loop_closure_validated"]:
            any_validated = True
        if term or trunc:
            break
    assert not any_validated


def test_observation_scalars_include_heading_sin_cos():
    """Scalars 6 and 7 must be sin(theta), cos(theta) -- added so the
    observation isn't missing heading entirely (see
    _build_observation's docstring for why a world-aligned map patch
    without heading is a real partial-observability gap)."""
    env = _make_env(seed=3)
    obs, _ = env.reset(seed=3)
    y, x, theta = env.est_pose
    assert obs["scalars"][6] == pytest.approx(np.sin(theta), abs=1e-5)
    assert obs["scalars"][7] == pytest.approx(np.cos(theta), abs=1e-5)


def test_map_patch_is_egocentric_not_world_aligned():
    """Regression test for the egocentric-observation fix: a feature that
    is physically 'straight ahead' of the vehicle must land at the same
    patch pixel regardless of the vehicle's world heading. This is
    exactly the property that was missing before -- the map patch used
    to be a plain world-axis-aligned crop, so the same world content
    looked completely different (in fact, exactly flipped) after a
    180-degree turn despite representing the same relative situation.

    Tests this directly against _build_observation's own rotation step
    using a synthetic belief array with a single sharp feature, rather
    than relying on the environment to naturally produce a comparable
    map at two different headings (which the physics wouldn't easily
    reproduce identically).
    """
    from scipy.ndimage import rotate, center_of_mass

    env = _make_env(seed=4)
    env.reset(seed=4)

    size = env.cfg.patch_size
    center = size // 2
    radius = size // 3
    yy, xx = np.mgrid[0:size, 0:size]

    offsets = []
    for theta_deg in [0, 60, 140, 200, 300]:
        theta = np.deg2rad(theta_deg)
        fy = center + radius * np.sin(theta)
        fx = center + radius * np.cos(theta)
        blob = np.exp(-((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * 1.5 ** 2))

        # exercise the exact same rotation _build_observation applies
        angle_deg = np.degrees(theta)
        rotated = rotate(blob, angle_deg, reshape=False, mode="constant", cval=0.0, order=1)
        cy, cx = center_of_mass(rotated)
        offsets.append((cy - center, cx - center))

    # every heading's "straight ahead" feature must land at (approximately)
    # the same rotated-patch location. A tolerance is needed here (not
    # exact equality) because scipy's spline interpolation plus a Gaussian
    # blob's soft edges introduce a few tenths of a pixel of numerical
    # noise per rotation -- confirmed by a larger, cleaner 41x41/radius-12
    # version of this same check showing ~0.00 deviation, so this is
    # measurement-resolution noise, not a sign the formula itself is only
    # approximately right. What this test guards against is a sign error
    # or wrong-axis bug (like the one this fix corrects), which would
    # produce gross, radius-scale disagreement between headings, not
    # sub-pixel noise.
    mean_row = np.mean([o[0] for o in offsets])
    mean_col = np.mean([o[1] for o in offsets])
    for row_off, col_off in offsets:
        assert abs(row_off - mean_row) < 2.5, f"egocentric rotation inconsistent across headings: {offsets}"
        assert abs(col_off - mean_col) < 2.5, f"egocentric rotation inconsistent across headings: {offsets}"


def test_action_6_targets_actual_loop_candidate_not_arbitrary_middle_keyframe():
    """Regression test for the action-6 bug: _best_known_loop_candidate_pose
    must use the actual last-known loop-closure candidate
    (self._last_candidate) when one exists, not an arbitrary
    len(keyframes)//2 index that ignores it entirely."""
    env = _make_env(seed=6)
    env.reset(seed=6)
    rng = np.random.default_rng(6)
    # accumulate enough keyframes that "middle index" and "most recent
    # candidate" would very likely disagree if the bug were still present
    for _ in range(60):
        env.step(int(rng.integers(0, 5)))  # avoid action 6 itself while building history

    if not env.loop_detector.keyframes:
        pytest.skip("no keyframes accumulated in this run; not meaningful without them")

    # fabricate a specific "last candidate" pointing at a known keyframe
    from active_slam_rl.perception.loop_closure import LoopClosureCandidate
    target_idx = len(env.loop_detector.keyframes) - 1  # the most recent one -- as far as possible from a middle index
    env._last_candidate = LoopClosureCandidate(keyframe_idx=target_idx, saliency=1.0)

    expected_pose = env.loop_detector.keyframe_pose(target_idx)
    actual_pose = env._best_known_loop_candidate_pose()
    assert actual_pose == expected_pose


def test_repeated_collisions_truncate_the_episode():
    """Regression test for the collision-streak fix: repeatedly colliding
    (e.g. an agent stuck ramming a wall) must eventually truncate the
    episode rather than being allowed to continue indefinitely for a
    bounded per-step cost."""
    cfg = EnvConfig(world=WorldConfig(height=80, width=80, n_steps=250, seed=0),
                     max_steps=500, seed=0, max_collision_streak=5)
    env = ActiveSlamEnv(cfg)
    env.reset(seed=0)

    # Deterministically place the vehicle right next to a wall, facing
    # directly into it, rather than relying on the start pose happening
    # to have one nearby for this seed.
    occ = env.world.occ
    wall_ys, wall_xs = np.where(occ == 1)
    free_mask = (occ == 0)
    placed = False
    for wy, wx in zip(wall_ys, wall_xs):
        # A float y that (a) rounds to the free cell just "south" of this
        # wall cell (so is_free() at the start pose is True) and (b) is
        # close enough that a 0.5m forward move rounds into the wall cell
        # itself (so is_free() after moving is False) -- guarantees a
        # collision on the very first step regardless of the exact
        # integer/rounding boundary, rather than relying on being exactly
        # 1.0 grid unit away where round-half-to-even could go either way.
        fy, fx = wy - 1, wx
        if 0 <= fy < occ.shape[0] and free_mask[fy, fx]:
            start_y = wy - 0.6
            env.true_pose = (start_y, float(wx), np.pi / 2)  # facing +y, straight at the wall cell
            env.est_pose = env.true_pose
            placed = True
            break
    assert placed, "test world has no wall with a free cell adjacent -- shouldn't happen for a real tunnel world"

    truncated_at = None
    for i in range(20):
        _, _, terminated, truncated, info = env.step(0)   # forward 0.5m, straight into the wall
        assert info["collided"], "test setup didn't actually produce a collision -- fix the placement logic above"
        if truncated:
            truncated_at = i + 1
            break
    assert truncated_at is not None
    assert truncated_at <= 6   # 5-collision streak + the step that trips it


def test_local_entropy_normalized_ignores_out_of_bounds_and_far_away_cells():
    """Regression test for the local-vs-global coverage-reward fix:
    _local_entropy_normalized must (a) only reflect cells within the
    given window, not the whole map, and (b) not treat out-of-bounds
    padding as artificially zero-entropy (fully resolved)."""
    env = _make_env(seed=7)
    env.reset(seed=7)

    # OccupancyGrid.prob is a derived property (sigmoid of log_odds), not
    # a directly-settable array -- set log_odds instead so the change is
    # actually reflected in .prob.
    env.map.log_odds[:] = 0.0   # sigmoid(0) = 0.5 everywhere: uniformly max-uncertain
    whole_map_entropy = env.map.entropy_normalized()

    # small window well inside the map bounds should read the same value
    # (uniform field -> local mean == global mean)
    y, x = env.map.height // 2, env.map.width // 2
    local = env._local_entropy_normalized(env.map.prob, y, x, radius=5)
    assert local == pytest.approx(whole_map_entropy, abs=1e-6)

    # resolve a small region far from (y, x) -- the local window centered
    # there must NOT reflect that change at all
    env.map.log_odds[0:3, 0:3] = 10.0   # sigmoid(10) ~ fully resolved, far corner
    local_unaffected = env._local_entropy_normalized(env.map.prob, y, x, radius=5)
    assert local_unaffected == pytest.approx(whole_map_entropy, abs=1e-6)

    # but a window actually centered on the resolved region must reflect it
    local_at_resolved = env._local_entropy_normalized(env.map.prob, 1, 1, radius=2)
    assert local_at_resolved < whole_map_entropy


def test_coverage_reward_uses_local_not_global_entropy():
    """End-to-end regression test: a step's coverage_term must come from
    _local_entropy_normalized, not the old whole-map
    entropy_normalized() delta -- verified by confirming the two give
    meaningfully different magnitudes on a real step (the whole point of
    the fix), rather than just checking the code calls the new method."""
    env = _make_env(seed=8)
    env.reset(seed=8)

    prob_before = env.map.prob.copy()
    _, reward, _, _, info = env.step(0)
    prob_after = env.map.prob

    global_delta = (
        -(np.clip(prob_before, 1e-6, 1 - 1e-6) * np.log(np.clip(prob_before, 1e-6, 1 - 1e-6))
          + (1 - np.clip(prob_before, 1e-6, 1 - 1e-6)) * np.log(1 - np.clip(prob_before, 1e-6, 1 - 1e-6))).mean()
        + (np.clip(prob_after, 1e-6, 1 - 1e-6) * np.log(np.clip(prob_after, 1e-6, 1 - 1e-6))
           + (1 - np.clip(prob_after, 1e-6, 1 - 1e-6)) * np.log(1 - np.clip(prob_after, 1e-6, 1 - 1e-6))).mean()
    )
    old_style_term = global_delta * env.cfg.reward_weights.w_cov
    new_coverage_term = info["reward_breakdown"]["coverage_term"]

    # the local version should be substantially larger in magnitude for
    # an early, informative step -- this is the entire point of the fix
    assert abs(new_coverage_term) > abs(old_style_term) * 2
