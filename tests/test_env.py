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
    assert obs["scalars"].shape == (6,)
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
