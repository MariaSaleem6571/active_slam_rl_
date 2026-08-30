"""
ActiveSlamEnv: the RL environment implementing the full architecture of
Figure 5 in the thesis proposal.

Per step, the forward pipeline runs exactly as described in section 5:

  raw sonar -> FS2D registration (q_t, Sigma_reg_t)
            -> volumetric map update (Bayesian, log-odds)
            -> change detection (eta_t^v, mask C_t)
            -> place recognition / loop closure (ell_t)
            -> state encoding (patch + scalars -> handed to PPO's encoder)
            -> reward (coverage, consistency, safety, loop bonus, change)

Two poses are tracked deliberately:

  * `true_pose`      -- ground truth, used only to step the physics/sonar
                         and to compute evaluation metrics (ATE/RPE). The
                         policy never observes this directly.
  * `est_pose`        -- built purely from FS2D odometry integrated over
                         time (with drift), corrected opportunistically by
                         loop closures. The *map* is built in this
                         estimated frame, so drift shows up exactly the way
                         it would in a real deployed system, and the RL
                         policy is rewarded for reducing it.

Action space (thesis section 5.6, depth dropped since the world is a 2D
plan-view -- see docstring in world_generator.py):
  0: forward 0.5 m   1: forward 1 m   2: forward 2 m
  3: yaw -30 deg     4: yaw +30 deg
  5: dwell-and-scan (full 360 sweep, no translation, cheaper drift)
  6: revisit best-known loop-closure candidate (steer toward it)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from active_slam_rl.env.world_generator import TunnelWorld, WorldConfig
from active_slam_rl.env.sonar_model import SonarModel, SonarConfig
from active_slam_rl.env.imu_dvl_model import IMUModel, IMUConfig, DVLModel, DVLConfig, true_relative_motion
from active_slam_rl.env.reward import compute_reward, compute_beta, AdaptiveDecayController, RewardWeights
from active_slam_rl.mapping.volumetric_map import OccupancyGrid
from active_slam_rl.mapping.change_detection import compute_change_mask, compute_innovation
from active_slam_rl.perception.loop_closure import LoopClosureDetector
from active_slam_rl.perception.sfm2d import StructureFromMotion2D, SfM2DConfig
from active_slam_rl.registration.fs2d import FS2DRegistration
from active_slam_rl.fusion.sfm import StateFusionModule, SfMConfig


@dataclass
class EnvConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    sonar: SonarConfig = field(default_factory=SonarConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    imu: IMUConfig = field(default_factory=IMUConfig)
    dvl: DVLConfig = field(default_factory=DVLConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    # NOTE: `sfm` above is fusion/sfm.py's StateFusionModule config (an EKF
    # -- "SfM" there is a pre-existing backronym, not Structure from
    # Motion; see perception/sfm2d.py's docstring). `sfm2d` below is the
    # actual Structure-from-Motion: a landmark map + pose-correction
    # estimator, kept as two fully independent instances (one per sonar
    # modality) -- see StructureFromMotion2D/SfM2DConfig docs.
    sfm2d: SfM2DConfig = field(default_factory=SfM2DConfig)
    patch_size: int = 32
    max_steps: int = 500
    battery_capacity: float = 500.0     # abstract energy units
    battery_cost_move: float = 1.0
    battery_cost_dwell: float = 2.5     # a full 360 sweep costs more
    process_noise_xy: float = 0.05
    process_noise_theta_deg: float = 1.0
    change_threshold: float = 1.5
    map_completeness_threshold: float = 0.6
    force_numpy_fs2d: bool = True       # native lib usually isn't linked yet
    # Fuse FS2D with IMU/DVL dead-reckoning via fusion.sfm.StateFusionModule
    # (see docs/ARCHITECTURE.md section 9) instead of integrating FS2D's
    # output alone. Set False to reproduce this codebase's pre-SfM behavior
    # exactly, byte-for-byte (including the exact RNG draw sequence) -- a
    # useful ablation switch, and an escape hatch if you need to compare
    # against results generated before this feature existed.
    use_sfm_fusion: bool = True
    # Structure-from-Motion (perception/sfm2d.py) is entirely separate
    # from the above and off by default, so existing configs/results are
    # unaffected unless explicitly opted in. `use_sfm2d=True` builds both
    # per-modality landmark maps (self.sfm2d_imaging / self.sfm2d_scanning)
    # regardless of `sfm2d_apply_correction_from`, purely so both maps are
    # available for inspection/plotting (see
    # metrics/plotting.py::plot_sfm2d_landmark_maps) even when you only
    # want one modality's corrections actually affecting est_pose.
    # `sfm2d_apply_correction_from` picks which modality's pose-correction
    # output is allowed to nudge est_pose: "imaging", "scanning", "both",
    # or "off" (build the maps, use neither for correction -- e.g. to
    # inspect map quality in isolation without it affecting the
    # trajectory the policy sees).
    use_sfm2d: bool = False
    sfm2d_apply_correction_from: str = "both"
    sfm2d_correction_gain: float = 0.5   # how strongly (0-1) to apply a computed correction, mirrors loop closure's `pull`
    seed: Optional[int] = None


class ActiveSlamEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: EnvConfig = EnvConfig(), render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config
        self.render_mode = render_mode
        self.rng = np.random.default_rng(config.seed)

        self.action_space = spaces.Discrete(7)
        ps = config.patch_size
        self.observation_space = spaces.Dict({
            "patch": spaces.Box(low=0.0, high=1.0, shape=(3, ps, ps), dtype=np.float32),
            "scalars": spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32),
        })

        self.fs2d = FS2DRegistration(force_numpy=config.force_numpy_fs2d)
        # Persistent across episodes on purpose -- see AdaptiveDecayController's
        # docstring: it accumulates a running statistic *across* resets, not per-episode.
        self._decay_controller = AdaptiveDecayController(
            initial_decay_rate=config.reward_weights.beta_decay_rate)
        # The SfM fusion EKF (fusion/sfm.py) draws no randomness of its
        # own, so -- unlike the two sensor models below -- it's safe to
        # create once here and just reset its state every episode (see
        # _reset_internal_state). IMUModel/DVLModel are deliberately *not*
        # created here: `self.rng` gets *reassigned* (not mutated in place)
        # by a seeded reset() call below, so binding a sensor model to
        # today's `self.rng` object in __init__ would leave it silently
        # drawing from a stale generator after the first reseed. They're
        # (re)built inside _reset_internal_state instead, exactly like
        # `self.sonar` already is, so they always bind the *current*
        # `self.rng`.
        self.sfm = StateFusionModule(config.sfm)
        # Neither draws any randomness of its own (pure geometry -- see
        # perception/sfm2d.py), so, like self.sfm above, safe to create
        # once here and just clear their landmark maps every episode (see
        # _reset_internal_state) rather than rebuild against self.rng.
        self.sfm2d_imaging = StructureFromMotion2D("imaging", config.sfm2d)
        self.sfm2d_scanning = StructureFromMotion2D("scanning", config.sfm2d)
        self._episode_count = 0
        self._reset_internal_state()

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._episode_count += 1
        self._reset_internal_state()
        obs = self._build_observation()
        info = self._build_info(reward_breakdown=None)
        return obs, info

    def step(self, action: int):
        self.t += 1
        prev_true = self.true_pose
        prev_est = self.est_pose

        collided, dwell = self._apply_action(action)

        # --- Sense ---
        y, x, theta = self.true_pose
        if dwell:
            ranges, angles, hits, frame = self.sonar.sense_scanning_360(y, x, theta)
            self.battery -= self.cfg.battery_cost_dwell
            frame_mode = "scanning"
        else:
            ranges, angles, hits, frame = self.sonar.sense_imaging(y, x, theta)
            self.battery -= self.cfg.battery_cost_move
            frame_mode = "imaging"

        # --- FS2D registration against previous ego frame -> odometry ---
        # Only register two frames captured by the *same* sonar modality.
        # sense_imaging (narrow ~130 deg FOV) and sense_scanning_360 (full
        # 360 deg sweep) produce frames with fundamentally different
        # coverage/beam-density, so a phase-correlation match across that
        # boundary isn't a meaningful motion estimate even though both
        # frames happen to share the same (frame_size, frame_size) shape
        # FS2D expects. On a modality switch we treat this step exactly
        # like the very first step of an episode (reg=None): the fusion
        # module already handles that by falling back to IMU/DVL dead
        # reckoning alone (see fusion/sfm.py), and _integrate_odometry
        # does the equivalent when fusion is off. We still update
        # _prev_frame/_prev_frame_mode below so *next* step can register
        # normally once two same-modality frames are available again.
        if self._prev_frame is not None and self._prev_frame_mode == frame_mode:
            reg = self.fs2d.register(self._prev_frame, frame)
        else:
            reg = None
        self._prev_frame = frame
        self._prev_frame_mode = frame_mode

        # --- IMU/DVL sensing + SfM fusion (feedback loop B / "Map -> SfM";
        # see docs/ARCHITECTURE.md section 9 and fusion/sfm.py) ---
        # Sensing only happens at all when fusion is switched on: EnvConfig
        # promises use_sfm_fusion=False reproduces this codebase's pre-SfM
        # behavior byte-for-byte, including the exact self.rng draw
        # sequence -- IMUModel/DVLModel.sense() both draw from self.rng, so
        # calling them unconditionally would silently break that promise
        # even while every *other* draw stayed in the same order.
        if self.cfg.use_sfm_fusion:
            # `true_relative_motion` reads this step's *true* motion purely
            # from true_pose -- it doesn't know or care which simulator
            # backend is in use, so this fusion step would work unmodified
            # against a different world/sensing backend too, exactly like
            # FS2D registration and everything downstream of it already does.
            true_dtheta, true_disp_body = true_relative_motion(prev_true, self.true_pose)
            imu_dtheta = self.imu_model.sense(true_dtheta)
            min_range_now = float(np.min(ranges)) if len(ranges) else self.cfg.sonar.max_range
            dvl_disp = self.dvl_model.sense(true_disp_body, min_obstacle_range=min_range_now)

            if reg is not None:
                reg_delta = np.array([reg.dx * self._scan_scale, reg.dy * self._scan_scale, reg.dtheta])
                reg_cov = reg.covariance.copy()
                reg_cov[:2, :2] *= self._scan_scale ** 2   # Var(k*X) = k^2 Var(X)
            else:
                reg_delta, reg_cov = None, None
            fusion = self.sfm.step(imu_dtheta=imu_dtheta, dvl_disp=dvl_disp,
                                    reg_delta=reg_delta, reg_cov=reg_cov)
            est_y, est_x, est_theta = self._integrate_odometry_from_delta(
                (fusion.dx, fusion.dy, fusion.dtheta))
        else:
            fusion = None
            est_y, est_x, est_theta = self._integrate_odometry(reg)
        self._last_fusion = fusion

        # --- Structure-from-Motion (perception/sfm2d.py) -- landmark map
        # + pose correction, entirely separate from the EKF fusion above
        # (see EnvConfig.sfm2d's comment / perception/sfm2d.py's docstring
        # for why "sfm" and "sfm2d" are two different things in this
        # codebase). Both per-modality maps are always built when
        # use_sfm2d=True regardless of sfm2d_apply_correction_from, purely
        # so both are available for inspection/plotting even when only one
        # modality's corrections are allowed to affect est_pose. Uses
        # est_pose (this step's already-fused estimate), never true_pose --
        # see StructureFromMotion2D.process_frame's docstring for why using
        # true_pose here would defeat the entire point.
        sfm2d_result = None
        if self.cfg.use_sfm2d:
            sfm2d_engine = self.sfm2d_imaging if frame_mode == "imaging" else self.sfm2d_scanning
            sfm2d_result = sfm2d_engine.process_frame(
                ranges, angles, (est_y, est_x, est_theta), self.t, max_range=self.cfg.sonar.max_range)
            allow_correction = self.cfg.sfm2d_apply_correction_from in (frame_mode, "both")
            if allow_correction and sfm2d_result.pose_correction is not None:
                gain = self.cfg.sfm2d_correction_gain
                dy_c, dx_c, dtheta_c = sfm2d_result.pose_correction
                est_y += gain * dy_c
                est_x += gain * dx_c
                est_theta += gain * dtheta_c
        self._last_sfm2d = sfm2d_result

        # --- Map update (Bayesian, in the estimated frame) ---
        self.map.snapshot_for_change_detection()
        self._update_map_from_beams(est_y, est_x, est_theta, ranges, angles)

        prob_curr = self.map.prob
        prob_prev = self.map._prev_prob
        variance = self.map.uncertainty()
        change_mask = compute_change_mask(prob_curr, prob_prev, variance,
                                           threshold=self.cfg.change_threshold)
        entropy_after = self.map.entropy_normalized()
        entropy_delta = self._prev_entropy - entropy_after   # positive = uncertainty resolved
        self._prev_entropy = entropy_after

        # --- Loop closure / place recognition ---
        # Captured *before* a closure this step can reset the clock below,
        # so the step that actually finds a closure still gets credited
        # with the high urgency that motivated finding it -- the reset
        # only takes effect starting next step.
        steps_since_last_closure = self.t - self._last_validated_closure_step
        local_unknown_fraction = self._local_unknown_fraction(est_y, est_x)
        beta_weights = replace(self.cfg.reward_weights, beta_decay_rate=self._decay_controller.decay_rate)
        beta = compute_beta(steps_since_last_closure, local_unknown_fraction, beta_weights)

        self.loop_detector.maybe_add_keyframe((est_y, est_x, est_theta), frame, self.t,
                                               mode=frame_mode)
        ell_t, candidate = self.loop_detector.query(frame, self.t, mode=frame_mode)
        loop_closure_validated = False
        info_gain = 0.0
        if candidate is not None:
            kf_scan = self.loop_detector.keyframe_scan(candidate.keyframe_idx)
            lc_reg = self.fs2d.register(kf_scan, frame)
            # Two extra gates beyond raw registration quality, both aimed at
            # the same failure mode: a *stationary* vehicle can otherwise
            # keep "closing a loop" against its own single old keyframe
            # forever, once it clears the recency-exclusion window, for a
            # free flat bonus with zero risk and zero new information. We
            # only credit a closure that (a) has actual accumulated drift
            # worth correcting, and (b) hasn't just been credited moments
            # ago against the same keyframe.
            enough_drift_to_matter = self.trace_cov > self._min_trace_cov_for_bonus
            cooldown_elapsed = (self.t - self._last_validated_closure_step) > self._loop_closure_cooldown
            if lc_reg.quality > 0.35 and enough_drift_to_matter and cooldown_elapsed:
                loop_closure_validated = True
                self._decay_controller.update(steps_since_last_closure)
                self._last_validated_closure_step = self.t
                kf_pose = self.loop_detector.keyframe_pose(candidate.keyframe_idx)
                # "Pose graph correction": pull the drifted estimate back
                # toward what the loop-closure constraint implies, weighted
                # by registration quality. This models the effect of a
                # backend pose-graph optimizer without implementing one.
                implied_y = kf_pose[0] + lc_reg.dy * self._scan_scale
                implied_x = kf_pose[1] + lc_reg.dx * self._scan_scale
                pull = float(np.clip(lc_reg.quality, 0.0, 1.0))
                info_gain = self.trace_cov * pull   # uncertainty "resolved" by the closure
                est_y = est_y + pull * (implied_y - est_y)
                est_x = est_x + pull * (implied_x - est_x)
                self.trace_cov *= (1.0 - 0.6 * pull)

        self.est_pose = (est_y, est_x, est_theta)

        # --- Change-detection reward shaping ---
        change_voxels_resolved = float(change_mask.sum())

        # --- Safety ---
        min_obstacle_dist = float(np.min(ranges)) if len(ranges) else self.cfg.sonar.max_range
        proximity_cost = max(0.0, 1.0 - min_obstacle_dist / 3.0)
        if collided:
            proximity_cost = 1.0

        # trace_cov grows with distance travelled without a loop closure
        # (models pose-uncertainty accumulation from dead reckoning).
        step_dist = np.hypot(self.true_pose[0] - prev_true[0], self.true_pose[1] - prev_true[1])
        self.trace_cov += 0.02 * (step_dist + (0.5 if dwell else 0.0))

        if step_dist < 0.05:
            self._stationary_streak += 1
        else:
            self._stationary_streak = 0

        reward_breakdown = compute_reward(
            entropy_delta=entropy_delta,
            info_gain=info_gain,
            proximity_cost=proximity_cost,
            loop_closure_validated=loop_closure_validated,
            change_voxels_resolved=change_voxels_resolved / (self.map.height * self.map.width),
            stationary_streak=self._stationary_streak,
            beta=beta,
            weights=self.cfg.reward_weights,
        )

        self._q_t = reg.quality if reg is not None else 0.0
        self._ell_t = ell_t
        self._last_reg = reg
        self._last_change_mask = change_mask
        self._last_collided = collided
        self._last_loop_closure = loop_closure_validated

        terminated = False
        truncated = self.t >= self.cfg.max_steps or self.battery <= 0

        obs = self._build_observation()
        info = self._build_info(reward_breakdown)

        # bookkeeping for evaluation metrics
        self._ate_accumulator.append(self._current_ate())
        self._path_length += step_dist
        self._collision_count += int(collided)
        self._trajectory_true.append(self.true_pose)
        self._trajectory_est.append(self.est_pose)

        return obs, reward_breakdown.total, terminated, truncated, info

    def render(self):
        # Rich rendering lives in visualization/live_viewer.py, which reads
        # this env's public state (world.occ, map.prob, trajectories) —
        # kept separate so headless training never imports matplotlib.
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _reset_internal_state(self):
        cfg = self.cfg
        world_cfg = cfg.world
        if world_cfg.seed is None:
            world_cfg = WorldConfig(**{**world_cfg.__dict__, "seed": int(self.rng.integers(0, 1_000_000))})
        self.world = TunnelWorld(world_cfg)
        self.map = OccupancyGrid(self.world.occ.shape[0], self.world.occ.shape[1])
        self.sonar = SonarModel(self.world, cfg.sonar, rng=self.rng)
        self.loop_detector = LoopClosureDetector()
        # Rebuilt fresh every episode against the *current* self.rng, same
        # reasoning as self.sonar above (see the long comment in __init__):
        # a fresh simulated gyro/DVL unit for a fresh deployment, and a
        # binding that survives a seeded reset() reassigning self.rng.
        #
        # Gated on use_sfm_fusion, same reasoning as the sensing call in
        # step(): IMUModel.reset() draws from self.rng to sample this
        # episode's true gyro bias, so doing this unconditionally would
        # also break the "use_sfm_fusion=False reproduces pre-SfM behavior
        # byte-for-byte" promise on EnvConfig.
        if cfg.use_sfm_fusion:
            self.imu_model = IMUModel(cfg.imu, rng=self.rng)
            self.imu_model.reset()   # samples this episode's true gyro bias
            self.dvl_model = DVLModel(cfg.dvl, rng=self.rng)
        else:
            self.imu_model = None
            self.dvl_model = None
        self.sfm.reset()   # no rng draw of its own; fresh bias *belief* either way
        self.sfm2d_imaging.reset()
        self.sfm2d_scanning.reset()
        self._last_sfm2d = None

        self.true_pose = self.world.start_pose
        self.est_pose = self.world.start_pose
        self.t = 0
        self.battery = cfg.battery_capacity
        self.trace_cov = 0.1
        self._prev_frame = None
        self._prev_frame_mode = None
        self._prev_entropy = self.map.entropy_normalized()
        self._q_t = 0.0
        self._ell_t = 0.0
        self._last_reg = None
        self._last_fusion = None
        self._last_change_mask = np.zeros_like(self.map.prob, dtype=bool)
        self._last_collided = False
        self._last_loop_closure = False
        self._scan_scale = cfg.sonar.max_range / (cfg.sonar.frame_size / 2.0)
        self._min_trace_cov_for_bonus = 0.15
        self._loop_closure_cooldown = 12   # steps
        self._last_validated_closure_step = -10_000
        self._stationary_streak = 0

        self._ate_accumulator = []
        self._path_length = 0.0
        self._collision_count = 0
        self._trajectory_true = [self.true_pose]
        self._trajectory_est = [self.est_pose]

    def _apply_action(self, action: int):
        y, x, theta = self.true_pose
        dwell = False
        collided = False
        noise_xy = self.cfg.process_noise_xy
        noise_th = np.deg2rad(self.cfg.process_noise_theta_deg)

        if action in (0, 1, 2):
            dist = {0: 0.5, 1: 1.0, 2: 2.0}[action]
            dy = dist * np.sin(theta) + self.rng.normal(0, noise_xy)
            dx = dist * np.cos(theta) + self.rng.normal(0, noise_xy)
            new_y, new_x = y + dy, x + dx
            if self.world.is_free(new_y, new_x):
                y, x = new_y, new_x
            else:
                collided = True
        elif action in (3, 4):
            dtheta = {3: -np.pi / 6, 4: np.pi / 6}[action]
            theta = theta + dtheta + self.rng.normal(0, noise_th)
        elif action == 5:
            dwell = True
        elif action == 6:
            target = self._best_known_loop_candidate_pose()
            if target is not None:
                ty, tx = target[0], target[1]
                heading = np.arctan2(ty - y, tx - x)
                theta = heading + self.rng.normal(0, noise_th)
                dist = 1.0
                new_y = y + dist * np.sin(theta)
                new_x = x + dist * np.cos(theta)
                if self.world.is_free(new_y, new_x):
                    y, x = new_y, new_x
                else:
                    collided = True
            else:
                dwell = True  # nothing to revisit yet -> just scan in place

        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        self.true_pose = (y, x, theta)
        return collided, dwell

    def _best_known_loop_candidate_pose(self):
        if not self.loop_detector.keyframes:
            return None
        idx = len(self.loop_detector.keyframes) // 2  # revisit an established, not-most-recent, keyframe
        return self.loop_detector.keyframe_pose(idx)

    def _integrate_odometry_from_delta(self, delta):
        """Rotate a body-frame (dx, dy, dtheta) delta into world coordinates
        using the *previous* estimated heading, and add it onto the running
        pose estimate. This is the shared math both `_integrate_odometry`
        (FS2D alone) and `step`'s SfM-fused path (FS2D+IMU+DVL, see
        fusion/sfm.py) integrate through -- factored out so the fused path
        doesn't have to duplicate it.
        """
        ey, ex, eth = self.est_pose
        dx_body, dy_body, dtheta = delta
        c, s = np.cos(eth), np.sin(eth)
        world_dx = c * dx_body - s * dy_body
        world_dy = s * dx_body + c * dy_body
        new_ey = ey + world_dy
        new_ex = ex + world_dx
        new_eth = eth + dtheta
        new_eth = (new_eth + np.pi) % (2 * np.pi) - np.pi
        return new_ey, new_ex, new_eth

    def _integrate_odometry(self, reg):
        """FS2D-only pose integration, with no IMU/DVL fusion. Kept as its
        own method (rather than inlining) for two reasons: it's the
        fallback path when `cfg.use_sfm_fusion=False`, and it's the exact
        pre-SfM behavior this codebase had before fusion/sfm.py existed --
        useful to keep reproducible on its own.
        """
        if reg is None:
            return self.est_pose
        return self._integrate_odometry_from_delta(
            (reg.dx * self._scan_scale, reg.dy * self._scan_scale, reg.dtheta))

    def _update_map_from_beams(self, y, x, theta, ranges, angles):
        step = 1.0
        for r, a in zip(ranges, angles):
            beam_theta = theta + a
            dyv, dxv = np.sin(beam_theta), np.cos(beam_theta)
            free_cells = []
            rr = 0.0
            while rr < r:
                rr += step
                py, px = y + rr * dyv, x + rr * dxv
                free_cells.append((int(round(py)), int(round(px))))
            hit_cell = None
            if r < self.cfg.sonar.max_range - 1e-6:
                hit_cell = (int(round(y + r * dyv)), int(round(x + r * dxv)))
            self.map.update_beam((int(round(y)), int(round(x))), free_cells, hit_cell)

    def _crop_patch(self, y, x):
        ps = self.cfg.patch_size
        half = ps // 2
        h, w = self.map.height, self.map.width
        cy, cx = int(round(y)), int(round(x))

        def crop(arr):
            out = np.zeros((ps, ps), dtype=np.float32)
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            sy0, sy1 = max(0, y0), min(h, y1)
            sx0, sx1 = max(0, x0), min(w, x1)
            if sy1 <= sy0 or sx1 <= sx0:
                return out
            oy0, ox0 = sy0 - y0, sx0 - x0
            out[oy0:oy0 + (sy1 - sy0), ox0:ox0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
            return out

        return crop

    def _local_unknown_fraction(self, y, x) -> float:
        """Fraction of the local map crop around (y, x) that's still
        unresolved (belief near 0.5) -- the geometry signal compute_beta
        uses to keep exploration credit alive in a wide, largely-
        unexplored area even if the loop-closure clock has run out. Same
        crop size/logic as the observation patch, same "near 0.5" test
        FrontierBasedPolicy already uses to detect unknown space.
        """
        crop = self._crop_patch(y, x)
        belief_patch = crop(self.map.prob)
        # Padding beyond the map edge is filled with 0.0 by _crop_patch,
        # which would be misread as "known free" -- restrict the fraction
        # to the region actually inside the map bounds.
        ps = self.cfg.patch_size
        half = ps // 2
        h, w = self.map.height, self.map.width
        cy, cx = int(round(y)), int(round(x))
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        oy0, ox0 = y0 - (cy - half), x0 - (cx - half)
        valid_region = belief_patch[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]
        if valid_region.size == 0:
            return 0.0
        unknown = np.abs(valid_region - 0.5) < 0.15
        return float(unknown.mean())

    def _build_observation(self):
        y, x, _ = self.est_pose
        crop = self._crop_patch(y, x)
        belief_patch = crop(self.map.prob)
        uncertainty_patch = crop(self.map.uncertainty())
        change_patch = crop(self._last_change_mask.astype(np.float32))
        patch = np.stack([belief_patch, uncertainty_patch, change_patch], axis=0).astype(np.float32)

        battery_frac = float(np.clip(self.battery / self.cfg.battery_capacity, 0.0, 1.0))
        time_frac = float(np.clip(1.0 - self.t / self.cfg.max_steps, 0.0, 1.0))
        # frame_mode_flag: 0.0 = last sonar frame was imaging (narrow FOV,
        # cheap), 1.0 = scanning/360-deg (dense, costs a dwell). Without
        # this the policy has no way to distinguish "q_t is low because the
        # registration itself is uncertain" from "q_t is low because the
        # cheap imaging modality structurally can't do much better" -- see
        # registration/fs2d.py's fold-ambiguity discussion and
        # env/sonar_model.py's "MODALITY TAGGING" section for why the two
        # modalities' registration reliability differs substantially.
        # This also gives the policy the context it needs to learn when
        # dwelling for a scanning-sonar frame is worth the battery cost.
        frame_mode_flag = 1.0 if self._prev_frame_mode == "scanning" else 0.0
        scalars = np.array([
            self._q_t,
            self._ell_t,
            float(np.clip(self.trace_cov, 0.0, 10.0)),
            battery_frac,
            time_frac,
            frame_mode_flag,
        ], dtype=np.float32)
        return {"patch": patch, "scalars": scalars}

    def _current_ate(self):
        ty, tx, _ = self.true_pose
        ey, ex, _ = self.est_pose
        return float(np.hypot(ty - ey, tx - ex))

    def _build_info(self, reward_breakdown):
        info = {
            "true_pose": self.true_pose,
            "est_pose": self.est_pose,
            "q_t": self._q_t,
            "ell_t": self._ell_t,
            "trace_cov": self.trace_cov,
            "collided": self._last_collided,
            "loop_closure_validated": self._last_loop_closure,
            # Which sonar modality produced this step's frame/registration
            # -- see env/sonar_model.py's "MODALITY TAGGING" section and
            # registration/fs2d.py's fold-ambiguity discussion for why
            # imaging vs scanning registration reliability differs
            # substantially. Lets metrics/plotting.py break q_t and other
            # per-step diagnostics down by modality instead of only seeing
            # an unexplained-looking mixed distribution.
            "frame_mode": self._prev_frame_mode,
            "beta": reward_breakdown.beta if reward_breakdown is not None else self.cfg.reward_weights.beta_initial,
            "beta_decay_rate": self._decay_controller.decay_rate,
            "ate": self._current_ate(),
            "path_length": self._path_length,
            "collision_count": self._collision_count,
            "battery": self.battery,
            "map_completeness": self.map.completeness(
                self.world.occ, threshold=self.cfg.map_completeness_threshold),
            "map_entropy": self.map.entropy(),
            # Diagnostics from fusion/sfm.py's StateFusionModule -- None
            # when cfg.use_sfm_fusion=False, or right after reset() before
            # any step has run. Not consumed by reward/observations (see
            # fusion/sfm.py's module docstring for why trace_cov above is
            # deliberately left as its own separate reward-shaping proxy,
            # untouched by this feature); this is for plotting/analysis,
            # e.g. watching bias_estimate_deg converge over an episode.
            "sfm": None if self._last_fusion is None else {
                "used_fs2d": self._last_fusion.used_fs2d,
                "used_dvl": self._last_fusion.used_dvl,
                "fs2d_rejected_outlier": self._last_fusion.fs2d_rejected_outlier,
                "bias_estimate_deg": self._last_fusion.bias_estimate_deg,
                "bias_std_deg": self._last_fusion.bias_std_deg,
                "covariance_trace": float(np.trace(self._last_fusion.covariance)),
            },
            # perception/sfm2d.py's StructureFromMotion2D -- the actual
            # Structure-from-Motion, entirely separate from "sfm" above
            # (see that module's docstring). None when cfg.use_sfm2d=False
            # or right after reset(). n_landmarks_* report both maps'
            # sizes regardless of which modality produced this step's
            # frame, since both maps persist and are worth watching
            # together; the rest (pose_correction/n_matched/etc.) is
            # specific to *this step's* active modality.
            "sfm2d": None if not self.cfg.use_sfm2d else {
                "active_modality": self._prev_frame_mode,
                "n_landmarks_imaging": len(self.sfm2d_imaging.get_map()),
                "n_landmarks_scanning": len(self.sfm2d_scanning.get_map()),
                "pose_correction": None if self._last_sfm2d is None else self._last_sfm2d.pose_correction,
                "n_matched": None if self._last_sfm2d is None else self._last_sfm2d.n_matched,
                "n_new_landmarks": None if self._last_sfm2d is None else self._last_sfm2d.n_new_landmarks,
                "residual_rms": None if self._last_sfm2d is None else self._last_sfm2d.residual_rms,
            },
        }
        if reward_breakdown is not None:
            info["reward_breakdown"] = reward_breakdown.__dict__
        return info
