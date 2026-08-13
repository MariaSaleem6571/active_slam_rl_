"""
MarineGym adapter (thesis section 5.9: "Primary Training Platform:
MarineGym").

READ docs/MARINEGYM_INTEGRATION.md FIRST -- it has the verified install
steps, the honest status of what MarineGym publicly supports today, and
the full reasoning behind the design here. This docstring is the short
version.

WHAT'S VERIFIED VS. WHAT YOU MUST CONFIRM
-------------------------------------------
Verified against MarineGym's own docs/repo (github.com/Marine-RL/MarineGym,
marinegym.netlify.app) as of this writing:
  * Install steps, system requirements, and the `python train.py task=...`
    launch convention (see docs/MARINEGYM_INTEGRATION.md).
  * Five task environments are publicly released: Hover, Circle Tracking,
    Helical Tracking, Lemniscate Tracking, Landing. Supported UUV models
    include BlueROV and BlueROVHeavy.
  * A sonar-based task is explicitly listed by the maintainers as "under
    development" -- it is NOT in the public repository. There is no
    `BlueROV2SonarTask` you can import.

NOT independently verified (MarineGym is built on OmniDrones, whose tasks
follow a predictable TorchRL/tensordict `EnvBase`-style structure, but the
exact class names in MarineGym's own source were not available to check
from here):
  * The precise base class / import path for authoring a new Task.
  * The exact tensordict key names for actions/observations/thruster
    commands.
Every place below that depends on one of these is marked
`# VERIFY AGAINST YOUR CLONE` with a pointer to the file to open.

DESIGN: a thin wrapper, not a rewrite
--------------------------------------
`MarineGymActiveSlamEnv` subclasses `ActiveSlamEnv` and overrides only the
three methods that are simulator-specific:

  * `_reset_internal_state` -- builds a `MarineGymWorldAdapter` (wraps
    occupancy/collision queries) and a `MarineGymSonarAdapter` (wraps your
    existing Isaac-Sim sonar module) with the same `.is_free()`/`.occ`
    and `.sense_imaging()`/`.sense_scanning_360()` interfaces
    `TunnelWorld`/`SonarModel` already expose.
  * `_apply_action` -- sends real thruster/velocity commands into the
    MarineGym task and steps physics, instead of the kinematic integrator.
  * `close()` -- shuts down the Isaac Sim app cleanly.

Everything else -- FS2D registration, the occupancy grid, change
detection, loop closure, the reward, the state encoder, PPO, metrics,
plotting -- is inherited completely unmodified. That's only possible
because `ActiveSlamEnv.step()` only ever talks to `self.world` and
`self.sonar` through those four methods; see sim_env.py if you want to
confirm this yourself.

TWO INTEGRATION STRATEGIES (see the doc for the full tradeoff)
-----------------------------------------------------------------
1. **This file**: one MarineGym task instance wrapped as an ordinary
   Gymnasium env. Fast to get correct, reuses your whole tested pipeline
   unchanged, but runs at ordinary (single-environment) speed -- you get
   MarineGym's physics/sonar fidelity, not its GPU-parallel throughput.
2. **Batched/GPU** (sketched, not implemented, in the doc): port the
   mapping/reward math to batched PyTorch tensors so thousands of
   environments step in lockstep on the GPU, the way MarineGym's own
   Hover/Track/Landing tasks reach their headline throughput. Substantially
   more engineering; start with option 1, move to this once it's correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from active_slam_rl.env.sim_env import ActiveSlamEnv, EnvConfig


# --------------------------------------------------------------------- #
# Duck-typed interfaces: anything satisfying these can be swapped in for
# TunnelWorld / SonarModel without changing ActiveSlamEnv at all.
# --------------------------------------------------------------------- #
class WorldLike(Protocol):
    occ: np.ndarray
    start_pose: tuple

    def is_free(self, y: float, x: float) -> bool: ...


class SonarLike(Protocol):
    def sense_imaging(self, y: float, x: float, theta: float): ...
    def sense_scanning_360(self, y: float, x: float, theta: float): ...


@dataclass
class MarineGymTaskConfig:
    """Everything needed to launch one MarineGym task instance. Field
    names mirror MarineGym's Hydra config groups (`task.*` overrides you'd
    otherwise pass on the `train.py` command line) -- see
    docs/MARINEGYM_INTEGRATION.md for how these map to real config files
    under `~/MarineGym/cfg/`.
    """
    task_name: str = "Hover"          # VERIFY: the closest existing task to subclass/copy for your sonar task
    drone_model: str = "BlueROVHeavy"  # verified option, per MarineGym's cfg
    num_envs: int = 1                  # 1 for this single-env wrapper; see strategy 2 for >1
    headless: bool = True
    world_size_m: float = 45.0         # half-extent of the voxelization region around the scene origin
    voxel_resolution_m: float = 0.5
    device: str = "cuda:0"


class MarineGymWorldAdapter:
    """Wraps a MarineGym/Isaac-Sim scene's collision geometry so it looks
    like `TunnelWorld` to `ActiveSlamEnv`: an `.occ` grid for the
    completeness metric, and `.is_free(y, x)` for raycasting/collision.

    Isaac Sim exposes scene-query APIs (e.g.
    `omni.isaac.core.utils.bounds` / physx scene-query raycasts) that can
    voxelize a static collision mesh once at scene load -- this class
    caches that result rather than re-querying Isaac Sim on every sonar
    beam, which would be far too slow. Fill in `_voxelize_scene` with your
    scene's actual asset/collision-mesh query; it's intentionally left as
    a documented extension point rather than guessed at, since it depends
    on which USD stage/tunnel or wreck asset you load.
    """

    def __init__(self, mg_task, cfg: MarineGymTaskConfig):
        self._task = mg_task
        self.cfg = cfg
        n = int(2 * cfg.world_size_m / cfg.voxel_resolution_m)
        self.occ = self._voxelize_scene(n)
        self.start_pose = self._read_start_pose()

    def _voxelize_scene(self, n: int) -> np.ndarray:
        """Returns an (n, n) uint8 grid, 1 = occupied/collision, 0 = free.

        TODO (VERIFY AGAINST YOUR CLONE): replace this placeholder with a
        real query against the loaded USD stage's collision mesh, e.g.
        casting a dense grid of downward/planar physx raycasts across the
        scene's bounding box and marking a hit as occupied. This is a
        one-time cost per scene (cache the result), not a per-step cost.
        """
        raise NotImplementedError(
            "Fill in _voxelize_scene() with a real collision-mesh query "
            "against your loaded MarineGym/Isaac-Sim stage. See "
            "docs/MARINEGYM_INTEGRATION.md, 'Building the ground-truth "
            "occupancy grid'."
        )

    def _read_start_pose(self) -> tuple:
        """(y, x, theta) in the same 2D plan-view convention ActiveSlamEnv
        uses elsewhere -- project the vehicle's true 3D pose from Isaac
        Sim down onto its horizontal plane.

        # VERIFY AGAINST YOUR CLONE: replace `self._task.get_state()` with
        # however your Task actually exposes the articulation's world pose
        # (in OmniDrones-style tasks this is commonly something like
        # `self._task.drone.get_world_poses()` -- open
        # ~/MarineGym/marinegym/ and confirm the real accessor).
        """
        pos, quat = self._task.get_state()  # VERIFY AGAINST YOUR CLONE
        x, y = float(pos[0]), float(pos[1])
        theta = _yaw_from_quat(quat)
        return (y, x, theta)

    def is_free(self, y: float, x: float) -> bool:
        h, w = self.occ.shape
        res = self.cfg.voxel_resolution_m
        half = self.cfg.world_size_m
        yi = int(round((y + half) / res))
        xi = int(round((x + half) / res))
        if 0 <= yi < h and 0 <= xi < w:
            return self.occ[yi, xi] == 0
        return False


class MarineGymSonarAdapter:
    """Wraps your existing Isaac-Sim sonar module (the range-sensor
    wrapper + Warp-based GPU-parallel version from prior work) so it
    exposes the exact two methods `ActiveSlamEnv.step()` calls:
    `sense_imaging(y, x, theta)` and `sense_scanning_360(y, x, theta)`,
    each returning `(ranges, angles, hit_points, frame)` -- identical to
    `SonarModel` in env/sonar_model.py.

    This class deliberately does NOT reimplement sonar physics -- that's
    already done in your prior sonar work. It only translates between
    call conventions. Replace `self._sonar_module.<...>` below with your
    actual module's real method names.
    """

    def __init__(self, sonar_module, frame_size: int = 64, max_range: float = 22.0):
        self._sonar_module = sonar_module   # your existing Isaac-Sim sonar wrapper
        self.frame_size = frame_size
        self.max_range = max_range

    def sense_imaging(self, y: float, x: float, theta: float):
        # VERIFY AGAINST YOUR CLONE: substitute your sonar module's real
        # "give me one imaging-sonar frame at this pose" call. If your
        # existing implementation already returns range/intensity data in
        # Isaac Sim's own layout, do the reshape/rescale here so the
        # output tuple exactly matches SonarModel.sense_imaging's shape
        # contract (ranges: (n_beams,), angles: (n_beams,) radians
        # relative to heading, hit_points: list of (y, x), frame: 2D
        # (frame_size, frame_size) egocentric image in [0, 1]).
        raw = self._sonar_module.get_imaging_frame(pose=(y, x, theta))  # VERIFY AGAINST YOUR CLONE
        return self._to_common_format(raw)

    def sense_scanning_360(self, y: float, x: float, theta: float):
        raw = self._sonar_module.get_scanning_sweep(pose=(y, x, theta))  # VERIFY AGAINST YOUR CLONE
        return self._to_common_format(raw)

    def _to_common_format(self, raw):
        """Adjust field names here to match whatever your sonar module's
        return type actually is (dataclass, dict, tensor, ...)."""
        ranges = np.asarray(raw.ranges, dtype=np.float64)
        angles = np.asarray(raw.angles, dtype=np.float64)
        hit_points = list(raw.hit_points)
        frame = np.asarray(raw.frame, dtype=np.float64)
        return ranges, angles, hit_points, frame


def _yaw_from_quat(quat) -> float:
    """Extract yaw (rotation about the vertical axis) from a
    (w, x, y, z) quaternion -- the standard Isaac Sim / USD ordering."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class MarineGymActiveSlamEnv(ActiveSlamEnv):
    """A single MarineGym task instance, wrapped in ActiveSlamEnv's
    Gymnasium interface. See the module docstring and
    docs/MARINEGYM_INTEGRATION.md for the full picture.
    """

    def __init__(self, mg_config: MarineGymTaskConfig, sonar_module,
                 env_config: EnvConfig = EnvConfig(), render_mode: Optional[str] = None):
        self.mg_config = mg_config
        self._sonar_module = sonar_module
        self._sim_app = None
        self._mg_task = None
        super().__init__(env_config, render_mode)

    # ------------------------------------------------------------------ #
    def _launch_isaac_sim(self):
        """Standard Isaac Sim standalone launch. This part IS verified --
        it's core Isaac Sim usage, documented at
        https://docs.omniverse.nvidia.com/isaacsim/, independent of
        MarineGym specifically. Must run before any `omni.*` import.
        """
        if self._sim_app is not None:
            return
        from isaacsim import SimulationApp
        self._sim_app = SimulationApp({"headless": self.mg_config.headless})

    def _build_marinegym_task(self):
        """# VERIFY AGAINST YOUR CLONE.
        Open ~/MarineGym/scripts/train.py and trace how `task=Hover`
        resolves to a Python object -- MarineGym uses Hydra config
        composition (cfg/ directory) to build a task, most likely via a
        registry/factory function in `marinegym/envs/__init__.py` or
        similar. Mirror that exact call here with `num_envs=1` and your
        sonar-task config/scene in place of `Hover`'s. Since sonar tasks
        aren't in the public release, this is the one piece of real
        authorship this integration requires of you -- likely a new file
        `marinegym/tasks/bluerov2_sonar.py` following the same structure
        as their existing Hover/Track/Landing task files, registering a
        `BlueROV2Sonar` task name the same way theirs are registered.
        """
        raise NotImplementedError(
            "Replace this with your MarineGym task construction call. "
            "See the VERIFY AGAINST YOUR CLONE note above and "
            "docs/MARINEGYM_INTEGRATION.md, 'Authoring the sonar task'."
        )

    # ------------------------------------------------------------------ #
    def _reset_internal_state(self):
        cfg = self.cfg
        self._launch_isaac_sim()
        if self._mg_task is None:
            self._mg_task = self._build_marinegym_task()

        # VERIFY AGAINST YOUR CLONE: your Task's actual reset call --
        # OmniDrones-style tasks typically return/accept a TensorDict.
        self._mg_task.reset()  # VERIFY AGAINST YOUR CLONE

        self.world = MarineGymWorldAdapter(self._mg_task, self.mg_config)
        self.sonar = MarineGymSonarAdapter(self._sonar_module,
                                            frame_size=cfg.sonar.frame_size,
                                            max_range=cfg.sonar.max_range)
        self.map = self.map.__class__(self.world.occ.shape[0], self.world.occ.shape[1])
        from active_slam_rl.perception.loop_closure import LoopClosureDetector
        self.loop_detector = LoopClosureDetector()

        self.true_pose = self.world.start_pose
        self.est_pose = self.world.start_pose
        self.t = 0
        self.battery = cfg.battery_capacity
        self.trace_cov = 0.1
        self._prev_frame = None
        self._prev_entropy = self.map.entropy_normalized()
        self._q_t = 0.0
        self._ell_t = 0.0
        self._last_reg = None
        self._last_change_mask = np.zeros_like(self.map.prob, dtype=bool)
        self._last_collided = False
        self._last_loop_closure = False
        self._scan_scale = cfg.sonar.max_range / (cfg.sonar.frame_size / 2.0)
        self._min_trace_cov_for_bonus = 0.15
        self._loop_closure_cooldown = 12
        self._last_validated_closure_step = -10_000
        self._stationary_streak = 0
        self._ate_accumulator = []
        self._path_length = 0.0
        self._collision_count = 0
        self._trajectory_true = [self.true_pose]
        self._trajectory_est = [self.est_pose]

    def _apply_action(self, action: int):
        """Maps the same 7 discrete motion primitives onto real thruster/
        velocity commands, steps Isaac Sim physics, then reads the true
        pose back.

        # VERIFY AGAINST YOUR CLONE: the exact command tensor MarineGym's
        # BlueROV2 task expects (likely a per-thruster force/velocity
        # vector, set via something like `self._mg_task.apply_action(cmd)`
        # in OmniDrones-style tasks) and how many physics substeps to
        # advance per env.step() to match this repo's `step_len`-scaled
        # kinematics in world_generator.py.
        """
        y, x, theta = self.true_pose
        collided = False
        dwell = False

        command = _action_to_thruster_command(action, theta)  # VERIFY AGAINST YOUR CLONE
        if action == 5:
            dwell = True
        self._mg_task.apply_action(command)  # VERIFY AGAINST YOUR CLONE
        self._mg_task.step_physics()          # VERIFY AGAINST YOUR CLONE

        pos, quat = self._mg_task.get_state()  # VERIFY AGAINST YOUR CLONE
        new_x, new_y = float(pos[0]), float(pos[1])
        new_theta = _yaw_from_quat(quat)

        collided = bool(getattr(self._mg_task, "collision_flag", False))  # VERIFY AGAINST YOUR CLONE
        self.true_pose = (new_y, new_x, new_theta)
        return collided, dwell

    def close(self):
        if self._sim_app is not None:
            self._sim_app.close()
            self._sim_app = None


def _action_to_thruster_command(action: int, theta: float):
    """Placeholder mapping from the 7 discrete actions to whatever
    command vector your MarineGym BlueROV2 task's apply_action() expects.
    A reasonable starting point for a 6-thruster BlueROV2 Heavy: forward
    actions drive surge thrust, yaw actions drive differential thrust
    between port/starboard thrusters, dwell/revisit are near-zero-thrust
    holds. Tune against your task's actual actuator model.
    """
    surge = {0: 0.3, 1: 0.6, 2: 1.0}.get(action, 0.0)
    yaw = {3: -0.5, 4: 0.5}.get(action, 0.0)
    return np.array([surge, 0.0, 0.0, 0.0, 0.0, yaw], dtype=np.float32)
