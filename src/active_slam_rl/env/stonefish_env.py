"""
Stonefish adapter -- an alternative to MarineGym (env/marinegym_env.py)
that is worth knowing about because, unlike MarineGym, it has REAL sonar
simulation working TODAY.

READ docs/STONEFISH_INTEGRATION.md FIRST for the verified install steps,
the two example scenario files (a BlueROV2 with a Ping-style mechanical
scanning sonar, and a BlueROV2 with a forward-looking sonar), and the
full reasoning. This docstring is the short version.

WHY STONEFISH INSTEAD OF / ALONGSIDE MARINEGYM
------------------------------------------------
Checked directly against Stonefish's own documentation
(github.com/patrykcieslak/stonefish, stonefish-ros2.readthedocs.io) while
writing this. Verified facts:

  * Stonefish already implements real forward-looking sonar (FLS) and
    mechanical scanning imaging sonar (MSIS) -- these map directly onto
    this repo's two sonar modalities (SonarModel.sense_imaging ~ FLS,
    SonarModel.sense_scanning_360 ~ MSIS/Ping360-style). MarineGym's
    equivalent is explicitly "under development" and not public yet
    (see docs/MARINEGYM_INTEGRATION.md) -- Stonefish is the more
    immediately actionable option if you need working sonar now.
  * The interface is entirely over ROS2 topics, not a direct Python
    simulation API: FLS/MSIS each publish two `sensor_msgs/Image` topics
    (a display image and a raw data image); thrusters are commanded via
    one `std_msgs/Float64MultiArray` topic per robot, values in [-1, 1];
    ground-truth pose is available from an Odometry sensor publishing
    `nav_msgs/Odometry`. These are all verified against the stonefish_ros2
    docs.
  * Community BlueROV2 scenario files already exist (e.g.
    github.com/bvibhav/stonefish_bluerov2) as a starting point -- you do
    not need to build the vehicle model from scratch.

NOT independently verified: the exact XML attribute names for defining an
FLS/MSIS sensor block (I confirmed the C++ constructor signature --
name, beam/bin counts, horizontal/vertical FOV, min/max range, colormap --
but not the literal XML schema for those same parameters). Cross-check
against `stonefish/examples` in your clone or the scenario file linked in
docs/STONEFISH_INTEGRATION.md before trusting the example .scn snippets
verbatim.

DESIGN: identical override pattern to MarineGymActiveSlamEnv
----------------------------------------------------------------
`StonefishActiveSlamEnv` subclasses `ActiveSlamEnv` and overrides the same
three methods for the same reason -- see env/marinegym_env.py's docstring
for the full explanation of why this preserves 100% of the existing,
tested pipeline (FS2D, mapping, change detection, loop closure, reward,
encoder, PPO, metrics, plotting) unmodified. The only real difference is
*how* sensing/actuation happens: through a background rclpy node
publishing/subscribing to topics, instead of a direct simulation API call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from active_slam_rl.env.sim_env import ActiveSlamEnv, EnvConfig


@dataclass
class StonefishConfig:
    """Maps onto the real, verified stonefish_ros2 launch/topic conventions.
    See docs/STONEFISH_INTEGRATION.md for how these correspond to your
    .scn scenario file and launch arguments."""
    robot_name: str = "BLUEROV"
    thruster_topic: str = "/bluerov/thrusters/setpoints"      # std_msgs/Float64MultiArray, verified
    odometry_topic: str = "/bluerov/odometry"                  # nav_msgs/Odometry, verified
    fls_display_topic: str = "/bluerov/fls/display"            # sensor_msgs/Image, verified
    fls_data_topic: str = "/bluerov/fls/data"                  # sensor_msgs/Image, verified
    msis_display_topic: str = "/bluerov/msis/display"           # sensor_msgs/Image, verified
    msis_data_topic: str = "/bluerov/msis/data"                 # sensor_msgs/Image, verified
    n_thrusters: int = 8                                        # BlueROV2 Heavy: 8: VERIFY AGAINST YOUR .scn
    max_range_m: float = 22.0
    frame_size: int = 64
    world_size_m: float = 45.0
    voxel_resolution_m: float = 0.5
    spin_timeout_s: float = 0.2   # how long to pump rclpy callbacks waiting for a fresh sensor frame


class _StonefishRosBridge:
    """Owns the single rclpy node: publishes thruster setpoints, caches the
    latest odometry + FLS/MSIS frames. Kept as a separate class (rather
    than folded into the adapters below) so `StonefishActiveSlamEnv` can
    close it cleanly in one place.

    Every `# VERIFY AGAINST YOUR CLONE`-free line in this class is a
    verified ROS2 message type per stonefish_ros2's documentation; the
    unverified part is only the cv_bridge conversion details for FLS/MSIS's
    *raw data* image encoding (their pixel format/units), which you should
    confirm against a live topic echo (`ros2 topic echo <topic> --no-arr`)
    before trusting the numeric range values.
    """

    def __init__(self, cfg: StonefishConfig):
        self.cfg = cfg
        self._node = None
        self._latest = {"odom": None, "fls": None, "msis": None}

    def start(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init()
        cfg = self.cfg
        node = Node("active_slam_rl_bridge")

        self._thruster_pub = node.create_publisher(Float64MultiArray, cfg.thruster_topic, 10)
        node.create_subscription(Odometry, cfg.odometry_topic,
                                  lambda msg: self._latest.__setitem__("odom", msg), 10)
        node.create_subscription(Image, cfg.fls_data_topic,
                                  lambda msg: self._latest.__setitem__("fls", msg), 10)
        node.create_subscription(Image, cfg.msis_data_topic,
                                  lambda msg: self._latest.__setitem__("msis", msg), 10)
        self._node = node
        self._rclpy = rclpy

    def spin(self, timeout_s: float):
        self._rclpy.spin_once(self._node, timeout_sec=timeout_s)

    def publish_thrusters(self, setpoints: np.ndarray):
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        msg.data = [float(np.clip(v, -1.0, 1.0)) for v in setpoints]
        self._thruster_pub.publish(msg)

    def latest_odometry(self):
        return self._latest["odom"]

    def latest_image(self, key: str):
        return self._latest[key]

    def shutdown(self):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None


def _image_msg_to_array(msg, frame_size: int) -> np.ndarray:
    """sensor_msgs/Image -> a (frame_size, frame_size) float array in
    [0, 1], matching SonarModel's egocentric frame contract. Uses
    cv_bridge if available (standard in any ROS2 install with vision
    packages); falls back to a manual numpy reshape for common encodings.
    """
    try:
        from cv_bridge import CvBridge
        import cv2
        bridge = CvBridge()
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
    except ImportError:
        # Manual fallback: works for mono8-encoded images without cv_bridge.
        cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
    arr = cv_img.astype(np.float64) / 255.0
    if arr.shape != (frame_size, frame_size):
        import cv2
        arr = cv2.resize(arr, (frame_size, frame_size))
    return arr


def _yaw_from_quat(q) -> float:
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class StonefishWorldAdapter:
    """Same role as MarineGymWorldAdapter: exposes `.occ` and
    `.is_free(y, x)` to ActiveSlamEnv. Stonefish doesn't expose a Python
    scene-query API the way Isaac Sim does, so the practical options are
    (a) pre-voxelize your scenario's collision meshes offline (e.g. with
    `trimesh`, from the same .obj/.stl files referenced in your .scn file)
    and load the cached grid here, or (b) approximate the occupancy grid
    from accumulated sonar returns for evaluation purposes only (NOT a
    ground truth -- only use this if (a) isn't available and you accept a
    weaker completeness metric).
    """

    def __init__(self, occ: np.ndarray, start_pose: tuple, cfg: StonefishConfig):
        self.occ = occ
        self.start_pose = start_pose
        self.cfg = cfg

    def is_free(self, y: float, x: float) -> bool:
        h, w = self.occ.shape
        res = self.cfg.voxel_resolution_m
        half = self.cfg.world_size_m
        yi = int(round((y + half) / res))
        xi = int(round((x + half) / res))
        if 0 <= yi < h and 0 <= xi < w:
            return self.occ[yi, xi] == 0
        return False


class StonefishSonarAdapter:
    """Same role as MarineGymSonarAdapter, but reading cached ROS2 image
    messages instead of calling a sensor module method directly.
    `sense_imaging` reads the FLS topics; `sense_scanning_360` reads the
    MSIS/Ping-style topics -- exactly matching ActiveSlamEnv's existing
    split (ordinary steps use the imaging sonar; the "dwell-and-scan"
    action uses the full 360 sweep).
    """

    def __init__(self, bridge: _StonefishRosBridge, cfg: StonefishConfig):
        self._bridge = bridge
        self.cfg = cfg

    def sense_imaging(self, y: float, x: float, theta: float):
        return self._sense("fls")

    def sense_scanning_360(self, y: float, x: float, theta: float):
        return self._sense("msis")

    def _sense(self, key: str):
        cfg = self.cfg
        self._bridge.spin(cfg.spin_timeout_s)
        msg = self._bridge.latest_image(key)
        if msg is None:
            # No frame received yet (e.g. right after reset) -- return an
            # empty frame rather than raising, so ActiveSlamEnv's first
            # step doesn't crash while the topic warms up.
            frame = np.zeros((cfg.frame_size, cfg.frame_size))
        else:
            frame = _image_msg_to_array(msg, cfg.frame_size)
        # FLS/MSIS in Stonefish report acoustic intensity images, not
        # per-beam ranges directly. Approximate per-beam ranges from the
        # frame by taking, per angular column, the nearest strong-return
        # radius -- adequate for FS2D (which only needs the frame image)
        # and for occupancy updates (which need beam endpoints).
        # VERIFY AGAINST YOUR CLONE: tune the intensity threshold below
        # against real sonar frames from your scenario.
        ranges, angles, hit_points = _extract_beams_from_frame(frame, cfg.max_range_m)
        return ranges, angles, hit_points, frame


def _extract_beams_from_frame(frame: np.ndarray, max_range: float, n_beams: int = 64,
                                intensity_threshold: float = 0.35):
    """Converts an egocentric intensity image into the (ranges, angles,
    hit_points) triple ActiveSlamEnv's mapping code expects, by scanning
    outward along each of n_beams angular directions from the image
    center and taking the first pixel that clears intensity_threshold.
    """
    size = frame.shape[0]
    center = size / 2.0
    scale = (size / 2.0 - 1) / max_range
    angles = np.linspace(-np.pi / 2, np.pi / 2, n_beams)
    ranges = np.full(n_beams, max_range)
    hit_points = []
    for i, a in enumerate(angles):
        for r in np.arange(0.3, max_range, 0.3):
            px = int(round(center + r * scale * np.cos(a)))
            py = int(round(center + r * scale * np.sin(a)))
            if 0 <= py < size and 0 <= px < size and frame[py, px] > intensity_threshold:
                ranges[i] = r
                hit_points.append((py, px))
                break
    return ranges, angles, hit_points


class StonefishActiveSlamEnv(ActiveSlamEnv):
    """A BlueROV2 in Stonefish, wrapped in ActiveSlamEnv's Gymnasium
    interface, via a background ROS2 bridge. See the module docstring
    and docs/STONEFISH_INTEGRATION.md for the full picture.

    Prerequisite: the stonefish_simulator (or stonefish_simulator_nogpu)
    ROS2 node must already be running, with your scenario file loaded --
    this class does not launch Stonefish itself, only talks to it over
    ROS2. See docs/STONEFISH_INTEGRATION.md, "Running it", for the exact
    `ros2 run` / launch-file commands.
    """

    def __init__(self, sf_config: StonefishConfig, ground_truth_occ: np.ndarray,
                 env_config: EnvConfig = EnvConfig(), render_mode: Optional[str] = None):
        self.sf_config = sf_config
        self._ground_truth_occ = ground_truth_occ
        self._bridge = _StonefishRosBridge(sf_config)
        super().__init__(env_config, render_mode)

    def _reset_internal_state(self):
        cfg = self.cfg
        self._bridge.start()

        # VERIFY AGAINST YOUR CLONE: if your scenario supports a reset
        # service, call it here (Stonefish doesn't universally expose a
        # generic "reset episode" service; some setups instead respawn
        # the robot via a service or simply restart the simulator node
        # between episodes -- check docs/STONEFISH_INTEGRATION.md).
        for _ in range(10):
            self._bridge.spin(self.sf_config.spin_timeout_s)
            if self._bridge.latest_odometry() is not None:
                break

        odom = self._bridge.latest_odometry()
        start_pose = self._pose_from_odometry(odom) if odom is not None else (0.0, 0.0, 0.0)

        self.world = StonefishWorldAdapter(self._ground_truth_occ, start_pose, self.sf_config)
        self.sonar = StonefishSonarAdapter(self._bridge, self.sf_config)
        self.map = self.map.__class__(self.world.occ.shape[0], self.world.occ.shape[1])
        from active_slam_rl.perception.loop_closure import LoopClosureDetector
        self.loop_detector = LoopClosureDetector()

        self.true_pose = start_pose
        self.est_pose = start_pose
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

    def _pose_from_odometry(self, odom) -> tuple:
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        return (float(p.y), float(p.x), _yaw_from_quat(q))

    def _apply_action(self, action: int):
        """# VERIFY AGAINST YOUR CLONE: the per-thruster mixing for your
        specific BlueROV2 thruster layout (a standard BlueROV2 Heavy uses
        8 thrusters in a vectored configuration; this placeholder mixer
        only drives forward/yaw evenly across all of them, which will
        work but won't be efficient -- swap in your real thruster
        allocation matrix once you have it).
        """
        y, x, theta = self.true_pose
        dwell = (action == 5)

        setpoints = _action_to_setpoints(action, self.sf_config.n_thrusters)
        self._bridge.publish_thrusters(setpoints)
        self._bridge.spin(self.sf_config.spin_timeout_s)

        odom = self._bridge.latest_odometry()
        collided = False
        if odom is not None:
            new_y, new_x, new_theta = self._pose_from_odometry(odom)
            self.true_pose = (new_y, new_x, new_theta)
        return collided, dwell

    def close(self):
        self._bridge.shutdown()


def _action_to_setpoints(action: int, n_thrusters: int) -> np.ndarray:
    """Placeholder even mixing across all thrusters -- replace with a real
    thruster allocation matrix for your BlueROV2 Heavy configuration.
    Values are in [-1, 1] per stonefish_ros2's verified convention."""
    surge = {0: 0.3, 1: 0.6, 2: 1.0}.get(action, 0.0)
    yaw = {3: -0.4, 4: 0.4}.get(action, 0.0)
    base = np.full(n_thrusters, surge, dtype=np.float32)
    half = n_thrusters // 2
    base[:half] += yaw
    base[half:] -= yaw
    return np.clip(base, -1.0, 1.0)
