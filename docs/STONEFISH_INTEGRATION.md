# Integrating with Stonefish (a stronger immediate option than MarineGym)

This is a second simulator option, alongside `docs/MARINEGYM_INTEGRATION.md`,
worth using *specifically because* it has real sonar working today.

## 0. Why Stonefish, concretely

Checked directly against Stonefish's own documentation and repository
(github.com/patrykcieslak/stonefish, stonefish-ros2.readthedocs.io):

* Stonefish already implements **forward-looking sonar (FLS)** and
  **mechanical scanning imaging sonar (MSIS)** as real, working sensors —
  not "under development" like MarineGym's sonar support. These map
  directly onto this repo's two modalities: FLS ~ `SonarModel.sense_imaging`,
  MSIS ~ `SonarModel.sense_scanning_360` (a Ping360-style sensor).
* The interface is over **ROS2 topics**, not a direct Python simulation
  API: FLS/MSIS each publish two `sensor_msgs/Image` topics (display +
  raw data); thrusters take a single `std_msgs/Float64MultiArray` per
  robot with values in **[-1, 1]**; ground-truth pose comes from an
  Odometry sensor (`nav_msgs/Odometry`). All verified against
  stonefish_ros2's docs.
* Community BlueROV2 scenario files already exist —
  github.com/bvibhav/stonefish_bluerov2 and the "stonefish_ros2_marine_robotics"
  project — as a starting point, so you're not building the vehicle model
  from scratch.

What is **not** independently verified here: the exact XML attribute
names for defining an FLS/MSIS sensor block. I confirmed the underlying
C++ constructor signature (name, beam/bin counts, horizontal/vertical
FOV, min/max range, colormap) but not the literal XML schema for the
same parameters — cross-check the example scenario files below against
`stonefish/examples` in your clone before trusting them verbatim.

---

## 1. Installing Stonefish + stonefish_ros2 (verified)

```bash
# System deps
sudo apt update
sudo apt install -y build-essential cmake libglm-dev libsdl2-dev git

# Bullet Physics, a specific version, built with double precision
cd ~
git clone "https://github.com/bulletphysics/bullet3.git" -b 2.89
cd bullet3 && mkdir build && cd build
cmake -DBUILD_PYBULLET=OFF -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install

# Stonefish itself (the C++ library)
cd ~
git clone "https://github.com/patrykcieslak/stonefish.git"
cd stonefish && mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# stonefish_ros2, into your ROS2 workspace (assumes ROS2 is already installed)
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone "https://github.com/patrykcieslak/stonefish_ros2.git"
cd ~/ros2_ws
colcon build --packages-select stonefish_ros2
source install/setup.bash
```

Note (from the docs): the SDL2 CMake config sometimes needs a one-line
fix — if you hit an SDL2 link error, edit
`/usr/lib/x86_64-linux-gnu/cmake/SDL2/sdl2-config.cmake` and remove the
space after `-lSDL2`.

A discrete GPU with OpenGL 4.3+ is required for the graphical simulator
(and therefore for FLS/MSIS, which are rendering-based "vision sensors").
There is a `stonefish_simulator_nogpu` node for headless/no-GPU use, but
it explicitly does not simulate cameras or other rendering-dependent
sensors — so for this repo's purposes (sonar is essential), you need the
GPU-enabled path, at minimum with an offscreen/headless-but-GPU render
target.

## 2. Getting a BlueROV2 scenario running

Rather than building a vehicle model from scratch, start from the
community example:

```bash
cd ~/ros2_ws/src
git clone https://github.com/bvibhav/stonefish_bluerov2.git
# follow that repo's own README for its specific setup steps
```

Then run the simulator against your scenario file:

```bash
ros2 run stonefish_ros2 stonefish_simulator \
    <path_to_data_dir> <path_to_scenario.scn> 100 1200 800 high
```

(sampling rate 100 Hz, 1200x800 window, high rendering quality — tune to
your machine.)

### Example scenario snippet A — Ping-style BlueROV2 (MSIS only)

```xml
<?xml version="1.0"?>
<scenario>
  <robot name="BLUEROV" fixed="false">
    <!-- ... base BlueROV2 body/thruster definitions, e.g. from
         stonefish_bluerov2's own scenario file ... -->

    <sensor name="Ping360" type="msis">
      <!-- VERIFY AGAINST YOUR CLONE: confirmed C++ constructor params are
           roughly (name, num_beams, num_range_bins, horizontal_fov_deg,
           vertical_fov_deg, min_range_m, max_range_m, colormap) --
           confirm the exact XML attribute spelling against
           stonefish/examples or the sensors.html source in your clone. -->
      <specs beams="400" range_bins="500" horizontal_fov="360" vertical_fov="20"
             min_range="0.5" max_range="22.0"/>
      <colormap name="hot"/>
      <origin xyz="0.0 0.0 0.05" rpy="0.0 0.0 0.0"/>
      <link name="Vehicle"/>
      <ros_publisher topic="/bluerov/msis/data" display_topic="/bluerov/msis/display"/>
    </sensor>

    <ros_subscriber thrusters="/bluerov/thrusters/setpoints"/>
    <ros_publisher servos="/bluerov/joint_states"/>
  </robot>
</scenario>
```

### Example scenario snippet B — forward-looking-sonar BlueROV2 (FLS only)

```xml
<?xml version="1.0"?>
<scenario>
  <robot name="BLUEROV" fixed="false">
    <!-- ... same base BlueROV2 body/thruster definitions ... -->

    <sensor name="ARISLike" type="fls">
      <specs beams="512" range_bins="500" horizontal_fov="120.0" vertical_fov="30.0"
             min_range="0.5" max_range="10.0"/>
      <colormap name="hot"/>
      <origin xyz="0.3 0.0 0.0" rpy="0.0 0.0 0.0"/>
      <link name="Vehicle"/>
      <ros_publisher topic="/bluerov/fls/data" display_topic="/bluerov/fls/display"/>
    </sensor>

    <ros_subscriber thrusters="/bluerov/thrusters/setpoints"/>
    <ros_publisher servos="/bluerov/joint_states"/>
  </robot>
</scenario>
```

### Combined — what this repo's adapter actually expects

`ActiveSlamEnv` calls *both* `sense_imaging` (ordinary steps) and
`sense_scanning_360` (the dwell-and-scan action) on the same vehicle —
exactly matching the thesis's own hardware section, which pairs an
imaging sonar with a mechanical scanning sonar on one BlueROV2. So the
real scenario file you want merges both `<sensor>` blocks from A and B
above into one `<robot>` definition, with both `ros_publisher` topics
distinct (`/bluerov/fls/*` and `/bluerov/msis/*`), matching the topic
names already wired into `StonefishConfig` in `env/stonefish_env.py`.

Also add an Odometry sensor for ground-truth pose (used the same way
`MarineGymWorldAdapter._read_start_pose` uses Isaac Sim's state):

```xml
<sensor name="OdomTruth" type="odometry">
  <link name="Vehicle"/>
  <ros_publisher topic="/bluerov/odometry"/>
</sensor>
```

---

## 3. The adapter in this repo

`env/stonefish_env.py`'s `StonefishActiveSlamEnv` follows the identical
pattern as `MarineGymActiveSlamEnv` (see `docs/MARINEGYM_INTEGRATION.md`
section 2.3 for the full explanation): it subclasses `ActiveSlamEnv` and
overrides only `_reset_internal_state`, `_apply_action`, and `close`,
so FS2D, mapping, change detection, loop closure, the reward, the state
encoder, PPO, BC, metrics, and plotting all carry over completely
unmodified.

The mechanical difference from the MarineGym adapter: instead of calling
a direct simulation API, `_StonefishRosBridge` runs a background `rclpy`
node that publishes thruster setpoints and caches the latest odometry
and FLS/MSIS frames, since ROS2 communication is asynchronous
publish/subscribe rather than request/response.

```python
from active_slam_rl.env.stonefish_env import StonefishActiveSlamEnv, StonefishConfig
from active_slam_rl.env.sim_env import EnvConfig
import numpy as np

# Ground-truth occupancy grid for the completeness metric -- voxelize
# your scenario's collision mesh offline (e.g. with `trimesh`) and load
# it here; see StonefishWorldAdapter's docstring for the two practical
# options.
ground_truth_occ = np.load("my_scene_occupancy.npy")

sf_config = StonefishConfig(
    thruster_topic="/bluerov/thrusters/setpoints",
    odometry_topic="/bluerov/odometry",
    fls_data_topic="/bluerov/fls/data",
    msis_data_topic="/bluerov/msis/data",
)
env = StonefishActiveSlamEnv(sf_config, ground_truth_occ, env_config=EnvConfig())
obs, info = env.reset()
```

**Before running this**, the `stonefish_simulator` node must already be
running with your scenario loaded (Section 2 above) — `env.reset()`
connects to it over ROS2, it does not launch Stonefish itself.

From here `env` is a drop-in replacement anywhere `ActiveSlamEnv` was
used — `rl/train.py::train()`, `scripts/run_training.py`,
`scripts/run_eval.py`, `scripts/visualize_demo.py` all accept it
unchanged.

## 4. What to verify / tune before trusting this for real training

* **Thruster mixing** (`_action_to_setpoints`) is a placeholder even
  mixing across all thrusters — replace with your BlueROV2 Heavy's real
  thruster allocation matrix (8 vectored thrusters; consult
  bvibhav/stonefish_bluerov2 or Blue Robotics' own documentation for the
  layout).
* **Beam extraction from the sonar image** (`_extract_beams_from_frame`)
  is a simple intensity-threshold scan and its threshold needs tuning
  against real frames from your scenario — echo a live topic
  (`ros2 topic echo /bluerov/fls/data --no-arr`) and look at the actual
  intensity value range before trusting the default of 0.35.
* **The ground-truth occupancy grid** must come from your scenario's own
  geometry (Section 2's voxelization note) — there's no live scene-query
  API to pull it from Stonefish directly, unlike Isaac Sim.
* **Episode reset**: Stonefish doesn't universally expose a generic
  "reset episode to start pose" service across all scenarios — check
  whether your specific scenario/robot setup supports a reset service,
  or whether you need to restart the simulator node between episodes
  (much slower, but always correct).

## 5. What I could not verify from here

No ROS2, Stonefish, or ArduSub install available in this environment, so
none of `env/stonefish_env.py` has been executed against the real
simulator — only checked for internal consistency (imports cleanly
without `rclpy`/`cv_bridge` installed, subclasses `ActiveSlamEnv`
correctly, doesn't break the existing test suite). Before trusting it:

* Confirm the exact FLS/MSIS scenario XML syntax against your clone —
  the snippets above are reconstructed from the verified C++ constructor
  signature, not copied from a literal XML example I could access.
* Manually echo every topic in `StonefishConfig` (`ros2 topic echo ...`)
  and confirm message shapes/encodings match what
  `_image_msg_to_array`/`_pose_from_odometry` assume, before pointing
  PPO at this.
* Run `StonefishActiveSlamEnv` for a handful of manual `env.step()` calls
  with the graphical simulator visible, and confirm each action actually
  moves the vehicle the way you expect.
