# Integrating with MarineGym / Isaac Sim

This is the concrete plan and code for moving `ActiveSlamEnv` off the
lightweight built-in simulator and onto MarineGym's Isaac-Sim-backed
BlueROV2 simulation. Read this in full before touching code — the honest
scope section below will save you time.

---

## 0. Honest scope: what MarineGym actually supports today

Checked directly against MarineGym's own documentation and repository
(github.com/Marine-RL/MarineGym, marinegym.netlify.app) while writing
this:

* **MarineGym is real, open-source (MIT license), and actively
  maintained** — published at IROS 2025, built on
  [OmniDrones](https://github.com/btx0424/OmniDrones) and NVIDIA Isaac
  Sim. It supports five UUV models (BlueROV, BlueROVHeavy, iAUV, LAUV,
  HAUV) and reports up to ~250,000 simulated steps/second on a single
  RTX 3060 via GPU-parallel rollouts.
* **Only five task environments are publicly verified right now: Hover,
  Circle Tracking, Helical Tracking, Lemniscate Tracking, and Landing.**
  The maintainers' own README states, verbatim: *"Additional
  environments, including vision-based and sonar-based tasks, are under
  development."* There is no sonar sensor, no BlueROV2-sonar task, and no
  active-SLAM task in the public release as of this writing.
* This means: **there is no `BlueROV2SonarTask` you can `import` and use
  today.** Getting sonar-based active SLAM running on MarineGym means
  *authoring* that task yourself, in the same place and following the
  same conventions MarineGym's own maintainers use for Hover/Track/
  Landing — which is exactly the kind of thing your own prior sonar work
  (a range-sensor wrapper plus a GPU-parallel Warp-based version) is for.

Everything in Section 1 below (installing MarineGym itself) is verified
against their real documentation. Everything in Section 2 (writing your
sonar task and wiring it into this repo) is architecture and a working
code skeleton — clearly marked wherever it depends on an exact class name
or method signature that only exists in your local clone, not in anything
published, so I could not verify it independently. Search for
`VERIFY AGAINST YOUR CLONE` in `env/marinegym_env.py` for every one of
those spots.

---

## 1. Installing MarineGym (verified)

### System requirements

* NVIDIA RTX 20–40 series GPU (RTX 2060 can run 2048 parallel envs; at
  least 8 GB VRAM recommended).
* 16 GB system RAM minimum (32 GB recommended), 50 GB free disk.
* Ubuntu 20.04 or 22.04 LTS, NVIDIA driver >= 535.
* Conda/Miniconda, Git, standard build tools, sudo access.

### Install steps

```bash
# System prep
sudo apt update
sudo apt install -y screen wget unzip git cmake build-essential

# Isaac Sim 4.1.0 standalone build
cd ~
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.1.0-rc.7%2B4.1.14801.71533b68.gl.linux-x86_64.release.zip
unzip isaac-sim-standalone@4.1.0-rc.7+4.1.14801.71533b68.gl.linux-x86_64.release.zip -d isaac410
echo 'export ISAACSIM_PATH="$HOME/isaac410"' >> ~/.bashrc
source ~/.bashrc

# Python environment (Isaac Sim 4.1.0 requires Python 3.10)
conda create -n sim python=3.10 -y
conda activate sim

# Clone MarineGym and IsaacLab
cd ~
git clone https://github.com/Marine-RL/MarineGym.git
git clone https://github.com/Marine-RL/IsaacLab.git
cd ~/IsaacLab
ln -s ../isaac410 _isaac_sim

# Conda environment overlays shipped with MarineGym
cp -r ~/MarineGym/conda_setup/etc "$CONDA_PREFIX"
conda deactivate
conda activate sim

# Core dependencies at tested versions
pip install --upgrade pip setuptools wheel
pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash

# IsaacLab
cd ~/IsaacLab
./isaaclab.sh --install

# MarineGym itself, editable
cd ~/MarineGym
pip install -e .
```

If external connectivity is limited, MarineGym's own docs suggest a
mirror: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`.

### Validate the install

```bash
echo "$ISAACSIM_PATH"   # expect: /home/<user>/isaac410
python -c "import marinegym; print('MarineGym installed successfully')"

conda activate sim
cd ~/MarineGym
python train.py task=HoverRand algo=ppo headless=true enable_livestream=false \
    mode=train task.drone_model.name=iAUV task.env.num_envs=4 total_frames=5000 \
    wandb.mode=offline
```

If that trains for a few thousand frames without erroring, the whole
Isaac Sim -> IsaacLab -> MarineGym chain is working.

### Everyday commands (for reference -- these are the ones MarineGym itself supports today)

```bash
python train.py task=Hover algo=ppo headless=false enable_livestream=false
python train.py task=Track algo=ppo headless=false enable_livestream=false
python train.py task=Landing algo=ppo headless=false enable_livestream=false

# Evaluate a checkpoint
python train.py task=Hover algo=ppo mode=evaluate headless=true
# Resume
python train.py task=Hover algo=ppo resume_path=./checkpoints/hover_latest.pt

# High-throughput benchmark (their headline number)
python train.py task=HoverRand algo=ppo headless=true enable_livestream=false \
    mode=train task.drone_model.name=iAUV task.env.num_envs=4096 total_frames=50000000
```

`task.drone_model.name` accepts `BlueROV`, `BlueROVHeavy`, `iAUV`, `LAUV`,
or `HAUV`. Config files live under `~/MarineGym/cfg/` -- that's where
actuator, sensor, and disturbance settings are defined per task, and
where a new sonar task's config would live too.

---

## 2. Authoring the sonar task and wiring it into this repo

### 2.1 Find the real Task API in your clone (do this before writing any code)

Since sonar tasks aren't public, there's no source for me to mirror
exactly. Before writing `_build_marinegym_task()` in
`env/marinegym_env.py`, open these two files in your `~/MarineGym` clone
and read them:

```
~/MarineGym/scripts/train.py        # traces how `task=Hover` becomes a live Python object
~/MarineGym/marinegym/              # the package itself -- find the Hover task's class definition
```

MarineGym is built on OmniDrones, whose tasks follow a predictable shape
(a TorchRL/tensordict `EnvBase`-style class with methods like
`_reset_idx`, `_pre_sim_step`, `_compute_state_and_obs`,
`_compute_reward_and_done`, working over `TensorSpec`/`CompositeSpec`
observation and action specs) -- expect something in that family, but
confirm the exact names before relying on them.

### 2.2 What you're actually authoring

A new task, most naturally `marinegym/tasks/bluerov2_sonar.py` (or
wherever the existing Hover/Track/Landing tasks live in your clone) plus
a matching config file under `cfg/`, registered the same way theirs are.
This task needs to:

1. Load a confined-space scene (tunnel/wreck USD asset) instead of
   Hover/Track's open-water scene.
2. Attach your existing sonar sensor work (the Isaac-Sim range-sensor
   wrapper and the Warp-based GPU-parallel version) to the BlueROVHeavy
   articulation.
3. Expose the vehicle's true pose and a collision/contact flag per step
   (needed by `MarineGymWorldAdapter`/`_apply_action` below).

This is real engineering work, not a config toggle -- which is exactly
why the maintainers list it as "under development" rather than shipped.

### 2.3 The adapter that plugs it into this repo

`env/marinegym_env.py`'s `MarineGymActiveSlamEnv` subclasses
`ActiveSlamEnv` and overrides exactly three things:

* `_reset_internal_state` -- builds a `MarineGymWorldAdapter` (occupancy/
  collision queries) and a `MarineGymSonarAdapter` (wraps your sonar
  module) that expose the same `.is_free()`/`.occ` and
  `.sense_imaging()`/`.sense_scanning_360()` interfaces
  `TunnelWorld`/`SonarModel` already do.
* `_apply_action` -- sends real thruster commands into the task and steps
  physics, instead of the kinematic integrator `world_generator.py` uses.
* `close()` -- shuts down the Isaac Sim app.

Everything else -- `registration/fs2d.py`, `mapping/volumetric_map.py`,
`mapping/change_detection.py`, `perception/loop_closure.py`,
`env/reward.py`, `state/encoder.py`, `rl/train.py`, `metrics/`,
`visualization/` -- is **inherited completely unmodified**, because
`ActiveSlamEnv.step()` only ever talks to `self.world` and `self.sonar`
through those four methods. That's the whole point of the adapter
pattern here: your entire tested pipeline (16 passing tests, the reward-
hacking fix, the BC bootstrap) carries over without a single change.

Every place in `env/marinegym_env.py` marked `# VERIFY AGAINST YOUR
CLONE` is exactly the boundary between "verified, real code" and
"structurally correct but needs your clone's real names filled in":
task construction, reset, `apply_action`, `step_physics`, reading back
pose/collision state, and the collision-mesh voxelization for the
ground-truth occupancy grid.

### 2.4 Wiring your existing sonar module in

```python
from active_slam_rl.env.marinegym_env import MarineGymActiveSlamEnv, MarineGymTaskConfig
from active_slam_rl.env.sim_env import EnvConfig

# your_sonar_module is whatever object your prior Isaac-Sim sonar work
# produces -- the range-sensor wrapper, or the Warp-accelerated version.
# MarineGymSonarAdapter only needs it to answer "give me a frame at this
# pose"; see MarineGymSonarAdapter._to_common_format for the exact shape
# it translates into.
from your_project.sonar import your_sonar_module

mg_config = MarineGymTaskConfig(
    task_name="BlueROV2Sonar",     # the task name you registered in 2.2
    drone_model="BlueROVHeavy",
    num_envs=1,
    headless=True,
)
env = MarineGymActiveSlamEnv(mg_config, sonar_module=your_sonar_module,
                              env_config=EnvConfig())
obs, info = env.reset()
```

From here, `env` is a drop-in replacement anywhere `ActiveSlamEnv` was
used before -- `rl/train.py::train()`, `scripts/run_training.py`,
`scripts/run_eval.py`, `scripts/visualize_demo.py` all accept it
unchanged, since they only depend on the Gymnasium interface.

### 2.5 Building the ground-truth occupancy grid

`MarineGymWorldAdapter._voxelize_scene` needs one real implementation
choice from you: how to turn your loaded USD tunnel/wreck asset's
collision mesh into the same kind of 2D (or 3D, if you extend
`OccupancyGrid` per `docs/ARCHITECTURE.md` section 9) `occ` grid
`TunnelWorld` produces. Two practical approaches:

* **Dense raycast grid**: cast a regular grid of downward or in-plane
  physx raycasts across the scene's bounding box (Isaac Sim's physx
  scene-query API supports this) and mark a hit as occupied. Simple,
  works for any asset, one-time cost per scene load -- cache the result.
* **Offline mesh voxelization**: if you already have the USD asset's
  mesh outside Isaac Sim, voxelize it once with a library like
  `trimesh` and load the cached grid at scene-load time instead of
  querying Isaac Sim at all.

Either way, this only needs to run once per episode (or once per unique
scene, if scenes are reused across episodes), not per sonar beam.

---

## 3. Single-environment wrapper vs. full batched GPU integration

`MarineGymActiveSlamEnv` as written wraps **one** MarineGym task
instance (`num_envs=1`) in the ordinary Gymnasium interface. This is
deliberately the first milestone:

* **What you get**: MarineGym's real physics fidelity and your real
  sonar sensor model, replacing the lightweight `TunnelWorld`/
  `SonarModel`, while every other tested module (FS2D, mapping, change
  detection, loop closure, reward, encoder, PPO, BC, metrics, plotting)
  keeps working with zero changes.
* **What you don't get yet**: MarineGym's headline GPU-parallel
  throughput (thousands of environments stepping in lockstep). SB3's
  `SubprocVecEnv` (what `rl/train.py::build_vec_env` uses) parallelizes
  across CPU processes, one environment per process -- it is not the
  same kind of parallelism as MarineGym's own batched-tensor
  `num_envs=4096`-style rollouts, and the two don't compose for free.

**Getting the GPU-parallel throughput** means a second, larger phase of
work: porting `OccupancyGrid`, `compute_innovation`/`compute_change_mask`,
and `compute_reward` from per-environment NumPy loops to batched PyTorch
tensor operations with a leading `num_envs` dimension (e.g. `OccupancyGrid`
becomes a `(num_envs, H, W)` tensor with the same log-odds update applied
via vectorized indexing instead of Python loops over beams), then
training with MarineGym's own TorchRL-based PPO (`algo=ppo` in
`train.py`) instead of SB3, since SB3 isn't built for tens-of-thousands-
of-environments-in-lockstep training. This is a substantial rewrite in
its own right -- get the single-environment version in Section 2 working
and validated first (you can directly compare its behavior against the
existing `ActiveSlamEnv` results in this repo, since the reward/metrics
math is identical), then treat batching as a distinct follow-on project.

---

## 4. What I could not verify from here

I don't have access to a MarineGym installation, Isaac Sim, or a GPU in
this environment, so nothing in Section 2's code has been executed -- it
has been checked for internal consistency (it imports cleanly, correctly
subclasses `ActiveSlamEnv`, and doesn't break the existing test suite),
but not run against the real simulator. Before trusting it for real
training, at minimum:

* Confirm every `# VERIFY AGAINST YOUR CLONE` line against your actual
  `~/MarineGym` source.
* Run `MarineGymActiveSlamEnv` for a handful of manual `env.step()` calls
  with `headless=False` and visually confirm the vehicle actually moves
  the way each action expects, before pointing PPO at it.
* Sanity-check `MarineGymWorldAdapter.is_free()` against a few known
  points in your loaded scene (somewhere you know is open water,
  somewhere you know is inside a wall) before trusting the completeness
  metric.
