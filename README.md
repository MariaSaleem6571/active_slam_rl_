# active-slam-rl

Reinforcement-learning-driven **active SLAM** for confined underwater
environments: FS2D registration → Bayesian mapping → change detection →
loop closure → a jointly-trained state encoder → a PPO policy → a
multi-objective reward, closed by seven feedback loops.

**Runs today, no special hardware/software needed.** Everything ships
with a working reference implementation — no native C++ library, no
Isaac Sim, no ROS — so you can install and train on a laptop CPU right
now. See `docs/` for how to swap in the real FS2D library or a
higher-fidelity simulator later.

## Read this first

This README only covers *running* the project. For everything else,
read the docs in `docs/`:

| Doc | What's in it |
|---|---|
| `docs/active_slam_rl_explained.pdf` | Every concept explained with physical intuition and diagrams — start here to actually understand how this works. |
| `docs/FILE_BY_FILE_GUIDE.pdf` | A guided tour of the repo, directory by directory and file by file — what's inside each one and why it exists. |
| `docs/ARCHITECTURE.md` | The same walkthrough in plain markdown, for quick reference while coding. |
| `docs/MARINEGYM_INTEGRATION.md` | Verified steps to move onto MarineGym/Isaac Sim, and its honest current limitations (sonar tasks aren't public yet). |
| `docs/STONEFISH_INTEGRATION.md` | A stronger immediate alternative — Stonefish already has real sonar working today. |

## Install

**Docker (recommended):**
```bash
docker build -t active-slam-rl .
docker compose run --rm active-slam-rl pytest tests/ -v
```

**Native:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest tests/ -v      # expect: 26 passed
```

(Prefix every command below with `docker compose run --rm active-slam-rl`
if you're using Docker instead of a local venv.)

## Run it

```bash
# Train (writes a live-updating results/training_curves.png as it goes)
python scripts/run_training.py --config configs/quick_demo.yaml

# Evaluate against the heuristic baselines
python scripts/run_eval.py --config configs/quick_demo.yaml \
    --model results/ppo_active_slam.zip --episodes 10 --stochastic

# Watch it work — renders an animated GIF
python scripts/visualize_demo.py --config configs/quick_demo.yaml \
    --model results/ppo_active_slam.zip --stochastic

# Bootstrap the policy from an expert heuristic before PPO fine-tuning
python scripts/run_bc_then_ppo.py --config configs/quick_demo.yaml
```

`configs/quick_demo.yaml` is small and fast (good for iterating);
`configs/default.yaml` is the full-scale config for a real training run.
**Always use the same config for a given model** across training, eval,
and visualization — the world/sonar/patch dimensions are baked into the
saved model's shape.

The `--stochastic` flag matters for an undertrained policy: greedy
(argmax) action selection can get stuck in a degenerate fixed point
(e.g. spinning in place) even when the policy behaves reasonably when
sampled. Use it until the policy has trained long enough to have a
confidently peaked action distribution.

## Repo layout

```
active_slam_rl/
├── configs/       world/sonar/reward/imu/dvl/sfm/training settings
├── docs/          the documents listed above
├── native/fs2d/   scaffolding to plug in the real FS2D C/C++ library later
├── src/active_slam_rl/
│   ├── registration/    FS2D motion estimation
│   ├── fusion/          SfM: fuses FS2D with IMU/DVL dead-reckoning
│   ├── mapping/         occupancy grid + change detection
│   ├── perception/      loop closure / place recognition
│   ├── state/           the CNN+MLP encoder feeding the policy
│   ├── env/              the simulated world, sonar, IMU/DVL sensing,
│   │                     reward, and Gym env (plus marinegym_env.py /
│   │                     stonefish_env.py adapters)
│   ├── rl/               PPO training, behavioral cloning, baselines
│   ├── metrics/          evaluation + plotting
│   └── visualization/    the animated rollout GIF
├── scripts/       the CLI entry points you actually run
└── tests/         pytest suite (51 tests)
```

## A known, already-fixed issue worth knowing about

An early training run found a real reward-hacking exploit: a stationary
vehicle could keep "closing a loop" against its own old keyframe forever
for a free reward. It's fixed (`env/reward.py` + `env/sim_env.py`, with a
regression test in `tests/test_reward.py`) — mentioned here because it's
a useful, real example of the kind of bug reward-shaping work runs into,
not because it's still a problem.

## SfM: fusing FS2D with IMU/DVL

Odometry now fuses FS2D's relative-pose estimate with IMU (gyro) + DVL
(Doppler Velocity Log) dead-reckoning via a small EKF
(`fusion/sfm.py`), closing the "no explicit SfM/IMU/DVL fusion" gap
`docs/ARCHITECTURE.md` used to flag. Full write-up, including the
concrete math, in `docs/ARCHITECTURE.md` section 3.

**Worth knowing before you look at results**: building this surfaced a
real accuracy gap in `registration/fs2d.py`'s numpy backend — on this
world's actual generated sonar frames (not the clean synthetic images
`test_registration.py` checks it against), its self-reported confidence
is often badly overoptimistic and its known rotation fold-ambiguity
resolves wrong on a large fraction of readings. The fusion module
defends against this (an outlier-rejection gate), it doesn't fix it — see
`docs/ARCHITECTURE.md` section 3 for the numbers and why investigating
FS2D's real accuracy is probably a higher-priority next step than
anything in this fusion module, since it affects the whole pipeline, not
just odometry.

Toggle with `EnvConfig.use_sfm_fusion` (default `True`); `False`
reproduces this codebase's pre-SfM behavior exactly, including the exact
RNG draw sequence — a clean way to compare before/after, or to reproduce
results generated before this feature existed.

## Explore/exploit cycling (beta-decay, geometry-aware)

The reward includes a time-varying factor `beta` weighting exploration
credit vs. loop-closure-seeking credit, combining two signals:

* **Time since the last validated loop closure** — decays exponentially,
  same idea as before: the longer it's been, the more incentive shifts
  toward finding a closure.
* **How much of the local area is still unexplored** (a "geometry
  floor") — computed from the same local map crop the state encoder
  already sees. In a wide, largely-unmapped area this stays high and
  keeps beta from decaying just because the clock ran out; in a narrow,
  already-mostly-mapped corridor it's low, so beta falls through to the
  time-based decay sooner. Beta is the *max* of the two signals, so
  either one alone is enough to justify staying in "explore" mode.

Beta resets to `beta_initial` the instant a closure validates, cycling
the policy between "go explore" and "go find a loop closure" without a
hand-coded mode switch. Tunable via `RewardWeights.beta_initial/
beta_decay_rate/beta_min` in `env/reward.py`; verified in
`tests/test_reward.py`, including a dedicated test for the
wide-area-keeps-exploring behavior.

**Worth being precise about**: this is a hand-designed reward-shaping
schedule, not something the RL algorithm "learns" in the neural-network
sense — it's the same category of thing as a learning-rate schedule. A
genuinely *learned* version (beta as the output of a small trainable
network head, optimized end-to-end) is a real technique but a much
bigger undertaking with real stability tradeoffs; this geometry-aware
version gets the practically important property (behavior adapts to
wide vs. narrow surroundings) without that complexity.

## Troubleshooting

- **Tests fail right after install** — check `pip list` shows `torch`,
  `gymnasium`, `stable-baselines3`; re-run `pip install -r requirements.txt`
  if not (safe to re-run).
- **Shape-mismatch error loading a model** — you trained and evaluated
  with different `--config` files. Use the same one for both.
- **Policy looks stuck / not moving in eval or the GIF** — add
  `--stochastic` (see above).
- Anything else: open `docs/active_slam_rl_explained.pdf`, section on
  the specific piece that's failing, or check `docs/ARCHITECTURE.md`.
