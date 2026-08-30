# active-slam-rl

Reinforcement-learning-driven active SLAM for confined underwater
environments: sonar sensing → FS2D registration → Bayesian mapping →
loop closure → optional Structure-from-Motion → a jointly-trained state
encoder → a PPO policy.

Runs today, no special hardware/software needed — a pure-Python/numpy
reference implementation, no native C++ library or external simulator
required.

## Install

**Docker:**
```bash
docker build -t active-slam-rl .
docker compose run --rm active-slam-rl pytest tests/ -v
```

**Without Docker:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest tests/ -v      # expect: 64 passed
```

(Prefix every command below with `docker compose run --rm active-slam-rl`
if you're using Docker instead of a local venv.)

## Train

```bash
# Small/fast config, good for iterating
python scripts/run_training.py --config configs/quick_demo.yaml

# Same, but with Structure-from-Motion (perception/sfm2d.py) turned on
python scripts/run_training.py --config configs/quick_demo_sfm2d.yaml

# Full-scale config for a real training run
python scripts/run_training.py --config configs/default.yaml

# Bootstrap the policy from an expert heuristic before PPO fine-tuning
python scripts/run_bc_then_ppo.py --config configs/quick_demo.yaml
```

**Always use the same config for a given model** across training, eval,
and visualization — the world/sonar/patch dimensions are baked into the
saved model's shape.

To try a specific SfM setting instead of the two ready-made configs,
copy `configs/quick_demo_sfm2d.yaml` and edit:
```yaml
env:
  use_sfm2d: true
  sfm2d_apply_correction_from: "both"   # "imaging" | "scanning" | "both" | "off"
```

Once you have a trained model:
```bash
# Evaluate against the heuristic baselines
python scripts/run_eval.py --config configs/quick_demo.yaml \
    --model results/ppo_active_slam.zip --episodes 10 --stochastic

# Watch it work -- renders an animated GIF
python scripts/visualize_demo.py --config configs/quick_demo.yaml \
    --model results/ppo_active_slam.zip --stochastic
```

The `--stochastic` flag matters for an undertrained policy: greedy
(argmax) action selection can get stuck in a degenerate fixed point
(e.g. spinning in place) even when the policy behaves reasonably when
sampled. Use it until training has run long enough for a confidently
peaked action distribution.

## Ablations

`configs/ablations/` has a standard active-SLAM ablation set: `full.yaml`
(everything on -- loop closure, sensor fusion, SfM2D, both sonar
modalities) and six others, each identical to `full.yaml` except one
factor (see each file's own `# ABLATION:` header comment for exactly
which one -- `no_loop_closure`, `no_sensor_fusion`, `no_sfm2d`,
`imaging_only`, `scanning_only`, `fixed_exploration`).

```bash
# Everything in configs/ablations/, 3 seeds each -- a fast smoke-scale
# study (quick_demo.yaml scale) good for confirming the pipeline and
# getting a first read on directionality
python scripts/run_ablations.py

# A real, paper-scale study: more seeds, longer training
python scripts/run_ablations.py --seeds 5 --timesteps 100000 --episodes 10

# Just a couple of variants while iterating
python scripts/run_ablations.py --configs full no_loop_closure
```

Writes `results/ablations/ablation_results.csv` (mean ± std per metric
across seeds) and `results/ablations/ablation_comparison.png` (grouped
bar chart, one panel per headline metric, `full` highlighted as the
reference) — the same format used in this area's papers (e.g. Chaplot
et al.'s Active Neural SLAM). Add a new ablation by dropping another
single-factor-changed YAML file into `configs/ablations/`; no code
changes needed.

## Output: results, plots, graphs

Training writes to `results/` as it runs:

| File | What it is |
|---|---|
| `results/ppo_active_slam.zip` | the trained model |
| `results/train_logs/episode_metrics.csv` | per-episode metrics (reward, ATE, map completeness, collisions, loop closures, and -- if `use_sfm_fusion`/`use_sfm2d` are on -- per-modality registration quality and fusion-gating stats) |
| `results/train_logs/training_curves.png` | live-updating reward / map completeness / ATE / collisions+loop-closures, refreshed automatically during training |
| `results/train_logs/training_diagnostics_curves.png` | registration quality by sonar modality + fusion gating over training |

`scripts/run_eval.py --out_dir results/eval` (default) writes:

| File | What it is |
|---|---|
| `results/eval/dashboard_<baseline>.png` | per-baseline episode dashboard: trajectory overlay, ATE/completeness/reward/entropy traces |
| `results/eval/baseline_comparison.png` | bar-chart comparison across baselines |

`scripts/visualize_demo.py` writes an animated rollout GIF to
`results/eval/rollout_demo.gif`.

`scripts/run_ablations.py` (see "Ablations" above) writes:

| File | What it is |
|---|---|
| `results/ablations/<name>/seed_<k>/...` | per-ablation, per-seed model + training logs, same layout as a normal training run |
| `results/ablations/ablation_results.csv` | mean ± std per metric, one row per ablation, across seeds |
| `results/ablations/ablation_comparison.png` | grouped bar chart across ablations, `full` highlighted as reference |

Two more plots exist in `active_slam_rl.metrics.plotting` for ad-hoc use
(not wired into a script by default): `plot_registration_and_fusion_diagnostics`
(per-episode registration/fusion detail) and `plot_sfm2d_landmark_maps`
(the SfM landmark maps over the true occupancy grid, imaging vs.
scanning sonar side by side).

## Troubleshooting

- **Tests fail right after install** — check `pip list` shows `torch`,
  `gymnasium`, `stable-baselines3`; re-run `pip install -r requirements.txt`
  if not (safe to re-run).
- **Shape-mismatch error loading a model** — you trained and evaluated
  with different `--config` files. Use the same one for both.
- **Policy looks stuck / not moving in eval or the GIF** — add
  `--stochastic` (see above).
