# Architecture & Concepts — a full walkthrough

This document explains every concept in the codebase from the ground up,
in the order data actually flows through one simulation step. Read it
alongside `env/sim_env.py::step()`, which is the literal orchestration of
everything described here.

---

## 1. Why this is called "active" SLAM

Classical SLAM is *passive*: a vehicle follows some externally given path
(a human pilot, a coverage pattern, a pipeline survey line) and SLAM just
tries to keep up, fusing whatever sensor data arrives into a consistent
map and trajectory. **Active SLAM** flips this: the *next action* is
chosen specifically to improve the map and localization, not just to go
somewhere. That's why the policy's reward includes terms for *reducing
uncertainty* and *closing loops*, not just "cover more area."

This matters especially underwater in confined spaces because acoustic
sensing is inherently noisy and ambiguous (speckle, multipath, limited
FOV) — a passive path will accumulate drift you can't correct unless the
vehicle deliberately revisits places it's already been.

## 2. FS2D: Fourier-based sonar registration

**Problem**: given two sonar frames (images) taken moments apart, how far
did the vehicle move and turn between them?

**Why Fourier-based**: ordinary image registration (e.g. matching
keypoints/features) struggles on sonar images because they're full of
*speckle* — grainy multiplicative noise inherent to coherent imaging (the
same physics that makes laser speckle and medical ultrasound grainy).
Phase correlation in the Fourier domain is far more robust to this,
because it works on the *phase* of the cross-power spectrum, which is
much less affected by multiplicative amplitude noise than pixel-level
features are.

**How it works** (`registration/fs2d.py::FourierMellinRegistration`):
1. **Rotation first.** Take the magnitude of each frame's 2D Fourier
   transform (this magnitude is *shift-invariant* — moving the vehicle
   doesn't change it, only rotating does). Resample that magnitude image
   into **log-polar coordinates**: this is the "Mellin" part of
   Fourier-Mellin — in log-polar space, a *rotation* in the original image
   becomes a simple *shift* along the angle axis, and a *scale* change
   becomes a shift along the log-radius axis. Phase-correlating the two
   log-polar images finds that shift directly, giving you the rotation.
2. **De-rotate** one frame by the estimated angle.
3. **Translation.** Ordinary 2D phase correlation between the (now
   co-rotated) frames finds the translation directly, with sub-pixel
   refinement via parabolic interpolation around the correlation peak.
4. **Confidence (q_t).** A sharp, well-defined correlation peak means the
   two scans genuinely agree on their overlap; a flat, ambiguous peak
   means the match is untrustworthy (e.g. mostly open water with little
   structure to register against). This peak sharpness *is* q_t.
5. **Covariance (Sigma_reg_t).** A cheap approximation: uncertainty shrinks
   as q_t grows. The real FS2D library would fit the local curvature of
   the correlation surface for a proper estimate; the interface is
   identical either way (see section 4 below).

## 3. The SfM (State fusion Module): fusing FS2D with IMU/DVL

**Problem**: FS2D (section 2) gives one estimate of how far the vehicle
moved each step. A real vehicle also carries an IMU (gyroscope) and a DVL
(Doppler Velocity Log, giving body-frame velocity by bouncing acoustic
pulses off a solid surface) — a second, independent estimate of the same
motion. Throwing that away and trusting FS2D alone leaves real accuracy
on the table, and gives up the one thing a gyro's bias needs to be
correctable at all: an independent reference to compare it against.

**How it works** (`fusion/sfm.py::StateFusionModule`,
`env/imu_dvl_model.py::IMUModel`/`DVLModel`):
1. **Sensing.** `IMUModel` reports a noisy, biased per-step delta-heading;
   `DVLModel` reports a noisy per-step body-frame displacement, with
   proximity-triggered dropout (a DVL loses "lock" when there's no solid
   surface in range — modeled here as more likely the closer the nearest
   wall is, fitting for a tunnel-crawling vehicle).
2. **Predict.** Bias-correct the gyro reading using the *current* bias
   estimate, and combine it with the DVL reading (or, on a dropout, a
   zero-mean estimate with deliberately inflated uncertainty) into one
   predicted `(dx, dy, dtheta)` for the step.
3. **Gate, then fuse.** Before trusting FS2D's own `(dx, dy, dtheta)`
   measurement, check it's *statistically consistent* with the IMU/DVL
   prediction (a chi-squared/Mahalanobis test — "Normalized Innovation
   Squared" gating, standard practice in robust Kalman/EKF
   implementations). If it passes, fuse the two independent estimates
   with the ordinary formula for combining two Gaussian estimates of the
   same quantity (`S = P_pred + R_meas`, `K = P_pred @ inv(S)`, etc. —
   algebraically identical to a Kalman update). If it fails, fall back to
   the IMU/DVL prediction alone for that step, exactly like a DVL
   dropout.
4. **Track the bias.** Whenever FS2D passes the gate, the gap between its
   rotation estimate and the raw (still-biased) gyro reading is a direct,
   textbook-derivable measurement of the bias itself (see the long
   derivation in `_update_bias`'s docstring) — a small, separate,
   persistent scalar EKF that gets better at correcting the gyro over the
   course of an episode.

**Why the gate exists — a concrete, empirically-found failure mode**:
`registration/fs2d.py`'s numpy backend documents its own known
limitation — its real-valued Fourier-Mellin rotation estimate is only
recoverable mod pi, disambiguated by picking whichever candidate gives a
sharper translation-correlation peak. That heuristic can pick wrong, and
empirically does so *often* on this world generator's actual sonar
frames: a spot-check across ~200 registrations found FS2D's own
confidence badly overoptimistic (median position error ~4x its
self-reported standard deviation) and its rotation estimate off by more
than 90 degrees on **nearly half** of all readings. Without gating, a
single confidently-wrong ~180 degree misfire is enough to drag the
tracked bias from a plausible ~1 degree to 50-60 degrees within a few
dozen steps — every individual Kalman update along the way is
algebraically correct given what it's told, which is exactly why
defending against a *confidently wrong* measurement needs an explicit
consistency check, not better bookkeeping. **This points at a
pre-existing accuracy gap in `registration/fs2d.py`'s numpy fallback on
realistic sonar-like frames (as opposed to the clean synthetic images
`tests/test_registration.py` checks it against) that's well worth
investigating on its own** — it affects loop-closure quality and reward
shaping too, independent of this fusion module, which only ever
*defends against* the symptom rather than fixing the underlying cause.

**Toggle**: `EnvConfig.use_sfm_fusion` (default `True`). Setting it
`False` reproduces this codebase's pre-SfM behavior exactly, byte for
byte — including the exact `self.rng` draw sequence, since IMU/DVL
sensing draws its own random numbers from that same shared generator
(see `tests/test_sfm_fusion.py::test_env_disabled_fusion_matches_pre_sfm_behavior_exactly`).
Note this means enabling fusion (the default) *does* shift the RNG-driven
trajectory for any fixed seed relative to pre-SfM runs, for the same
reason adding or removing any random draw anywhere in the step loop
would — see the comment on `test_env_resets_beta_urgency_clock_on_validated_closure`
in `tests/test_reward.py` for a worked example of what that means for a
seeded test, and how it was diagnosed and fixed.

**A deliberate simplification worth knowing about**: the gyro bias is
tracked as its own small, *decoupled* filter (point 4 above) rather than
folded into one larger joint `[x, y, theta, bias]` EKF that would also
capture the cross-covariance between them. Simpler to implement and test
in isolation, at the honest cost of ignoring that (typically small)
cross-covariance — the same kind of proportionate, explicitly-flagged
trade-off as the rest of this section.

## 4. Bayesian occupancy mapping

**Problem**: fuse many noisy range readings into a single, consistent
belief about which cells are wall/rock and which are open water.

Each cell (voxel) `v` in the grid holds a belief `p(o_v) in [0, 1]` — the
probability that cell is occupied. Every time a sonar beam either (a)
passes freely through a cell, or (b) returns an echo from a cell, that's
*evidence* that nudges the belief up or down. Doing this multiplicatively
in probability space is numerically unstable (probabilities near 0 or 1
lose precision), so it's done additively in **log-odds space**:

```
l = log( p / (1-p) )         # logit
p_new = sigmoid( l_old + evidence )
```

This is exactly what `mapping/volumetric_map.py::OccupancyGrid` does:
`l_occ` is the log-odds increment for an "occupied" hit, `l_free` for a
"free" pass-through, clamped by `l_max` so the map never becomes
*overconfident* (a single bad reading shouldn't be irreversible).

**Map entropy** `H(M_t)` (used in the reward) is the sum, over every cell,
of the Bernoulli entropy `-p log p - (1-p) log(1-p)`. A cell at `p=0.5`
(totally unknown) has maximum entropy; a cell confidently resolved to
`p≈0` or `p≈1` has near-zero entropy. So *entropy reduction* is a direct,
principled measure of "how much did this action actually teach us,"
independent of whether what we learned was "wall" or "open water" — this
is why it's used as the reward's coverage term instead of a naive "new
cells visited" counter.

## 5. Change detection: telling real drift/change from noise

Sonar is noisy, so a cell's belief will jitter slightly step to step even
with nothing physically different. The question change detection answers
is: *is this cell's belief changing more than noise alone would explain?*

```
eta_t^v = | p_t(v) - p_{t-1}(v) | / sqrt(Var[p_t(v)] + eps)
```

This is a **normalized innovation** — dividing by the local uncertainty
means a big jump in a cell that's already uncertain isn't flagged (that's
expected noise), but the same-sized jump in a cell that seemed *settled*
is flagged as a real change. Cells that clear a threshold get grouped into
connected components (`change_cluster_stats`) — think of this as "here's a
region where the map just became inconsistent, go look at it again,"
which is exactly what a human SLAM operator would do when they notice two
overlapping scans disagree.

## 6. Loop closure & place recognition

**Problem**: dead-reckoning (integrating FS2D's frame-to-frame odometry
over time) *always* drifts — small errors compound. The only way to
correct this without an external position reference (no GPS underwater)
is to recognize when you've come back to somewhere you've already mapped,
and use that recognition to correct the accumulated error.

`perception/loop_closure.py` implements this in two parts:
1. **Descriptor**: each visited pose gets a compact, *rotation-invariant*
   summary of its sonar frame — here, a radial intensity histogram (bin
   pixels by distance from the sensor, regardless of angle). Rotation
   invariance matters because you might revisit a place facing a totally
   different direction than when you first saw it.
2. **Matching**: at every step, compare the current descriptor against
   every stored keyframe. The best cosine-similarity score *is* the
   loop-closure saliency `ell_t` fed into the state. If it clears a
   threshold, that's a *candidate* loop closure, which then gets a real
   FS2D registration against the stored keyframe's scan to produce an
   actual pose constraint.
3. **Correction**: in a full SLAM system this constraint would go into a
   pose-graph optimizer that redistributes the correction across the
   whole trajectory. `sim_env.py` implements a simplified version: it
   directly pulls the current position estimate toward what the loop
   closure implies, weighted by how confident the match was
   (`lc_reg.quality`). This is a deliberate simplification — see
   `docs/ARCHITECTURE.md`'s "Known simplifications" section below.

## 7. The state encoder E_phi

The policy can't see raw sonar or the whole map — that's too
high-dimensional and would need a different network for every world size.
Instead, `state/encoder.py` builds a fixed-size latent vector `s_t` from:
* a small **local crop** of the map (belief, uncertainty, and change-mask
  channels) around the vehicle — a 3D convolutional-style stack (2D here,
  since the world is a 2D plan view) extracts spatial features from this,
* a handful of **scalars** — registration quality `q_t`, loop-closure
  saliency `ell_t`, accumulated pose uncertainty, and mission constraints
  (battery/time remaining) — fused through a small MLP.

Critically, this encoder is **trained jointly with the policy** (PPO's
gradients flow all the way back through it — this is "feedback loop G" in
the thesis). That means the network *learns what to pay attention to* in
the map/uncertainty crop, rather than a human hand-designing those
features. `SB3StateEncoderExtractor` is the glue that lets
Stable-Baselines3 use this as a drop-in "features extractor" for its
`MultiInputPolicy`.

## 8. The RL policy (PPO) and reward

**Why PPO**: it's a standard, robust on-policy algorithm that handles
discrete action spaces (our 7 motion primitives) well, is comparatively
easy to tune, and doesn't require a replay buffer — all useful properties
for research code you'll iterate on a lot.

**Reward** (`env/reward.py`):
```
r_t = w_cov * ΔH_t + w_cons * ΔI_t - w_safe * c_prox,t + w_loop * b_lc,t + w_change * (resolved change voxels)
```
* `ΔH_t`: entropy *reduction* this step (see section 3) — rewards
  genuinely informative actions, not just moving.
* `ΔI_t`: information gain from a *validated* loop closure — modeled here
  as the amount of accumulated pose uncertainty (`trace_cov`) that closure
  resolved.
* `c_prox,t`: a proximity-to-obstacle penalty (0 normally, ramping toward
  1 near a wall, and 1 outright on a collision).
* `b_lc,t`: a flat bonus specifically for a *validated* loop closure
  (registration quality above threshold), to make "go find and confirm a
  loop closure" unambiguously worth it even beyond the entropy/consistency
  terms.
* the change-detection term rewards resolving previously-flagged
  inconsistent regions (feedback loop D).

These weights (`RewardWeights` in `env/reward.py`) are exactly the kind of
thing you'll want to retune once you're running longer experiments — see
the README's "honest status" section for a concrete failure mode
(dwelling in place) this dial controls.

## 9. The seven feedback loops, concretely

The thesis names these abstractly (Figure 5); here's exactly where each
one is in the code, all inside `ActiveSlamEnv.step()`:

* **A (Registration → State)**: `self._q_t = reg.quality` gets written
  into the observation's scalar vector every step.
* **B (Map → Registration prior / Map → SfM)**: the SfM half of this is
  now implemented — see section 3 (`fusion/sfm.py`) — fusing FS2D with
  IMU/DVL. The "Map → Registration prior" half (feeding the map back in
  as a prior to *improve* FS2D's own registration, as opposed to fusing
  its output downstream) is still not separately modeled; the map *does*
  feed back into the state through a different path (next bullet).
* **C (Map → Reward)**: `entropy_delta` (computed from `self.map`) feeds
  directly into `compute_reward`.
* **D (Change Detection → Reward)**: `change_voxels_resolved` feeds into
  `compute_reward`'s change term.
* **E (Change Detection → State)**: the change mask is one of the three
  channels in the observation's `patch`.
* **F (Loop Closure → State)**: `self._ell_t = ell_t` goes into the
  scalar vector every step.
* **G (Policy → Encoder)**: this isn't a line of code in `sim_env.py` at
  all — it's a structural property of `state/encoder.py` +
  `rl/train.py`: because `StateEncoder` is registered as PPO's
  `features_extractor_class`, its weights get gradient updates from the
  same backward pass as the policy/value heads. Nothing routes gradients
  manually; that's what "trained jointly" means.

## 10. Known simplifications (read this before treating results as final)

Being upfront about where this departs from a full production SLAM stack:

* **A reward-hacking exploit was found and fixed here, and it's a useful
  worked example of a general RL pitfall.** A stationary vehicle could
  keep re-matching its current scan against its own single old keyframe
  every time the loop-closure detector's recency-exclusion window passed,
  collecting the flat loop-closure bonus forever for zero risk. The fix
  (`env/sim_env.py` + `env/reward.py`) gates the bonus by (a) whether
  there's real accumulated pose uncertainty left to correct, and (b) a
  cooldown since the last payout — plus an independent, separate penalty
  for consecutive near-zero-displacement steps. `tests/test_reward.py`
  encodes this as a regression test. The general lesson: any bonus tied to
  a *detector firing* rather than to *the thing the detector is supposed
  to indicate* is a standing invitation for the policy to learn to fool
  the detector instead of doing the task — worth checking for in any new
  reward term you add.

* **2D, not 3D.** The world, map, and sonar model are 2D plan-view. Every
  algorithm (Bayesian update, entropy, change detection, Fourier-Mellin
  registration) generalizes to 3D without changing its math, but the
  actual arrays here are `(H, W)`, not `(D, H, W)`. Swapping to 3D means
  changing `OccupancyGrid`'s array shapes and `SonarModel`'s raycasting to
  cast in 3D (elevation angle too) — the update rules are unchanged.
* **SfM/IMU/DVL fusion is now implemented** (section 3,
  `fusion/sfm.py` + `env/imu_dvl_model.py`) — no longer a gap. What it
  surfaced instead: `registration/fs2d.py`'s numpy backend appears
  considerably less accurate on this world's actual generated sonar
  frames than its own self-reported confidence suggests, and its known
  rotation fold-ambiguity resolves wrong far more often than "an
  occasional edge case" (see section 3's empirical numbers). The SfM
  module defends against this (an NIS outlier gate), but doesn't fix it —
  **investigating FS2D's real accuracy on realistic sonar-like frames
  (as opposed to the clean synthetic images `test_registration.py` uses)
  is a natural, likely higher-value next step**, since it affects
  odometry, loop-closure quality, and reward shaping everywhere, not just
  the new fusion path.
* **Loop-closure correction is a heuristic pull, not a pose-graph
  optimizer.** A real backend (e.g. GTSAM, g2o) would redistribute the
  correction across the whole trajectory and covariance-consistently.
  Here, `sim_env.py` just interpolates the current estimate toward what
  the loop closure implies. This is enough to demonstrate *why* loop
  closure matters (you can see drift visibly shrink in
  `visualize_demo.py`'s GIF) but isn't a substitute for a real optimizer
  before real deployment.
* **The demo training run is short** (see README section 5) — a proof the
  pipeline runs, not a converged policy.

## 11. Extending to 3D / TSDF / MarineGym

See `env/marinegym_env.py`'s docstring and `native/fs2d/README.md` for the
concrete, mechanical steps to swap in the real simulator and registration
library respectively — both are written so that swap touches *only* the
sensing/physics layer, with registration, mapping, perception, state
encoding, RL, and reward code completely unchanged.
