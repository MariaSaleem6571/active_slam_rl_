"""
IMU + DVL sensing front-end -- the two inertial/velocity sensors the
thesis's SfM ("State fusion Module") block fuses with FS2D (see
docs/ARCHITECTURE.md section 8, feedback loop B, and fusion/sfm.py for
the fusion algorithm itself). This file only plays the same role
`sonar_model.py` plays for acoustic sensing: turn the vehicle's *true*
motion into a *noisy, realistic sensor reading* that downstream code
never gets to see the true value of.

WHY THESE TWO SENSORS, CONCRETELY
-----------------------------------
Real AUV navigation stacks (this is standard, not specific to this
thesis) dead-reckon between fixes using exactly this pair:

  * IMU (gyroscope)  -- reports rotation. Cheap, always available, but
    has a slowly-drifting bias: even a perfectly stationary gyro reports
    a small nonzero rate, and that number itself wanders over time.
    Integrated blindly, a biased gyro produces a heading error that
    grows without bound.
  * DVL (Doppler Velocity Log) -- reports body-frame velocity by
    bouncing acoustic pulses off a solid reflecting surface (usually the
    seafloor; in this confined-tunnel setting, a nearby wall) and
    measuring the Doppler shift. Precise per-reading, but *fails
    outright* ("loses bottom/wall lock") when there's no solid surface
    within range to reflect off -- exactly the kind of thing that
    happens in the wide-open or oddly-shaped stretches of a tunnel this
    thesis's world generator produces.

Fusing either one alone with FS2D would already help; fusing both is
what a real system does, and it's what lets the fusion algorithm in
`fusion/sfm.py` demonstrate something interesting: covariance shrinking
when sensors agree, degrading gracefully (not catastrophically) when one
of them (DVL) drops out.

MODELING SIMPLIFICATION: PER-STEP, NOT PER-SECOND
----------------------------------------------------
Real gyros report angular *rate* (deg/s) and DVLs report *velocity*
(m/s); you then integrate over your sample period to get a delta. This
codebase has no notion of real time anywhere else -- `process_noise_xy`/
`process_noise_theta_deg` on `EnvConfig` are already *per-step* standard
deviations, not per-second ones -- so, consistently, the two models below
report an already-integrated *delta for this one step* directly (a delta
heading, a delta position in the body frame) rather than a rate you'd
then have to integrate yourself. Physically this is "the sensor's own
internal mechanization already integrated the raw rate/velocity for you
before handing you a reading," which is exactly what real inertial
navigation units do internally anyway -- so nothing physically dishonest
is happening here, just a deliberate choice of what unit to stop at,
matching the rest of the project's abstraction level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IMUConfig:
    """See the module docstring for why this reports a per-step delta
    heading rather than a rate."""

    # White (per-step) measurement noise on the delta-heading reading.
    gyro_noise_std_deg: float = 1.0
    # The gyro's *true* bias is sampled once per episode (a fresh
    # simulated sensor unit each "deployment") from this distribution --
    # not the filter's belief about the bias, the actual physical value
    # the filter has to discover. See fusion/sfm.py's bias tracking.
    gyro_bias_init_std_deg: float = 1.5
    # That true bias itself isn't perfectly constant either -- real gyro
    # bias wanders slowly with temperature/time. Modeled as a small
    # random walk added to the true bias every step.
    gyro_bias_walk_std_deg: float = 0.03


@dataclass
class DVLConfig:
    """Body-frame displacement-per-step, with occasional dropout."""

    # White (per-step) noise on each body-frame axis (forward, lateral),
    # in world units (the same units as the map grid / true_pose).
    vel_noise_std: float = 0.04
    # Baseline per-step chance of losing bottom/wall lock even in normal
    # conditions (real DVLs aren't perfectly reliable either).
    dropout_prob_base: float = 0.01
    # Elevated dropout chance once the nearest obstacle return is closer
    # than `near_wall_range` -- modeling reduced reliability in tight
    # confines, which is exactly the operating regime this thesis
    # targets. (A real DVL's failure mode is closer to "no reflecting
    # surface in range" than "too close a surface"; tying dropout to
    # *proximity* rather than *distance-to-target-range* is a documented
    # simplification, chosen because this 2D sim's only ready proxy for
    # "how confined is it right now" is the sonar's own min beam range.)
    dropout_prob_near_wall: float = 0.3
    near_wall_range: float = 1.5


def true_relative_motion(prev_pose: tuple, curr_pose: tuple) -> tuple:
    """Decompose the *true* motion between two consecutive true poses into
    (dtheta, body_disp), in exactly the frame convention `sim_env.py`'s
    `_integrate_odometry` already uses for FS2D's own (dx, dy, dtheta):
    body_disp = (forward, lateral) such that rotating it by the pose's
    heading *at the start of the step* recovers the world-frame
    displacement. IMU/DVL readings are built from this ground truth by
    the two models below, the same way `SonarModel` builds noisy ranges
    from `world.is_free`.

    Note on the action space's action 6 ("revisit"): that action rotates
    and translates within the same discrete step, using the *new*
    heading for the translation. This function still reports one combined
    (dtheta, body_disp) pair for the whole step, computed using the
    heading at the *start* of the step -- consistent with how every other
    per-step quantity in this simulator (including FS2D's own dx/dy) is
    already a single atomic delta with no sub-stepping, at the cost of
    that one action type not decomposing into a perfectly faithful
    sub-step kinematic account. Noted here rather than silently accepted.
    """
    py, px, pth = prev_pose
    cy, cx, cth = curr_pose
    dtheta = _wrap_angle(cth - pth)
    world_dx, world_dy = cx - px, cy - py
    c, s = np.cos(pth), np.sin(pth)
    forward = c * world_dx + s * world_dy
    lateral = -s * world_dx + c * world_dy
    return dtheta, np.array([forward, lateral], dtype=float)


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class IMUModel:
    """Single-axis (yaw) gyroscope simulation. Owns the *true* bias for
    the current episode -- the thing `fusion.sfm.StateFusionModule` has
    to estimate without ever being told it directly."""

    def __init__(self, cfg: IMUConfig = IMUConfig(), rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self.true_bias = 0.0  # radians; set for real in reset()

    def reset(self) -> None:
        """Sample a fresh true bias for a new episode (a new simulated
        deployment gets a gyro with its own, unknown-to-the-filter,
        bias)."""
        self.true_bias = np.deg2rad(self.rng.normal(0.0, self.cfg.gyro_bias_init_std_deg))

    def sense(self, true_dtheta: float) -> float:
        """Return a noisy, biased measurement of this step's delta
        heading, given the *true* delta heading."""
        self.true_bias += np.deg2rad(self.rng.normal(0.0, self.cfg.gyro_bias_walk_std_deg))
        noise = np.deg2rad(self.rng.normal(0.0, self.cfg.gyro_noise_std_deg))
        return float(true_dtheta + self.true_bias + noise)


class DVLModel:
    """Body-frame displacement-per-step sensor with proximity-dependent
    dropout. `sense()` returns `None` on a dropout step -- callers (see
    `fusion.sfm.StateFusionModule.step`) must handle a missing DVL
    reading by falling back to FS2D + IMU alone, not treat `None` as
    zero."""

    def __init__(self, cfg: DVLConfig = DVLConfig(), rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    def sense(self, true_disp_body: np.ndarray, min_obstacle_range: float) -> Optional[np.ndarray]:
        cfg = self.cfg
        dropout_prob = cfg.dropout_prob_near_wall if min_obstacle_range < cfg.near_wall_range \
            else cfg.dropout_prob_base
        if self.rng.uniform() < dropout_prob:
            return None
        noise = self.rng.normal(0.0, cfg.vel_noise_std, size=2)
        return np.asarray(true_disp_body, dtype=float) + noise
