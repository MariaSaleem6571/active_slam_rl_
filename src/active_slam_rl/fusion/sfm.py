"""
StateFusionModule -- the thesis's "SfM" block (docs/ARCHITECTURE.md
section 8: "B (Map -> Registration prior / Map -> SfM)", and section 9's
"No explicit SfM/IMU/DVL fusion... a small EKF... a natural next step").

WHAT PROBLEM THIS SOLVES
--------------------------
Before this module existed, `sim_env.py::_integrate_odometry` fed FS2D's
relative-pose estimate directly into the running position/heading
estimate and called it done. That throws away two things a real system
would use:

  1. IMU (gyro) and DVL (velocity) give a *second, independent* estimate
     of the same per-step motion FS2D is estimating. Two independent
     noisy estimates of the same thing are always better fused than
     either alone -- that's the whole point of sensor fusion, and it's
     free accuracy FS2D-only integration was leaving on the table.
  2. FS2D has no memory: nothing about it improves over time. A gyro's
     *bias*, on the other hand, is a fixed (if unknown, if slowly
     drifting) property of the physical sensor for the whole episode --
     and FS2D, precisely because it's an *independent* measurement, is
     exactly the kind of external reference that lets you estimate that
     bias online and get progressively better dead-reckoning as the
     episode goes on.

HOW THE FUSION WORKS (per step)
----------------------------------
There are up to two independent Gaussian estimates of "how far did the
vehicle actually move this step":

  * `delta_pred` -- dead reckoning: bias-corrected IMU delta-heading,
    plus DVL's body-frame delta-position (or, on a DVL dropout, "no
    information" modeled as a zero-mean estimate with inflated
    variance -- see `dvl_dropout_disp_std`).
  * `reg_delta`  -- FS2D's own relative-pose measurement (already
    unit-converted to world units by the caller), with its own
    covariance `reg_cov`.

These are combined with the ordinary formula for fusing two independent
Gaussian estimates of the same quantity -- algebraically identical to a
single Kalman filter update step, with `delta_pred`/`P_pred` playing the
role of the prior and `reg_delta`/`reg_cov` the role of the measurement:

    S      = P_pred + R_meas
    K      = P_pred @ inv(S)
    fused  = pred + K @ (meas - pred)
    P_new  = (I - K) @ P_pred

This makes no assumption about *why* two independent estimates exist --
it doesn't care that one came from an accelerometer-and-gyro
mechanization and the other from a sonar phase correlation. That
generality is exactly why the same four lines of algebra show up
everywhere from GPS/INS fusion to this thesis's tunnel-crawling AUV.

WHY THE PER-STEP DELTA IS FUSED FRESH EACH STEP, NOT A RUNNING POSE
----------------------------------------------------------------------
A more ambitious design would maintain one continuously-propagated
[x, y, theta, bias] EKF across the whole episode, with the position/
heading *and* the bias sharing one joint covariance that captures their
cross-correlation. That is the "fully joint" version of this filter.
What's implemented here instead deliberately decouples into two
timescales:

  * a *fresh*, memoryless fusion of `delta_pred` vs. `reg_delta` every
    step (no persisted position covariance -- `ActiveSlamEnv` already
    owns the running pose estimate and its own `trace_cov` proxy for
    reward-shaping purposes; this module only ever reasons about *this
    step's* motion), and
  * a *slowly evolving*, persisted scalar EKF over the gyro bias alone,
    updated using the discrepancy between FS2D and the raw gyro reading
    whenever FS2D is available.

This is simpler to implement, reason about, and unit-test in isolation
than the joint version, at the honest cost of ignoring the (typically
small) cross-covariance between "this step's pose fusion" and "the
running bias belief." Proportionate to what ARCHITECTURE.md asked for
("a small EKF... between FS2D and the pose integrator"), not a rewrite
of the whole pose-tracking design -- and, in the same spirit as this
codebase's other documented simplifications (see ARCHITECTURE.md section
9), flagged here rather than silently glossed over.

WHY THIS MODULE ALSO GATES FS2D AS A POSSIBLE OUTLIER
---------------------------------------------------------
`registration/fs2d.py`'s own numpy backend documents a known limitation:
its real-valued Fourier-Mellin rotation estimate is only recoverable mod
pi, so it disambiguates by testing both `dtheta` and `dtheta +/- pi` and
keeping whichever gives a sharper *translation* correlation peak. That
heuristic is usually right, but it isn't guaranteed to be -- in a
repetitive or left-right-symmetric stretch of tunnel, both candidates can
give comparably sharp translation peaks, and the wrong one gets kept.
When that happens, FS2D reports a confidently-wrong ~180 degree rotation
(a high `quality` score, and correspondingly a *small*, confident
covariance) rather than a large, honestly-uncertain one.

A plain Kalman fusion has no defense against a measurement that's
*confidently* wrong -- it trusts whatever covariance it's handed, and a
tight covariance on a garbage value pulls the fused estimate (and, worse,
the persisted bias estimate) straight to that garbage. Empirically
running this exact pipeline surfaced exactly that failure: a single
~180 degree fold-ambiguity misfire was enough to drag the tracked gyro
bias from a plausible ~1 degree up toward 50-60 degrees over a few dozen
steps, even though every individual Kalman update was algebraically
correct given what it was told.

The standard fix -- and what's implemented in step 2 of `step()` below --
is Normalized Innovation Squared (NIS) gating: before trusting `reg_delta`
at all, check whether it's statistically consistent with the IMU/DVL
prediction it's about to be fused with (Mahalanobis distance against
their *combined* covariance). A ~180 degree discrepancy against a
sub-degree-scale predicted uncertainty is not a "big but plausible"
disagreement, it's off by tens of standard deviations -- so gating
rejects it outright and falls back to IMU/DVL dead-reckoning for that one
step, exactly like a DVL dropout. This is a standard, textbook technique
in robust Kalman/EKF implementations (used pervasively in real
navigation/SLAM stacks precisely to survive occasional bad measurements
without special-casing the specific sensor/failure mode), not something
invented ad hoc for this one bug.

UNITS AND ORDERING CONVENTION
--------------------------------
Every 3-vector in this module is `[dx, dy, dtheta]` -- `dx`/`dy` are a
body-frame displacement (forward, lateral) in world units, `dtheta` in
radians -- exactly matching `RegistrationResult.covariance`'s documented
ordering in `registration/fs2d.py`. This module is deliberately
unit-agnostic about *where* dx/dy/dtheta come from: `sim_env.py` is
responsible for converting FS2D's pixel-space `(dx, dy)` into world units
(via `_scan_scale`) before calling in here, exactly as it already did for
the FS2D-only path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi). An innovation like (+179 deg) - (-179 deg) is a
    genuine ~2 degree disagreement once wrapped, not the ~358 degree one
    naive subtraction would give -- both for the NIS gate below and for
    the Kalman update itself, an unwrapped theta innovation would corrupt
    exactly the near-+-180-degree-heading cases this module most needs to
    handle correctly (see the fold-ambiguity discussion above)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class SfMConfig:
    """The filter's *assumed* sensor noise levels -- i.e. what it's
    tuned to, as opposed to `IMUConfig`/`DVLConfig`'s *true* simulated
    noise. In a real system these are calibrated separately and rarely
    match the physical sensor exactly; the defaults below simply mirror
    `IMUConfig`/`DVLConfig`'s defaults, an "optimally tuned filter"
    idealization appropriate for a reference implementation. Decoupled
    on purpose from `IMUConfig`/`DVLConfig` (rather than reading them
    directly) so a deliberately *mistuned* filter is a one-line ablation
    for anyone who wants to study robustness to bad noise calibration --
    a realistic and common failure mode in deployed navigation systems.
    """

    assumed_gyro_noise_std_deg: float = 1.0
    assumed_dvl_noise_std: float = 0.04
    # "No information this step" variance substituted for DVL on a
    # dropout -- large relative to a normal reading's variance, so the
    # fusion below correctly leans almost entirely on FS2D (+ IMU) for
    # translation that step, rather than trusting a fabricated zero.
    dvl_dropout_disp_std: float = 5.0
    # The filter's *prior* uncertainty about the bias at the start of an
    # episode, and how fast that uncertainty is allowed to grow again
    # between corrections (the bias random walk's process noise, from
    # the filter's point of view).
    bias_init_std_deg: float = 1.5
    bias_process_noise_std_deg: float = 0.03
    # Normalized-Innovation-Squared outlier gate (see the module docstring
    # section "WHY THIS MODULE ALSO GATES FS2D AS A POSSIBLE OUTLIER"): a
    # FS2D reading whose Mahalanobis distance from the IMU/DVL prediction
    # exceeds this is rejected outright for that step (treated as if
    # reg_delta/reg_cov were None) rather than fused. Default is the
    # chi-square critical value for 3 degrees of freedom at p=0.999
    # (scipy.stats.chi2.ppf(0.999, df=3) ~= 16.27) -- deliberately
    # conservative (only rejects genuinely egregious disagreements, like a
    # ~180 degree fold-ambiguity misfire) rather than tuned to reject
    # ordinary noisy-but-honest registrations.
    chi2_gate_threshold: float = 16.27


@dataclass
class FusionResult:
    """Output of one `StateFusionModule.step()` call. `dx`/`dy`/`dtheta`
    are the fused per-step delta, in the same body-frame/world-unit/
    radian convention as `RegistrationResult` -- pass this straight into
    `ActiveSlamEnv._integrate_odometry_from_delta`."""

    dx: float
    dy: float
    dtheta: float
    covariance: np.ndarray            # 3x3, order (dx, dy, dtheta)
    used_fs2d: bool
    used_dvl: bool
    fs2d_rejected_outlier: bool
    bias_estimate_deg: float
    bias_std_deg: float


class StateFusionModule:
    """See the module docstring for the full derivation. Owns exactly one
    piece of persistent state across steps: the running gyro-bias
    estimate and its variance. Like `AdaptiveDecayController` /
    `LoopClosureDetector`, one instance is created per episode (see
    `ActiveSlamEnv._reset_internal_state`) -- a fresh bias belief for a
    fresh simulated deployment, not carried across episodes. (Carrying a
    learned calibration across episodes -- the way a real vehicle
    remembers its last calibration between missions -- is a reasonable
    alternative design, deliberately not implemented here to avoid
    calibration information leaking across what are otherwise
    procedurally-independent training episodes/worlds.)
    """

    def __init__(self, cfg: SfMConfig = SfMConfig()):
        self.cfg = cfg
        self.bias_estimate = 0.0   # radians
        self.bias_variance = np.deg2rad(cfg.bias_init_std_deg) ** 2

    def reset(self) -> None:
        self.bias_estimate = 0.0
        self.bias_variance = np.deg2rad(self.cfg.bias_init_std_deg) ** 2

    def step(self, imu_dtheta: float, dvl_disp: Optional[np.ndarray],
             reg_delta: Optional[np.ndarray], reg_cov: Optional[np.ndarray]) -> FusionResult:
        """One fusion step.

        Parameters
        ----------
        imu_dtheta : float
            This step's raw (still-biased) gyro delta-heading reading,
            in radians -- i.e. `IMUModel.sense(...)`'s return value.
        dvl_disp : Optional[np.ndarray], shape (2,)
            This step's DVL body-frame displacement reading in world
            units, or `None` on a dropout -- i.e. `DVLModel.sense(...)`'s
            return value.
        reg_delta : Optional[np.ndarray], shape (3,)
            FS2D's `[dx, dy, dtheta]`, already converted to world units
            (dx/dy) -- `None` if no FS2D registration ran this step
            (e.g. the episode's first step, with no previous frame to
            compare against).
        reg_cov : Optional[np.ndarray], shape (3, 3)
            FS2D's covariance for `reg_delta`, in the same converted
            units. Required (non-`None`) whenever `reg_delta` is.
        """
        cfg = self.cfg

        # --- 1. Dead-reckoned prediction from IMU (bias-corrected) + DVL. ---
        dtheta_pred = imu_dtheta - self.bias_estimate
        var_dtheta_pred = self.bias_variance + np.deg2rad(cfg.assumed_gyro_noise_std_deg) ** 2

        if dvl_disp is not None:
            disp_pred = np.asarray(dvl_disp, dtype=float)
            var_disp_pred = cfg.assumed_dvl_noise_std ** 2
            used_dvl = True
        else:
            disp_pred = np.zeros(2)
            var_disp_pred = cfg.dvl_dropout_disp_std ** 2
            used_dvl = False

        delta_pred = np.array([disp_pred[0], disp_pred[1], dtheta_pred])
        P_pred = np.diag([var_disp_pred, var_disp_pred, var_dtheta_pred])

        # --- 2. Check FS2D for consistency before trusting it (NIS gate;
        # see the module docstring's "WHY THIS MODULE ALSO GATES FS2D AS A
        # POSSIBLE OUTLIER"), then fuse it in if it passes. ---
        used_fs2d = False
        fs2d_rejected_outlier = False
        innovation = None
        S = None
        if reg_delta is not None and reg_cov is not None:
            innovation = np.asarray(reg_delta, dtype=float) - delta_pred
            innovation[2] = _wrap_angle(innovation[2])
            S = P_pred + reg_cov
            mahalanobis_sq = float(innovation @ np.linalg.solve(S, innovation))
            if mahalanobis_sq > cfg.chi2_gate_threshold:
                fs2d_rejected_outlier = True
            else:
                used_fs2d = True

        if used_fs2d:
            K = P_pred @ np.linalg.inv(S)
            delta_fused = delta_pred + K @ innovation
            delta_fused[2] = _wrap_angle(delta_fused[2])
            P_fused = (np.eye(3) - K) @ P_pred
        else:
            delta_fused = delta_pred
            P_fused = P_pred

        # --- 3. Update the running gyro-bias belief. Only possible with an
        # independent, *trusted* FS2D cross-check available this step (an
        # outlier rejected above is exactly as untrustworthy for the bias
        # update as it was for the pose fusion); the bias estimate's
        # uncertainty still grows every step regardless (the random-walk
        # process noise), same as any EKF state you're not currently
        # correcting. ---
        if used_fs2d:
            reg_theta_var = float(reg_cov[2, 2])
            self._update_bias(float(reg_delta[2]), imu_dtheta, reg_theta_var)
        self.bias_variance += np.deg2rad(cfg.bias_process_noise_std_deg) ** 2

        return FusionResult(
            dx=float(delta_fused[0]), dy=float(delta_fused[1]), dtheta=float(delta_fused[2]),
            covariance=P_fused, used_fs2d=used_fs2d, used_dvl=used_dvl,
            fs2d_rejected_outlier=fs2d_rejected_outlier,
            bias_estimate_deg=float(np.rad2deg(self.bias_estimate)),
            bias_std_deg=float(np.rad2deg(np.sqrt(max(self.bias_variance, 0.0)))),
        )

    def _update_bias(self, reg_dtheta: float, imu_dtheta: float, reg_theta_var: float) -> None:
        """Scalar EKF update for the gyro bias.

        One-line derivation: the raw gyro reading is
            imu_dtheta = true_dtheta + bias + noise_imu,
        and FS2D gives an *independent* estimate of the same true
        rotation,
            reg_dtheta = true_dtheta + noise_fs2d.
        Subtracting cancels the (unknown) true_dtheta entirely:
            reg_dtheta - imu_dtheta = -bias + (noise_fs2d - noise_imu),
        which is a direct linear measurement of `-bias`: h(bias) = -bias,
        so H = -1. That makes this an ordinary scalar Kalman update, with
        the measurement noise variance being the sum of both sensors'
        (FS2D's and the gyro's) per-step noise variance -- both
        contribute noise to this comparison. The subtraction is
        angle-wrapped for the same reason `step()`'s pose-fusion
        innovation is: near +-180 degrees, a naive difference is wrong by
        a full turn.
        """
        H = -1.0
        z = _wrap_angle(reg_dtheta - imu_dtheta)
        innovation = _wrap_angle(z - (H * self.bias_estimate))
        R_eff = reg_theta_var + np.deg2rad(self.cfg.assumed_gyro_noise_std_deg) ** 2
        S = H * H * self.bias_variance + R_eff
        K = self.bias_variance * H / S
        self.bias_estimate = self.bias_estimate + K * innovation
        self.bias_variance = (1.0 - K * H) * self.bias_variance
