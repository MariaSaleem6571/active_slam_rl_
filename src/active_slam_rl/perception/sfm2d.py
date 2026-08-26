"""
2D Structure-from-Motion (landmark-based, range+bearing), kept as two fully
independent instances -- one for imaging sonar, one for scanning/360 sonar.

NAMING NOTE -- READ THIS BEFORE YOU GO LOOKING FOR "SfM" ELSEWHERE IN THIS
CODEBASE. `fusion/sfm.py`'s `StateFusionModule` (imported into sim_env.py as
`self.sfm`, config field `EnvConfig.sfm: SfMConfig`) is an EKF that fuses
IMU/DVL/FS2D odometry -- it has nothing to do with Structure from Motion.
"SfM" there is a backronym ("State Fusion Module") that predates this file
and unfortunately collides with the real thing. THIS module
(`perception/sfm2d.py`, class `StructureFromMotion2D`, instantiated in
sim_env.py as `self.sfm2d_imaging` / `self.sfm2d_scanning`) is the actual
Structure-from-Motion: it builds and maintains a sparse landmark map from
raw beam returns, and uses re-observed landmarks to produce an independent
pose-correction signal, the way a real SfM / "motion-only bundle
adjustment" pipeline would. If this repo ever renames things to remove the
collision, `fusion/sfm.py` is the one that should change (e.g. to
`fusion/state_fusion.py`) -- "SfM" here is the one actually entitled to
the name.

WHY RANGE+BEARING SENSORS DON'T NEED CLASSICAL TRIANGULATION
--------------------------------------------------------------
Camera-based SfM triangulates because a single camera ray is
depth-ambiguous -- you need two-or-more views to recover a 3D point. Sonar
directly measures (range, bearing), so a *single* beam return already
gives a full 2D position once you know the sensing pose:

    landmark_y = pose_y + range * sin(pose_theta + bearing)
    landmark_x = pose_x + range * cos(pose_theta + bearing)

"Structure" here means something adjacent but analogous: (a) *data
association* across frames -- deciding which beam returns, observed from
different poses, are repeat observations of the same physical point --
and (b) *refinement* -- fusing repeated noisy observations of a landmark
into a better position estimate, then using the discrepancy between
"where a previously-seen landmark should appear from here" and "where it
was actually observed this frame" to correct the *pose* estimate. That
second part is a single-iteration Gauss-Newton solve on a standard
2D range+bearing resection (identical structure to motion-only bundle
adjustment / PnP in camera SfM), linearized once per frame rather than
iterated to convergence -- matching this codebase's existing per-step
(not offline-batch) style everywhere else (FS2D, loop closure, the fusion
EKF all work the same way: one estimate per env.step(), not an
accumulate-then-optimize batch process).

WHY IMAGING AND SCANNING GET SEPARATE MAPS, NOT ONE SHARED MAP
------------------------------------------------------------------
Mirrors the reasoning already applied to FS2D registration
(env/sim_env.py's cross-modality-registration guard) and loop-closure
keyframe matching (perception/loop_closure.py): the two sonar modalities
differ substantially in range-noise characteristics, angular resolution,
and reliability (see registration/fs2d.py's fold-ambiguity discussion).
Landmarks built from noisy, sparse imaging-sonar returns and landmarks
built from denser, more precise scanning-sonar returns aren't
interchangeable evidence about the same physical point at comparable
confidence, and merging them into one map would let one modality's
mistakes silently corrupt the other's otherwise-good structure estimate.
Keeping two independent maps also directly supports comparing the two
modalities' mapping quality side by side (see
metrics/plotting.py::plot_sfm2d_landmark_maps), which was the point of
separating them in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SfM2DConfig:
    match_gate_m: float = 0.6
    # A frame needs at least this many re-observed (n_obs >= 2) landmark
    # matches before we trust a pose correction from it -- 3 constraints
    # is the theoretical minimum for a 3-DOF (y, x, theta) solve, so this
    # asks for a healthier margin above that: right at the minimum, the
    # normal equations can be numerically well-determined yet still fit a
    # bad association or an unlucky, poorly-distributed geometry (e.g. all
    # matched landmarks clustered in a narrow bearing range) rather than
    # true structure, producing a large, confident-looking, wrong
    # correction. Empirically this happened at min_matched=4 with just a
    # handful of matches (see the max_correction_* gate below for the
    # second, complementary line of defense against exactly this).
    min_matched_for_correction: int = 6
    max_landmarks: int = 4000
    range_noise_std: float = 0.25          # should track SonarConfig.range_noise_std
    bearing_noise_std_deg: float = 2.0
    # Sanity gate on the *magnitude* of a single frame's computed
    # correction, independent of min_matched_for_correction above -- a
    # genuine correction for ordinary per-step drift should be small
    # (this environment's process noise is a small fraction of a meter
    # per step; see EnvConfig.process_noise_xy). A single-step correction
    # bigger than this is far more likely to be an ill-conditioned or
    # bad-association solve than genuine accumulated drift, and applying
    # it would inject exactly the kind of confident-but-wrong jump this
    # gate exists to catch (mirrors fusion/sfm.py's own NIS outlier gate
    # on FS2D registrations, applied here to SfM2D's own correction
    # output instead).
    max_correction_translation_m: float = 1.0
    max_correction_rotation_deg: float = 15.0


@dataclass
class Landmark:
    position: np.ndarray   # (2,) world-frame (y, x)
    n_obs: int
    last_seen_t: int
    modality: str


@dataclass
class SfM2DResult:
    pose_correction: Optional[Tuple[float, float, float]]   # (dy, dx, dtheta) or None
    covariance: Optional[np.ndarray]                          # (3, 3) or None
    n_matched: int
    n_new_landmarks: int
    residual_rms: Optional[float]


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class StructureFromMotion2D:
    """One independent landmark map + pose-correction estimator for a
    single sonar modality ("imaging" or "scanning"). sim_env.py keeps two
    separate instances so the two modalities' maps and correction streams
    never mix -- see this module's docstring for why.
    """

    def __init__(self, modality: str, config: SfM2DConfig = SfM2DConfig()):
        assert modality in ("imaging", "scanning")
        self.modality = modality
        self.cfg = config
        self.landmarks: List[Landmark] = []
        self._tree: Optional[cKDTree] = None
        self._tree_dirty = True

    def process_frame(self, ranges: np.ndarray, angles: np.ndarray,
                       pose_estimate: Tuple[float, float, float],
                       timestep: int, max_range: float) -> SfM2DResult:
        """`ranges`/`angles` are this frame's raw per-beam returns in body
        frame, exactly as returned by SonarModel.sense_imaging /
        sense_scanning_360 (angles measured relative to heading; a beam
        that hit nothing reads ranges[i] == max_range -- see
        SonarModel._cast_beams). `pose_estimate` MUST be the vehicle's own
        current pose estimate (est_pose), never true_pose: using true_pose
        here would leak ground truth into the very correction this is
        supposed to be estimating, defeating the entire point of the
        exercise (this mirrors env/sim_env.py's own comment about why
        est_pose vs. true_pose are tracked separately in the first place).
        """
        Y, X, Theta = pose_estimate
        cfg = self.cfg

        ranges = np.asarray(ranges, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        valid = ranges < (max_range - 1e-6)   # a beam that never hit anything isn't a landmark observation
        if not np.any(valid):
            return SfM2DResult(None, None, 0, 0, None)

        r_valid = ranges[valid]
        a_valid = angles[valid]
        bearing_world = Theta + a_valid
        cand_y = Y + r_valid * np.sin(bearing_world)
        cand_x = X + r_valid * np.cos(bearing_world)

        if self.landmarks and (self._tree is None or self._tree_dirty):
            self._tree = cKDTree(np.array([lm.position for lm in self.landmarks]))
            self._tree_dirty = False

        if self._tree is not None:
            query_pts = np.stack([cand_y, cand_x], axis=1)
            dists, idxs = self._tree.query(query_pts, k=1)
        else:
            dists = np.full(len(r_valid), np.inf)
            idxs = np.zeros(len(r_valid), dtype=int)

        # Pass 1: decide matches/spawns using the *pre-correction* estimate.
        # This is fine even though the estimate may be biased -- matching
        # only needs to get the right landmark within match_gate_m, which
        # tolerates ordinary drift.
        match_kind = []   # "matched" | "new" | "drop", per valid beam
        matched_idx_for_correction, matched_r, matched_a = [], [], []
        n_new = 0
        for i in range(len(r_valid)):
            if len(self.landmarks) > 0 and dists[i] <= cfg.match_gate_m:
                match_kind.append(("matched", idxs[i]))
                lm = self.landmarks[idxs[i]]
                if lm.n_obs >= 2:
                    matched_idx_for_correction.append(idxs[i])
                    matched_r.append(r_valid[i])
                    matched_a.append(a_valid[i])
            elif len(self.landmarks) < cfg.max_landmarks:
                match_kind.append(("new", None))
                n_new += 1
            else:
                match_kind.append(("drop", None))

        n_matched = len(matched_idx_for_correction)
        if n_matched < cfg.min_matched_for_correction:
            correction, cov, rms = None, None, None
        else:
            correction, cov, rms = self._solve_pose_correction(
                Y, X, Theta, matched_idx_for_correction, np.array(matched_r), np.array(matched_a))
            if correction is not None:
                trans_mag = float(np.hypot(correction[0], correction[1]))
                rot_mag_deg = float(np.rad2deg(abs(correction[2])))
                if (trans_mag > cfg.max_correction_translation_m
                        or rot_mag_deg > cfg.max_correction_rotation_deg):
                    # Sanity gate tripped -- see SfM2DConfig's
                    # max_correction_translation_m/max_correction_rotation_deg
                    # docstring. Still counts as "matched" for n_matched
                    # reporting, but the correction itself is discarded
                    # (None), and Pass 2 below falls back to using the
                    # pre-correction estimate for landmark updates too.
                    correction, cov, rms = None, None, None

        # Pass 2: apply landmark position updates using the *corrected*
        # pose if we got one this frame, not the pre-correction estimate --
        # otherwise every update nudges the map itself toward whatever
        # bias the pose estimate currently has, which then makes that same
        # bias look "confirmed" by the map on the next frame (the map and
        # the pose estimate co-adapt toward each other's error instead of
        # the map correcting the pose). Falls back to the pre-correction
        # estimate when no correction was computed this frame (nothing
        # better available yet).
        if correction is not None:
            Yc, Xc, Thetac = Y + correction[0], X + correction[1], Theta + correction[2]
            bearing_world_c = Thetac + a_valid
            cand_y = Yc + r_valid * np.sin(bearing_world_c)
            cand_x = Xc + r_valid * np.cos(bearing_world_c)

        for i, (kind, idx) in enumerate(match_kind):
            if kind == "matched":
                lm = self.landmarks[idx]
                gain = 1.0 / (lm.n_obs + 1)
                lm.position = lm.position + gain * (np.array([cand_y[i], cand_x[i]]) - lm.position)
                lm.n_obs += 1
                lm.last_seen_t = timestep
                self._tree_dirty = True
            elif kind == "new":
                self.landmarks.append(Landmark(
                    position=np.array([cand_y[i], cand_x[i]]),
                    n_obs=1, last_seen_t=timestep, modality=self.modality))
                self._tree_dirty = True

        return SfM2DResult(correction, cov, n_matched, n_new, rms)

    def _solve_pose_correction(self, Y, X, Theta, landmark_idx, r_obs, a_obs
                                ) -> Tuple[Optional[Tuple[float, float, float]],
                                           Optional[np.ndarray], Optional[float]]:
        """Single Gauss-Newton step solving for the (dy, dx, dtheta)
        correction to (Y, X, Theta) that best reconciles this frame's
        observed (range, bearing) to each matched landmark with that
        landmark's current estimated position -- i.e. 2D resection /
        motion-only bundle adjustment. See module docstring for the
        Jacobian derivation.
        """
        cfg = self.cfg
        positions = np.array([self.landmarks[i].position for i in landmark_idx])
        dy = positions[:, 0] - Y
        dx = positions[:, 1] - X
        r_pred = np.hypot(dy, dx)
        r_pred = np.maximum(r_pred, 1e-6)
        bearing_pred = _wrap(np.arctan2(dy, dx) - Theta)

        e_r = r_obs - r_pred
        e_b = _wrap(a_obs - bearing_pred)
        e = np.empty(2 * len(landmark_idx))
        e[0::2], e[1::2] = e_r, e_b

        J = np.zeros((2 * len(landmark_idx), 3))
        J[0::2, 0] = -dy / r_pred
        J[0::2, 1] = -dx / r_pred
        J[0::2, 2] = 0.0
        J[1::2, 0] = -dx / (r_pred ** 2)
        J[1::2, 1] = dy / (r_pred ** 2)
        J[1::2, 2] = -1.0

        sigma_r2 = cfg.range_noise_std ** 2
        sigma_b2 = np.deg2rad(cfg.bearing_noise_std_deg) ** 2
        w = np.empty(2 * len(landmark_idx))
        w[0::2] = 1.0 / sigma_r2
        w[1::2] = 1.0 / sigma_b2
        W = np.diag(w)

        JTW = J.T @ W
        A = JTW @ J
        b = JTW @ e
        try:
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return None, None, None
        delta = cov @ b
        residual_rms = float(np.sqrt(np.mean(e ** 2)))
        return (float(delta[0]), float(delta[1]), float(delta[2])), cov, residual_rms

    def get_map(self) -> List[Landmark]:
        return self.landmarks

    def reset(self):
        """Clear the landmark map for a fresh episode -- draws no
        randomness of its own, so this is the only state that needs
        resetting (mirrors fusion/sfm.py's StateFusionModule.reset())."""
        self.landmarks = []
        self._tree = None
        self._tree_dirty = True
