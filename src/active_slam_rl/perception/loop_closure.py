"""
Place recognition / loop-closure saliency (feedback loop F).

A full learned sonar descriptor network (as the thesis's "SONAR descriptor
library") is out of scope for a from-scratch implementation, but the
*interface and information content* it must provide to the rest of the
pipeline is exactly reproduced here with a lightweight, interpretable
descriptor:

  1. Each visited pose keeps a compact rotation-invariant descriptor of its
     sonar frame (radial intensity histogram — cheap, and by construction
     invariant to the vehicle's heading at that pose, which is what makes
     it useful for recognizing a *revisited* place seen from a different
     angle).
  2. At the current pose, we compare the current descriptor against all
     stored keyframe descriptors. The best match's similarity becomes the
     loop-closure saliency ell_t in [0, 1] injected into the state vector.
  3. If similarity clears a threshold, the match is registered via FS2D
     against the stored keyframe scan to get a loop-closure constraint
     (dx, dy, dtheta, quality) — this is what produces Delta I_t (pose-graph
     information gain) and the b_lc,t bonus in the reward.

Swap-in point: replace `SonarDescriptor.compute` with a learned embedding
(e.g. a small CNN / NetVLAD-style head) without touching anything else —
`LoopClosureDetector` only depends on descriptors supporting a distance.

MODALITY TAGGING
-----------------
`ActiveSlamEnv` has two mutually-exclusive sonar modalities per step
(`SonarModel.sense_imaging` -- narrow FOV, or `sense_scanning_360` -- a
full 360-degree sweep; see `env/sonar_model.py`). Their egocentric frames
differ enough in coverage and beam density that a descriptor/FS2D match
between them isn't meaningful -- a 360-degree sweep frame simply doesn't
correspond pixel-for-pixel to a ~130-degree forward cone. Every keyframe
therefore records the sonar `mode` ("imaging"/"scanning") it was captured
with, and `query()` only compares the current frame's descriptor against
keyframes captured in the *same* mode, so any candidate handed back for
FS2D re-registration (`ActiveSlamEnv.step`) is guaranteed to be a
same-modality pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Keyframe:
    pose: Tuple[float, float, float]   # (x, y, theta) at time of capture
    descriptor: np.ndarray
    scan: np.ndarray
    timestep: int
    mode: str = "imaging"   # "imaging" or "scanning" -- which sonar modality
                             # captured `scan`; see module docstring


@dataclass
class LoopClosureCandidate:
    keyframe_idx: int
    saliency: float               # ell_t in [0, 1]
    pose_delta_hint: Optional[Tuple[float, float, float]] = None


class SonarDescriptor:
    """Rotation-invariant radial intensity histogram descriptor."""

    def __init__(self, n_bins: int = 32):
        self.n_bins = n_bins

    def compute(self, scan: np.ndarray) -> np.ndarray:
        h, w = scan.shape
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_r = r.max() + 1e-6
        bins = np.clip((r / max_r * self.n_bins).astype(int), 0, self.n_bins - 1)
        hist = np.zeros(self.n_bins)
        counts = np.zeros(self.n_bins)
        np.add.at(hist, bins.ravel(), scan.ravel())
        np.add.at(counts, bins.ravel(), 1)
        hist = hist / np.maximum(counts, 1)
        norm = np.linalg.norm(hist) + 1e-8
        return hist / norm


class LoopClosureDetector:
    """Maintains a keyframe database and scores the current scan against it
    to produce the loop-closure saliency ell_t used in feedback loop F.
    """

    def __init__(self, similarity_threshold: float = 0.9, min_pose_distance: float = 10.0,
                 n_bins: int = 32):
        self.descriptor_fn = SonarDescriptor(n_bins=n_bins)
        self.keyframes: List[Keyframe] = []
        self.similarity_threshold = similarity_threshold
        self.min_pose_distance = min_pose_distance  # avoid matching the immediate past

    def maybe_add_keyframe(self, pose, scan, timestep, mode: str = "imaging"):
        """Add a keyframe if we've moved far enough from the last one
        (keeps the database compact and avoids trivial self-matches).

        `mode` records which sonar modality captured `scan` ("imaging" or
        "scanning") -- see module docstring's "MODALITY TAGGING" section.
        Deliberately compared against the last keyframe *of any* mode for
        the distance gate (it's still the same vehicle trajectory), but
        stored per-keyframe so `query()` can restrict matching to the same
        modality.
        """
        if not self.keyframes:
            self._add(pose, scan, timestep, mode)
            return
        last = self.keyframes[-1]
        d = np.hypot(pose[0] - last.pose[0], pose[1] - last.pose[1])
        if d > self.min_pose_distance:
            self._add(pose, scan, timestep, mode)

    def _add(self, pose, scan, timestep, mode: str = "imaging"):
        desc = self.descriptor_fn.compute(scan)
        self.keyframes.append(Keyframe(pose=pose, descriptor=desc, scan=scan,
                                        timestep=timestep, mode=mode))

    def query(self, scan: np.ndarray, current_timestep: int, exclude_recent: int = 20,
              mode: str = "imaging") -> Tuple[float, Optional[LoopClosureCandidate]]:
        """Returns (ell_t, best_candidate_or_None).

        Only keyframes captured in the same sonar `mode` as the current
        frame are considered -- an imaging-sonar frame and a scanning-sonar
        frame don't cover comparable geometry, so neither the descriptor
        similarity nor a downstream FS2D re-registration against a
        cross-modality keyframe would be meaningful (see module docstring).
        """
        if not self.keyframes:
            return 0.0, None
        desc = self.descriptor_fn.compute(scan)
        best_sim, best_idx = -1.0, -1
        for i, kf in enumerate(self.keyframes):
            if current_timestep - kf.timestep < exclude_recent:
                continue
            if kf.mode != mode:
                continue
            sim = float(np.dot(desc, kf.descriptor))  # cosine similarity, descriptors are unit-norm
            if sim > best_sim:
                best_sim, best_idx = sim, i
        if best_idx == -1:
            return 0.0, None
        saliency = float(np.clip(best_sim, 0.0, 1.0))
        candidate = None
        if saliency >= self.similarity_threshold:
            candidate = LoopClosureCandidate(keyframe_idx=best_idx, saliency=saliency)
        return saliency, candidate

    def keyframe_pose(self, idx: int):
        return self.keyframes[idx].pose

    def keyframe_scan(self, idx: int):
        return self.keyframes[idx].scan
