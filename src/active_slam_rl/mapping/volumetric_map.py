"""
Volumetric map M_t with per-voxel uncertainty (thesis section 5.3).

We implement a 2D occupancy grid (rows x cols voxels/cells) rather than a
full 3D TSDF, because:
  * the demo/training environment (env/sim_env.py) is a 2D tunnel-plan-view
    world (matching how the sonar frames are already 2D range-intensity
    images used by FS2D above), and
  * every equation in the thesis (Bayesian update, entropy, change
    detection) is dimension-agnostic — the exact same update rule applies
    voxel-by-voxel whether the grid is 2D or 3D.

`OccupancyGrid` below is written so that swapping in a 3D array (shape
(D, H, W) instead of (H, W)) requires no change to the update logic itself
— every operation is elementwise / numpy-broadcastable. See
`docs/ARCHITECTURE.md` for the note on extending this to real TSDF/Octomap/
Voxblox output when integrating with MarineGym's actual sonar data.

Bayesian update (log-odds form, for numerical stability):
    l_t(v) = l_{t-1}(v) + inverse_sensor_model(z_t | v, x_t)
    p_t(v) = sigmoid(l_t(v))
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class OccupancyGrid:
    """Bayesian occupancy grid storing, per voxel v:
      - occupancy belief p(o_v) in [0, 1]      -> self.prob
      - uncertainty sigma_v^2 (belief variance) -> self.uncertainty()
      - visibility counter n_v                  -> self.visits
    """

    def __init__(self, height: int, width: int, resolution: float = 1.0,
                 l_occ: float = 0.85, l_free: float = -0.4, l_max: float = 8.0):
        self.height = height
        self.width = width
        self.resolution = resolution
        # log-odds map, initialized at l=0 -> p=0.5 (fully unknown)
        self.log_odds = np.zeros((height, width), dtype=np.float64)
        self.visits = np.zeros((height, width), dtype=np.int32)
        self.l_occ = l_occ     # log-odds increment for an occupied hit
        self.l_free = l_free   # log-odds increment for a free-space pass-through
        self.l_max = l_max     # clamp to avoid overconfident saturation

        # previous belief snapshot, kept for change detection (eta_t^v)
        self._prev_prob = self.prob.copy()

    @property
    def prob(self) -> np.ndarray:
        """p_t(o_v), the current occupancy belief for every voxel."""
        return _sigmoid(self.log_odds)

    def uncertainty(self) -> np.ndarray:
        """sigma_v^2: Bernoulli variance p(1-p) of the occupancy belief.
        High near p=0.5 (truly unknown), low near 0 or 1 (confident)."""
        p = self.prob
        return p * (1.0 - p)

    def entropy(self) -> float:
        """H(M_t) = -sum_v [p log p + (1-p) log(1-p)]  (total map entropy,
        reported as an evaluation metric)."""
        p = np.clip(self.prob, 1e-6, 1 - 1e-6)
        h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return float(h.sum())

    def entropy_normalized(self) -> float:
        """Mean per-voxel entropy, in [0, ln(2)]. Used inside the reward's
        Delta V^unc term so reward magnitude doesn't scale with grid size."""
        return self.entropy() / (self.height * self.width)

    def snapshot_for_change_detection(self):
        """Call once per step, *before* update(), to freeze p_{t-1}."""
        self._prev_prob = self.prob.copy()

    def update_beam(self, origin_rc, free_cells, hit_cell):
        """Bayesian update along one sonar beam.

        Parameters
        ----------
        free_cells : list[(row, col)]
            Cells the beam passed through without a return (free-space
            evidence).
        hit_cell : (row, col) or None
            Cell where the beam returned an echo (occupied evidence), or
            None if the beam reached max range with no return.
        """
        for (r, c) in free_cells:
            if 0 <= r < self.height and 0 <= c < self.width:
                self.log_odds[r, c] = np.clip(
                    self.log_odds[r, c] + self.l_free, -self.l_max, self.l_max)
                self.visits[r, c] += 1
        if hit_cell is not None:
            r, c = hit_cell
            if 0 <= r < self.height and 0 <= c < self.width:
                self.log_odds[r, c] = np.clip(
                    self.log_odds[r, c] + self.l_occ, -self.l_max, self.l_max)
                self.visits[r, c] += 1

    def completeness(self, ground_truth_occ: np.ndarray, threshold: float = 0.6) -> float:
        """Volumetric completeness metric (thesis section 8.1):
            |V_reconstructed ∩ V_ground_truth| / |V_ground_truth| * 100%
        A cell counts as "reconstructed" once its belief is confidently
        resolved (either occupied or free) rather than merely visited.
        """
        resolved_occ = (self.prob > threshold)
        gt_occ = ground_truth_occ.astype(bool)
        if gt_occ.sum() == 0:
            return 100.0
        intersect = np.logical_and(resolved_occ, gt_occ).sum()
        return 100.0 * intersect / gt_occ.sum()
