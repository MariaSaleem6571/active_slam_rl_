"""
Probabilistic change detection (thesis section 5.4, feedback loops D & E).

For every voxel v, the normalized innovation is:

    eta_t^v = | p_t(o_v) - p_{t-1}(o_v) |  /  sqrt(Var[p_t(o_v)] + eps)

Voxels where eta_t^v exceeds a threshold are grouped into a change mask
C_t. This mask:
  (D) locally boosts the reward for revisiting/rescanning that area, and
  (E) is fed into the state encoder as an explicit high-priority region.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def compute_innovation(prob_curr: np.ndarray, prob_prev: np.ndarray,
                        variance: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """eta_t^v for every voxel, vectorized."""
    return np.abs(prob_curr - prob_prev) / np.sqrt(variance + eps)


def compute_change_mask(prob_curr: np.ndarray, prob_prev: np.ndarray,
                         variance: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """Boolean change mask C_t: voxels whose innovation exceeds `threshold`."""
    eta = compute_innovation(prob_curr, prob_prev, variance)
    return eta > threshold


def change_cluster_stats(change_mask: np.ndarray):
    """Cluster the raw change mask into connected components and report
    per-cluster size and centroid — useful both for the reward bonus (D)
    and for visualizing "where the map is currently unstable"."""
    labeled, n_clusters = label(change_mask)
    clusters = []
    for i in range(1, n_clusters + 1):
        ys, xs = np.where(labeled == i)
        clusters.append({
            "size": int(len(ys)),
            "centroid": (float(ys.mean()), float(xs.mean())),
        })
    return clusters
