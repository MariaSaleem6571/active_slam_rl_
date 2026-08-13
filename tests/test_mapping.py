import numpy as np

from active_slam_rl.mapping.volumetric_map import OccupancyGrid
from active_slam_rl.mapping.change_detection import compute_change_mask, compute_innovation


def test_occupancy_grid_updates_toward_evidence():
    grid = OccupancyGrid(10, 10)
    assert np.allclose(grid.prob, 0.5)  # fully unknown prior
    grid.update_beam((5, 5), free_cells=[(5, 6), (5, 7)], hit_cell=(5, 8))
    assert grid.prob[5, 6] < 0.5   # free evidence pushes belief down
    assert grid.prob[5, 8] > 0.5   # occupied evidence pushes belief up


def test_entropy_decreases_as_belief_resolves():
    grid = OccupancyGrid(10, 10)
    h0 = grid.entropy()
    for _ in range(20):
        grid.update_beam((5, 5), free_cells=[(5, 6)], hit_cell=(5, 8))
    h1 = grid.entropy()
    assert h1 < h0


def test_completeness_metric_bounds():
    grid = OccupancyGrid(10, 10)
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[8, 8] = 1
    c = grid.completeness(gt)
    assert 0.0 <= c <= 100.0


def test_change_mask_flags_large_innovation():
    prob_prev = np.full((5, 5), 0.5)
    prob_curr = prob_prev.copy()
    prob_curr[2, 2] = 0.95  # large jump
    variance = np.full((5, 5), 0.05)
    mask = compute_change_mask(prob_curr, prob_prev, variance, threshold=1.0)
    assert mask[2, 2]
    assert not mask[0, 0]
