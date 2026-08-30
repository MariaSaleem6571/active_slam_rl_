"""
Procedurally generated narrow underwater tunnel worlds (thesis 5.9, Stage 2).

Produces a 2D top-down occupancy grid (1 = wall/rock, 0 = navigable water)
via a randomized corridor "digger" that:
  * carves a main corridor with a random walk, with randomized width,
  * occasionally spawns branches (matching "branch frequency" as a
    randomization parameter from the thesis),
  * scatters debris (small occupied blobs) at a controllable density,
  * optionally closes the loop back near the start, so a genuine
    loop-closure opportunity exists (Stage 3 also wants "long corridors
    with no loop closures" — set `loop_probability=0.0` for that case).

This 2D plan-view stands in for the full 3D tunnel/wreck geometry a
higher-fidelity 3D simulator would render, should this project attach to
one later; the downstream RL/env interface (ActiveSlamEnv) is designed
so that swap only requires a new sensing/world adapter, not changes to
the policy, reward, or training code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorldConfig:
    height: int = 140
    width: int = 140
    corridor_width_range: tuple = (6, 10)
    branch_probability: float = 0.03       # per-step chance of spawning a branch
    max_branches: int = 3
    debris_density: float = 0.01           # fraction of free cells turned into debris
    loop_probability: float = 0.6          # chance the corridor loops back near start
    n_steps: int = 900                     # random-walk steps for the main corridor
    step_len: int = 2
    turn_std_deg: float = 18.0
    seed: int | None = None


class TunnelWorld:
    def __init__(self, config: WorldConfig = WorldConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.occ = np.ones((config.height, config.width), dtype=np.uint8)  # start all-wall
        self.start_pose = None
        self.free_mask = None
        self._generate()

    def _carve_disk(self, cy, cx, radius):
        h, w = self.occ.shape
        y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
        x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        self.occ[y0:y1, x0:x1][mask] = 0

    def _random_walk_corridor(self, start, n_steps, width_range):
        cfg = self.cfg
        y, x = start
        theta = self.rng.uniform(0, 2 * np.pi)
        path = [(y, x)]
        for _ in range(n_steps):
            theta += np.deg2rad(self.rng.normal(0, cfg.turn_std_deg))
            y += cfg.step_len * np.sin(theta)
            x += cfg.step_len * np.cos(theta)
            y = np.clip(y, 5, cfg.height - 5)
            x = np.clip(x, 5, cfg.width - 5)
            radius = self.rng.uniform(*width_range) / 2.0
            self._carve_disk(y, x, radius)
            path.append((y, x))
        return path

    def _scatter_debris(self):
        free_ys, free_xs = np.where(self.occ == 0)
        n_free = len(free_ys)
        n_debris = int(n_free * self.cfg.debris_density)
        if n_debris == 0 or n_free == 0:
            return
        idx = self.rng.choice(n_free, size=n_debris, replace=False)
        for i in idx:
            r = self.rng.uniform(0.6, 1.6)
            self._carve_disk_inverse(free_ys[i], free_xs[i], r)

    def _carve_disk_inverse(self, cy, cx, radius):
        """Place a small occupied blob (debris) inside free space."""
        h, w = self.occ.shape
        y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
        x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        self.occ[y0:y1, x0:x1][mask] = 1

    def _generate(self):
        cfg = self.cfg
        start = (cfg.height * 0.5, cfg.width * 0.15)
        main_path = self._random_walk_corridor(start, cfg.n_steps, cfg.corridor_width_range)

        n_branches = self.rng.integers(0, cfg.max_branches + 1)
        for _ in range(n_branches):
            branch_start = main_path[self.rng.integers(len(main_path) // 4, len(main_path))]
            branch_len = self.rng.integers(cfg.n_steps // 6, cfg.n_steps // 3)
            narrower = (cfg.corridor_width_range[0] * 0.7, cfg.corridor_width_range[1] * 0.8)
            self._random_walk_corridor(branch_start, branch_len, narrower)

        if self.rng.uniform() < cfg.loop_probability:
            # Bias the tail of the path back toward the start to create a
            # genuine loop-closure opportunity.
            end = main_path[-1]
            n_close = 200
            y, x = end
            ty, tx = start
            for i in range(n_close):
                frac = i / n_close
                y = y + (ty - y) * 0.02 + self.rng.normal(0, 1.5)
                x = x + (tx - x) * 0.02 + self.rng.normal(0, 1.5)
                y = np.clip(y, 5, cfg.height - 5)
                x = np.clip(x, 5, cfg.width - 5)
                radius = self.rng.uniform(*cfg.corridor_width_range) / 2.0
                self._carve_disk(y, x, radius)

        self._scatter_debris()
        self.free_mask = (self.occ == 0)
        self.start_pose = (float(start[0]), float(start[1]), 0.0)

    def is_free(self, y: float, x: float) -> bool:
        h, w = self.occ.shape
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < h and 0 <= xi < w:
            return self.occ[yi, xi] == 0
        return False

    def free_fraction_explored_budget(self) -> int:
        return int(self.free_mask.sum())
