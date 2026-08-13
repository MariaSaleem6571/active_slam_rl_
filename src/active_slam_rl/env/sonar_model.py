"""
Sonar sensing front-end (thesis section 5.1).

Two modalities are modeled, both reduced to the same abstraction the thesis
describes: calibrated range-intensity returns, modality-agnostic downstream.

  * ImagingSonar   -- wide FOV (~130 deg), many beams, single frame per
                      step (e.g. ARIS/Oculus-style).
  * ScanningSonar  -- narrow beam, full 360 deg sweep, higher angular
                      precision, modeled as a full-rotation composite frame
                      (e.g. Ping360-style); costs one "dwell" step.

Both return:
  * ranges  : per-beam range to the nearest occupied cell (or max range)
  * frame   : an egocentric 2D image (H, W) built from the beam returns,
              suitable as input to FS2D registration and to the state
              encoder's patch branch.

Noise model: additive Gaussian range noise plus a multiplicative speckle
term on intensity (typical of coherent sonar imaging), matching the
thesis's mention of "sonar-specific factors such as side-lobes, beam
divergence, and multipath likelihood" -- modeled here as extra range-noise
variance and an occasional spurious "ghost" return, rather than a full
acoustic propagation simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SonarConfig:
    fov_deg: float = 130.0          # imaging sonar FOV (Ping/ARIS-style ~130 deg)
    n_beams: int = 96
    max_range: float = 22.0         # world units (cells)
    range_noise_std: float = 0.25
    multipath_ghost_prob: float = 0.03
    frame_size: int = 64            # output egocentric image is frame_size x frame_size


class SonarModel:
    """Shared raycasting engine for both imaging and scanning sonar modes."""

    def __init__(self, world, config: SonarConfig = SonarConfig(), rng=None):
        self.world = world
        self.cfg = config
        self.rng = rng or np.random.default_rng()

    def _cast_beams(self, y, x, theta, angles):
        cfg = self.cfg
        ranges = np.full(len(angles), cfg.max_range, dtype=np.float64)
        hit_points = []
        step = 0.5
        for i, a in enumerate(angles):
            beam_theta = theta + a
            dy, dx = np.sin(beam_theta), np.cos(beam_theta)
            r = 0.0
            hit = None
            while r < cfg.max_range:
                r += step
                py, px = y + r * dy, x + r * dx
                if not self.world.is_free(py, px):
                    hit = (py, px)
                    break
            if hit is not None:
                noisy_r = max(0.1, r + self.rng.normal(0, cfg.range_noise_std))
                ranges[i] = min(noisy_r, cfg.max_range)
                hit_points.append((y + ranges[i] * dy, x + ranges[i] * dx))
            else:
                ranges[i] = cfg.max_range
        # Occasional spurious multipath "ghost" return.
        if self.rng.uniform() < cfg.multipath_ghost_prob and len(angles) > 0:
            j = self.rng.integers(len(angles))
            ranges[j] = self.rng.uniform(0.3, cfg.max_range)
        return ranges, hit_points

    def _render_egocentric_frame(self, ranges, angles):
        """Rasterize beam returns into an egocentric (vehicle-frame) image
        for use by FS2D and the state encoder. Intensity falls off with
        range and gets a speckle multiplicative noise term, mimicking real
        sonar imagery statistics.
        """
        cfg = self.cfg
        size = cfg.frame_size
        frame = np.zeros((size, size), dtype=np.float64)
        center = size / 2.0
        scale = (size / 2.0 - 1) / cfg.max_range
        for r, a in zip(ranges, angles):
            # sonar convention: forward = +x (right in the egocentric image), a measured from heading
            px = center + r * scale * np.cos(a)
            py = center + r * scale * np.sin(a)
            xi, yi = int(round(px)), int(round(py))
            if 0 <= yi < size and 0 <= xi < size:
                intensity = max(0.05, 1.0 - r / cfg.max_range)
                speckle = self.rng.gamma(shape=4.0, scale=1.0 / 4.0)  # mean-1 speckle
                frame[yi, xi] = min(1.0, intensity * speckle)
                # small blob so the image isn't a single-pixel dot -> more realistic + easier for FFT registration
                if 0 < yi < size - 1 and 0 < xi < size - 1:
                    frame[yi - 1:yi + 2, xi - 1:xi + 2] = np.maximum(
                        frame[yi - 1:yi + 2, xi - 1:xi + 2], frame[yi, xi] * 0.5)
        return frame

    def sense_imaging(self, y, x, theta):
        cfg = self.cfg
        half_fov = np.deg2rad(cfg.fov_deg) / 2.0
        angles = np.linspace(-half_fov, half_fov, cfg.n_beams)
        ranges, hit_points = self._cast_beams(y, x, theta, angles)
        frame = self._render_egocentric_frame(ranges, angles)
        return ranges, angles, hit_points, frame

    def sense_scanning_360(self, y, x, theta):
        """Full 360-degree mechanical-scan sweep (Ping360-style):
        narrower beam, higher angular precision, costs a 'dwell'."""
        cfg = self.cfg
        n_beams_360 = cfg.n_beams * 3  # denser angular sampling for the full sweep
        angles = np.linspace(-np.pi, np.pi, n_beams_360, endpoint=False)
        ranges, hit_points = self._cast_beams(y, x, theta, angles)
        frame = self._render_egocentric_frame(ranges, angles)
        return ranges, angles, hit_points, frame
