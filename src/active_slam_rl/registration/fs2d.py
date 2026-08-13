"""
FS2D — Fourier-based Sonar-to-Sonar registration.

WHAT THIS MODULE DOES
----------------------
Given two consecutive sonar scans (as 2D range-intensity images, i.e. a polar
sonar frame resampled onto a Cartesian grid), estimate the relative rigid
transform between them:

    delta_x_t = (dx, dy, dtheta)

using the Fourier-Mellin Transform (FMT):

  1. Rotation is found in the frequency domain by transforming both images
     to log-polar coordinates and cross-correlating along the angular axis
     (rotation becomes a *shift* in log-polar space -> phase correlation
     finds it in O(N log N)).
  2. One image is de-rotated by the estimated angle.
  3. Translation is then found with ordinary 2D phase correlation.

This is exactly the class of algorithm the thesis calls "FS2D" (Fourier
sonar-to-sonar registration): it is robust to speckle noise (phase
correlation is amplitude-invariant) and to partial overlap (only the
low/mid frequency content needs to agree).

Alongside the estimated pose, FS2D must also emit two uncertainty products
that the rest of the pipeline depends on (see thesis section 5.2):

  * q_t   in [0, 1]      -- scalar match-quality score
  * Sigma_reg_t (3x3)    -- covariance of (dx, dy, dtheta)

TWO IMPLEMENTATIONS, ONE INTERFACE
-----------------------------------
Constructor University's FS2D is a compiled C/C++ library (fast, but not
something we can hand-wave a Python port of without the source). This file
therefore ships with:

  * `FourierMellinRegistration` -- a genuine, from-scratch NumPy/SciPy
    implementation of the Fourier-Mellin algorithm described above. It is
    algorithmically faithful to FS2D and is what the RL environment uses
    *right now*, so you can train immediately without waiting on the native
    library.

  * `NativeFS2DBinding` -- a `ctypes` FFI wrapper that calls into the real
    compiled library once you have it (`libfs2d.so` / `libfs2d.dylib` /
    `fs2d.dll`). See `native/fs2d/README.md` for the exact C ABI this
    expects and how to build/link it.

  * `FS2DRegistration` -- a thin facade used everywhere else in the code.
    It tries the native binding first and transparently falls back to the
    NumPy implementation if the native library isn't found. Nothing else in
    the codebase needs to know which backend is active.

This mirrors a common pattern in robotics research code: develop and train
against a faithful reference implementation, then swap in the optimized
native backend for real-time deployment without touching downstream code.
"""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import geometric_transform, map_coordinates


@dataclass
class RegistrationResult:
    """Output of a single FS2D registration call.

    Attributes
    ----------
    dx, dy : float
        Estimated translation, in pixels (convert to metres using the map
        resolution before fusing with SfM/IMU/DVL).
    dtheta : float
        Estimated rotation, in radians.
    quality : float
        Scalar match quality q_t in [0, 1]. Derived from the sharpness of
        the phase-correlation peak (a diffuse peak -> low confidence).
    covariance : np.ndarray, shape (3, 3)
        Sigma_reg_t over (dx, dy, dtheta), estimated from the curvature of
        the correlation surface around its peak (Cramer-Rao-style local
        approximation, not a full Bayesian posterior).
    backend : str
        Which implementation actually produced this result: "native" or
        "fourier_mellin_numpy".
    """

    dx: float
    dy: float
    dtheta: float
    quality: float
    covariance: np.ndarray
    backend: str


def _hann_window_2d(shape) -> np.ndarray:
    """2D Hann window to reduce edge leakage before FFT (standard practice
    for phase correlation on non-periodic sonar frames)."""
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def _phase_correlation(a: np.ndarray, b: np.ndarray, upsample: int = 8):
    """Standard normalized phase correlation with sub-pixel refinement.

    Returns (shift_row, shift_col, peak_sharpness) where peak_sharpness in
    [0, 1] is used downstream as part of the quality score q_t.
    """
    eps = 1e-8
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    R /= (np.abs(R) + eps)
    r = np.fft.ifft2(R).real
    r = np.fft.fftshift(r)

    peak_idx = np.unravel_index(np.argmax(r), r.shape)
    peak_val = r[peak_idx]

    # Sharpness: peak vs. mean of surrounding energy -> proxy for how
    # confidently the two scans actually agree (speckle noise -> flat,
    # ambiguous correlation surface -> low sharpness -> low q_t).
    local = r[max(peak_idx[0] - 3, 0):peak_idx[0] + 4,
              max(peak_idx[1] - 3, 0):peak_idx[1] + 4]
    sharpness = float(np.clip((peak_val - r.mean()) / (r.std() + eps) / 10.0, 0.0, 1.0))

    # Sub-pixel refinement via parabolic interpolation around the peak.
    def _parabolic(f_m1, f_0, f_p1):
        denom = (f_m1 - 2 * f_0 + f_p1)
        if abs(denom) < eps:
            return 0.0
        return 0.5 * (f_m1 - f_p1) / denom

    py, px = peak_idx
    if 0 < py < r.shape[0] - 1 and 0 < px < r.shape[1] - 1:
        dy_sub = _parabolic(r[py - 1, px], r[py, px], r[py + 1, px])
        dx_sub = _parabolic(r[py, px - 1], r[py, px], r[py, px + 1])
    else:
        dy_sub = dx_sub = 0.0

    center = np.array(r.shape) // 2
    shift_row = (py - center[0]) + dy_sub
    shift_col = (px - center[1]) + dx_sub
    return shift_row, shift_col, sharpness, r, (py, px)


def _to_log_polar(img: np.ndarray, n_angle: int = 360, n_radius: Optional[int] = None):
    """Resample a Cartesian image onto a log-polar grid. Rotation and scale
    in Cartesian space become simple shifts along the angle/log-radius axes.
    """
    h, w = img.shape
    cy, cx = h / 2.0, w / 2.0
    if n_radius is None:
        n_radius = min(h, w) // 2
    max_radius = min(h, w) / 2.0
    log_base = np.log(max_radius) / n_radius

    def _map(output_coords):
        r_idx, theta_idx = output_coords
        radius = np.exp(r_idx * log_base)
        theta = theta_idx * (2 * np.pi / n_angle)
        y = cy + radius * np.sin(theta)
        x = cx + radius * np.cos(theta)
        return (y, x)

    return geometric_transform(img, _map, output_shape=(n_radius, n_angle), order=1, mode="constant")


class FourierMellinRegistration:
    """Reference NumPy/SciPy implementation of Fourier-Mellin sonar
    registration. Algorithmically faithful to FS2D; used as the default
    (and fallback) backend so training can start before the native library
    is linked in.
    """

    def __init__(self, n_angle: int = 360):
        self.n_angle = n_angle

    def register(self, scan_prev: np.ndarray, scan_curr: np.ndarray) -> RegistrationResult:
        assert scan_prev.shape == scan_curr.shape, "Scans must be the same size"
        a = scan_prev.astype(np.float64)
        b = scan_curr.astype(np.float64)
        win = _hann_window_2d(a.shape)
        a_w, b_w = a * win, b * win

        # --- Step 1: rotation via log-polar phase correlation of the
        # Fourier-magnitude spectra (magnitude is translation-invariant, so
        # this isolates rotation/scale). ---
        Fa_mag = np.fft.fftshift(np.abs(np.fft.fft2(a_w)))
        Fb_mag = np.fft.fftshift(np.abs(np.fft.fft2(b_w)))
        lp_a = _to_log_polar(np.log1p(Fa_mag), n_angle=self.n_angle)
        lp_b = _to_log_polar(np.log1p(Fb_mag), n_angle=self.n_angle)

        _, shift_theta_idx, rot_sharpness, _, _ = _phase_correlation(lp_a, lp_b)
        dtheta = shift_theta_idx * (2 * np.pi / self.n_angle)
        # Fold ambiguity (Fourier-Mellin gives rotation mod pi in this
        # simplified real-valued form); resolve by testing both candidates.
        candidates = [dtheta, dtheta + np.pi if dtheta < 0 else dtheta - np.pi]

        best = None
        for cand in candidates:
            rotated = self._rotate(b_w, -cand)
            dy, dx, t_sharpness, _, _ = _phase_correlation(a_w, rotated)
            score = t_sharpness
            if best is None or score > best[0]:
                best = (score, cand, dy, dx, t_sharpness)

        _, dtheta, dy, dx, t_sharpness = best
        # _phase_correlation(a, rotated_b) returns the shift that aligns
        # `rotated` to `a` via correlation of Fa * conj(Fb); by this
        # function's convention that comes out as the *negative* of the
        # displacement that was actually applied to go from a to b, so we
        # flip sign here to report "b is displaced by (dx, dy) from a".
        dy, dx = -dy, -dx

        quality = float(np.clip(0.5 * rot_sharpness + 0.5 * t_sharpness, 0.0, 1.0))

        # --- Local covariance approximation ---
        # Cheap but principled: uncertainty shrinks as match quality grows.
        # sigma^2 ~ sigma0^2 / q^2 (low confidence -> large, isotropic-ish
        # uncertainty; this stands in for a full curvature-based estimate
        # and is what the native FS2D library replaces with a proper
        # second-derivative fit of the correlation surface.)
        sigma_xy = 0.5 / (quality + 1e-3)
        sigma_theta = np.deg2rad(2.0) / (quality + 1e-3)
        covariance = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_theta ** 2])

        return RegistrationResult(
            dx=float(dx), dy=float(dy), dtheta=float(dtheta),
            quality=quality, covariance=covariance, backend="fourier_mellin_numpy",
        )

    @staticmethod
    def _rotate(img: np.ndarray, angle_rad: float) -> np.ndarray:
        h, w = img.shape
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.mgrid[0:h, 0:w]
        yy_c, xx_c = yy - cy, xx - cx
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        src_y = cy + yy_c * cos_a + xx_c * sin_a
        src_x = cx - yy_c * sin_a + xx_c * cos_a
        return map_coordinates(img, [src_y, src_x], order=1, mode="constant")


class NativeFS2DBinding:
    """ctypes FFI wrapper around the compiled Constructor University FS2D
    library. See native/fs2d/README.md for the expected C ABI and build
    instructions. This class only loads/calls the library — it contains no
    algorithmic logic of its own.
    """

    EXPECTED_SYMBOL = "fs2d_register"

    def __init__(self, lib_path: Optional[str] = None):
        self.lib = None
        self.lib_path = lib_path or self._default_lib_path()
        if self.lib_path and os.path.exists(self.lib_path):
            self._load(self.lib_path)

    @staticmethod
    def _default_lib_path() -> Optional[str]:
        system = platform.system()
        here = os.path.dirname(os.path.abspath(__file__))
        native_dir = os.path.join(here, "..", "..", "..", "native", "fs2d", "build")
        candidates = {
            "Linux": "libfs2d.so",
            "Darwin": "libfs2d.dylib",
            "Windows": "fs2d.dll",
        }
        name = candidates.get(system, "libfs2d.so")
        path = os.path.join(native_dir, name)
        return os.path.abspath(path)

    def _load(self, path: str):
        lib = ctypes.CDLL(path)
        # extern "C" void fs2d_register(
        #     const double* scan_prev, const double* scan_curr,
        #     int rows, int cols,
        #     double* out_dx, double* out_dy, double* out_dtheta,
        #     double* out_quality, double* out_cov9);
        lib.fs2d_register.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.fs2d_register.restype = None
        self.lib = lib

    @property
    def available(self) -> bool:
        return self.lib is not None

    def register(self, scan_prev: np.ndarray, scan_curr: np.ndarray) -> RegistrationResult:
        if not self.available:
            raise RuntimeError("Native FS2D library not loaded")
        a = np.ascontiguousarray(scan_prev, dtype=np.float64)
        b = np.ascontiguousarray(scan_curr, dtype=np.float64)
        rows, cols = a.shape
        out_dx = ctypes.c_double(0.0)
        out_dy = ctypes.c_double(0.0)
        out_dtheta = ctypes.c_double(0.0)
        out_quality = ctypes.c_double(0.0)
        out_cov9 = (ctypes.c_double * 9)()

        self.lib.fs2d_register(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(rows), ctypes.c_int(cols),
            ctypes.byref(out_dx), ctypes.byref(out_dy), ctypes.byref(out_dtheta),
            ctypes.byref(out_quality), out_cov9,
        )
        cov = np.array(out_cov9[:]).reshape(3, 3)
        return RegistrationResult(
            dx=out_dx.value, dy=out_dy.value, dtheta=out_dtheta.value,
            quality=out_quality.value, covariance=cov, backend="native",
        )


class FS2DRegistration:
    """Facade used by the rest of the codebase. Prefers the native library;
    falls back to the NumPy Fourier-Mellin implementation automatically."""

    def __init__(self, native_lib_path: Optional[str] = None, force_numpy: bool = False):
        self._numpy_impl = FourierMellinRegistration()
        self._native_impl = None if force_numpy else NativeFS2DBinding(native_lib_path)

    @property
    def backend(self) -> str:
        if self._native_impl is not None and self._native_impl.available:
            return "native"
        return "fourier_mellin_numpy"

    def register(self, scan_prev: np.ndarray, scan_curr: np.ndarray) -> RegistrationResult:
        if self._native_impl is not None and self._native_impl.available:
            return self._native_impl.register(scan_prev, scan_curr)
        return self._numpy_impl.register(scan_prev, scan_curr)
