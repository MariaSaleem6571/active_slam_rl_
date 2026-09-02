# Linking the native FS2D library (C/C++)

`src/active_slam_rl/registration/fs2d.py` calls into the real Constructor
Robotics FS2D registration library through `ctypes`, **if** it finds a
compiled shared library at:

```
native/fs2d/build/libfs2d.so      # Linux
native/fs2d/build/libfs2d.dylib   # macOS
native/fs2d/build/fs2d.dll        # Windows
```

If it is not found, `FS2DRegistration` silently falls back to the pure
NumPy Fourier-Mellin implementation (`FourierMellinRegistration`) so you can
develop and train right now without the native code.

## Status

The upstream algorithm is vendored as a submodule at
`native/fs2d/vendor/fourier-soft-2d` (upstream:
https://github.com/constructor-robotics/fourier-soft-2d — Bülow & Birk,
IJCV 2018; Hansen & Birk, ICRA 2023). `fs2d_c_api.cpp` bridges it to the
plain C ABI below, calling directly into `softDescriptorRegistration`'s
public registration methods (see the detailed rationale in
`fs2d_c_api.cpp`'s own header comment: why it inlines the same
computation `registrationOfTwoVoxelsSOFTFast` does rather than calling
that function directly, and why the quality/covariance outputs are a
documented heuristic rather than something upstream provides).

**This has been built and verified, not just written.** Compiled cleanly
on the first attempt (OpenCV, PCL common+io, FFTW3, OpenMP, CGAL, Eigen3
dev packages). Checked against:

- Synthetic test images: exact translation recovery, rotation recovery
  to within a couple of degrees.
- 100 trials of **real** generated imaging-sonar frames (this project's
  own `TunnelWorld`/`SonarModel`) at random rotations: median error ~5
  degrees, ~5% gross-error (>90 degree) rate — notably better than this
  project's own numpy `FourierMellinRegistration` backend's ~19 degree
  median / ~9% gross-error rate on the identical test (see
  `registration/fs2d.py`'s fold-ambiguity docstring for that backend's
  numbers).
- The real training entrypoint end-to-end (`scripts/run_training.py`,
  not just direct calls): 2048 real timesteps at 67 fps with the native
  backend active, no crashes, normal CSV logging.
- All of `tests/test_registration.py`, including the native-vs-numpy
  comparison test, which now actually runs (was skip-only before this
  was built).

**One real bug found and fixed along the way:** constructing a fresh
`softDescriptorRegistration` per call leaked several MB per call (its
destructor doesn't release everything its constructor allocates for the
SOFT-transform lookup tables) — confirmed by watching memory climb
unboundedly across repeated calls, and this is what caused an
out-of-memory crash partway through a full test suite run before the
fix. `fs2d_c_api.cpp` now caches and reuses a single instance across
calls instead (see `get_cached_registration()`), which fixed the leak
and, as a bonus, dropped per-call latency from ~16.5ms to ~9.75ms once
warm (no more repeated table regeneration).

Because of this verified accuracy improvement, `EnvConfig.force_numpy_fs2d`
now defaults to `false` (native preferred when built, automatic fallback
to numpy if not) — see that field's comment in `env/sim_env.py`.

After cloning or pulling this repo, fetch the submodule's files first:
```bash
git submodule update --init --recursive
```

## Build steps

**Ubuntu/Debian**, this is what was actually used and verified:
```bash
sudo apt-get install --no-install-recommends cmake libopencv-dev \
    libpcl-dev libfftw3-dev libomp-dev libcgal-dev libeigen3-dev
```
Fair warning: this pulls in a large transitive dependency closure via
`libpcl-dev` (VTK9, and a Qt5/JDK toolchain that isn't optional even
with `--no-install-recommends` on Ubuntu 24.04's packaging) — expect on
the order of a few GB of disk space and several hundred packages.
**Docker is the better default** if you'd rather not do that on your
main machine: the submodule's own `Dockerfile` (at
`native/fs2d/vendor/fourier-soft-2d/Dockerfile`) already has this exact
stack assembled and known-working.

Then, either way:
```bash
cd native/fs2d
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
```

This produces `libfs2d.so` (or the platform equivalent) in
`native/fs2d/build/`. Nothing on the Python side needs to change — the next
time you construct `FS2DRegistration()`, `.backend` will report `"native"`
instead of `"fourier_mellin_numpy"`, and every downstream module (mapping,
state encoder, reward) keeps working unmodified, since they only depend on
the `RegistrationResult` dataclass, not on which backend produced it.

## Required C ABI

```c
extern "C" void fs2d_register(
    const double* scan_prev,   // row-major, size rows*cols
    const double* scan_curr,   // row-major, size rows*cols
    int rows, int cols,
    double* out_dx,            // translation, x (columns), pixels
    double* out_dy,            // translation, y (rows), pixels
    double* out_dtheta,        // rotation, radians
    double* out_quality,       // q_t in [0, 1]
    double* out_cov9           // Sigma_reg_t, row-major 3x3 (dx, dy, dtheta)
);
```

This is the one function `NativeFS2DBinding` in `fs2d.py` looks up and
calls — nothing else about the C++ internals needs to be exposed.
`fs2d_c_api.cpp` currently only handles a square, power-of-two `rows`/
`cols` (matching `SonarConfig.frame_size=64`'s default, and matching
upstream's own assumption baked into its SOFT-transform bandwidth
parameters) — a non-power-of-two size returns a zero-quality result
rather than mis-registering silently.

## Sanity-checking the swap

`tests/test_registration.py` includes a parametrized test that runs the
same pair of synthetic scans through both backends and checks they agree
to within a tolerance — run it after linking the native library to make
sure the swap didn't silently change behavior:

```bash
pytest tests/test_registration.py -v
```

Given the native backend's very different internal algorithm (SOFT
spherical-harmonic correlation vs. this project's own Fourier-Mellin/
log-polar approach), don't expect bit-identical output — the test's
tolerance already accounts for that; if it fails, that's more likely a
real discrepancy worth looking at than a false alarm.
