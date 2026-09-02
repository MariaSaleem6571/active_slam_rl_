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
plain C ABI below — it's a real, non-stub implementation (not the old
"drop in your code" template), calling directly into
`softDescriptorRegistration`'s public registration methods (see the
detailed rationale in `fs2d_c_api.cpp`'s own header comment: why it
inlines the same computation `registrationOfTwoVoxelsSOFTFast` does
rather than calling that function directly, and why the quality/
covariance outputs are a documented heuristic rather than something
upstream provides).

**This has been written by careful reading of upstream's source, not
verified by an actual successful build.** Compiling it needs OpenCV, PCL,
FFTW3, OpenMP, and CGAL dev packages — that combined dependency closure
(~478 packages on Ubuntu 24.04, including VTK9 and a transitively-pulled
Qt5/JDK toolchain via `libpcl-dev`) is far more than fit in the sandbox
this was developed in. **Build it on your own machine or in Docker and
report back the first compile error, if any** — there's a reasonable
chance of at least one signature mismatch to fix on the first attempt at
something this size, and that's genuinely faster to fix with a real
compiler error in hand than to keep reasoning about blind.

After cloning or pulling this repo, fetch the submodule's files first:
```bash
git submodule update --init --recursive
```

## Build steps

**Recommended: Docker.** The submodule's own `Dockerfile` (at
`native/fs2d/vendor/fourier-soft-2d/Dockerfile`) already assembles the
exact OpenCV/PCL/FFTW3/CGAL/Boost stack this needs — it's a known-working
environment, whereas hand-installing ~478 apt packages on a bare host is
slower and more likely to hit a missing/mismatched package. Two ways to
use it:

- Add this project's `Dockerfile`/`docker-compose.yml` as a build stage
  that also runs `cmake`/`cmake --build` in `native/fs2d/`, using the
  submodule's `Dockerfile` as a reference for which packages to install
  (not attempted here — this project's own Docker setup wasn't written
  with native C++ deps in mind, extending it is its own task).
- Or, simplest to try first: build inside a container started from the
  submodule's own image, with this whole repo bind-mounted in, and just
  run the `cmake`/`cmake --build` commands below inside it.

**Without Docker**, if your machine already has (or can install) these:
```bash
# Ubuntu/Debian
sudo apt-get install cmake libopencv-dev libpcl-dev libfftw3-dev \
    libomp-dev libcgal-dev libeigen3-dev
```

Then, either way:
```bash
cd native/fs2d
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
```

This should produce `libfs2d.so` (or the platform equivalent) in
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
