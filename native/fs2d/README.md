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

## The upstream source is now vendored as a submodule

The real algorithm's source is checked out at
`native/fs2d/vendor/fourier-soft-2d` (upstream:
https://github.com/constructor-robotics/fourier-soft-2d — Bülow & Birk,
"Scale-Free Registrations in 3D... Fourier-Mellin-SOFT transforms," IJCV
2018; Hansen & Birk, "Using Registration with Fourier-SOFT in 2D (FS2D)
for Robust Scan Matching of Sonar Range Data," ICRA 2023).

**After cloning or pulling this repo, fetch the submodule's files:**
```bash
git submodule update --init --recursive
```

**What's NOT done yet -- this is real follow-up work, not a detail:**
upstream ships as a Dockerized FastAPI microservice (build via its own
`Dockerfile`/`docker-compose.yml`, depends on OpenCV, PCL, FFTW3, CGAL,
Eigen3, Boost, and its own vendored `soft20` SOFT-transform library) built
around file-based I/O (read a PNG, write a CSV/PNG), not a linkable
library exposing the plain C ABI below. The two most likely files to
wrap are `vendor/fourier-soft-2d/src/registration/src/
softDescriptorRegistration.cpp` (the `softDescriptorRegistration` class --
the actual registration algorithm) and `vendor/fourier-soft-2d/src/
registrationOfTwoImageScans.cpp` (its existing CLI entry point, so its
exact call pattern is already right there to read). Bridging the two
means: read `softDescriptorRegistration`'s real method signatures, write
`fs2d_c_api.cpp` (rename from the `_template` version below) calling into
it directly from a `double*` buffer instead of loading an image file, and
get CMake linking against upstream's dependency list. None of that is
done here -- only the submodule pointer.

## Required C ABI

Whatever the internal C++ implementation looks like, expose exactly one
`extern "C"` entry point with this signature (a template is in
`fs2d_c_api.h` / `fs2d_c_api_template.cpp` in this folder):

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
calls — nothing else about your C++ internals needs to be exposed.

## Build steps (once the C API wrapper exists)

1. The FS2D sources are now at `native/fs2d/vendor/fourier-soft-2d`
   (submodule; run `git submodule update --init --recursive` first if
   you haven't). This step used to say "drop the sources into
   `native/fs2d/src/`" -- that's superseded now that they're vendored
   properly, but writing `fs2d_c_api.cpp` (step 2) is still not done.
2. Write `fs2d_c_api.cpp` (based on `fs2d_c_api_template.cpp`) so it
   matches the signature above, calling into
   `vendor/fourier-soft-2d/src/registration/src/softDescriptorRegistration.cpp`'s
   actual API -- see this file's "What's NOT done yet" section above for
   the specific files to start from.
3. Build:

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

## Sanity-checking the swap

`tests/test_registration.py` includes a parametrized test that runs the
same pair of synthetic scans through both backends and checks they agree
to within a tolerance — run it after linking the native library to make
sure the swap didn't silently change behavior:

```bash
pytest tests/test_registration.py -v
```
