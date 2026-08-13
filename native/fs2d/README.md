# Linking the native FS2D library (C/C++)

`src/active_slam_rl/registration/fs2d.py` calls into the real Constructor
University FS2D registration library through `ctypes`, **if** it finds a
compiled shared library at:

```
native/fs2d/build/libfs2d.so      # Linux
native/fs2d/build/libfs2d.dylib   # macOS
native/fs2d/build/fs2d.dll        # Windows
```

If it is not found, `FS2DRegistration` silently falls back to the pure
NumPy Fourier-Mellin implementation (`FourierMellinRegistration`) so you can
develop and train right now without the native code.

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

## Build steps (once you have the real FS2D source)

1. Drop the Constructor University FS2D sources into `native/fs2d/src/`.
2. Wrap your existing top-level registration call in `fs2d_c_api_template.cpp`
   (rename to `fs2d_c_api.cpp`) so it matches the signature above.
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
