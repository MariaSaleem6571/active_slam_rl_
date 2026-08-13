// Rename to fs2d_c_api.cpp once the real FS2D internals are dropped in.
//
// This file only needs to translate between the flat C arrays ctypes hands
// you and whatever richer C++ types your existing FS2D implementation uses
// (e.g. cv::Mat, Eigen::MatrixXd, a custom SonarFrame class, ...). Replace
// the body of fs2d_register() below with a call into your real
// registration routine and fill the output pointers from its result.

#include "fs2d_c_api.h"
#include <cstring>

// #include "YourRealFS2D.hpp"   // <-- the actual Constructor University code

extern "C" void fs2d_register(
    const double* scan_prev,
    const double* scan_curr,
    int rows,
    int cols,
    double* out_dx,
    double* out_dy,
    double* out_dtheta,
    double* out_quality,
    double* out_cov9
) {
    // --- 1. Wrap the raw buffers in whatever type your FS2D expects ---
    // Example if it takes a flat vector + dims:
    //   std::vector<double> a(scan_prev, scan_prev + rows * cols);
    //   std::vector<double> b(scan_curr, scan_curr + rows * cols);
    //   auto result = YourFS2D::registerScans(a, b, rows, cols);

    // --- 2. Fill outputs from the real result ---
    //   *out_dx      = result.dx;
    //   *out_dy      = result.dy;
    //   *out_dtheta  = result.dtheta;
    //   *out_quality = result.quality;
    //   std::memcpy(out_cov9, result.covariance.data(), 9 * sizeof(double));

    // Placeholder identity result so this compiles standalone before you
    // wire in the real algorithm:
    *out_dx = 0.0;
    *out_dy = 0.0;
    *out_dtheta = 0.0;
    *out_quality = 0.0;
    double cov[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::memcpy(out_cov9, cov, 9 * sizeof(double));
}
