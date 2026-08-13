#ifndef FS2D_C_API_H
#define FS2D_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Row-major sonar frames scan_prev/scan_curr, size rows*cols each.
 * Writes the estimated relative pose (dx, dy, dtheta), a scalar match
 * quality in [0, 1], and the 3x3 pose covariance (row-major, order
 * dx, dy, dtheta) into the output pointers.
 */
void fs2d_register(
    const double* scan_prev,
    const double* scan_curr,
    int rows,
    int cols,
    double* out_dx,
    double* out_dy,
    double* out_dtheta,
    double* out_quality,
    double* out_cov9
);

#ifdef __cplusplus
}
#endif

#endif /* FS2D_C_API_H */
