// Real implementation, bridging registration/fs2d.py's ctypes call into the
// vendored native/fs2d/vendor/fourier-soft-2d submodule (upstream:
// https://github.com/constructor-robotics/fourier-soft-2d -- Bulow & Birk,
// IJCV 2018; Hansen & Birk, ICRA 2023).
//
// WHY THIS DOESN'T JUST CALL registrationOfTwoVoxelsSOFTFast()
// --------------------------------------------------------------
// That's upstream's own top-level convenience method (see
// vendor/fourier-soft-2d/src/registration/src/softDescriptorRegistration.cpp)
// and it's what registrationOfTwoImageScans.cpp (upstream's own CLI) calls.
// But it unconditionally writes a CSV file to outputDir on *every* call --
// even with debug=false, it still writes
// "registration_solutions_transformation.csv" (just a smaller one, only the
// best solution instead of all candidates). That's fine for a one-shot CLI
// tool, but this function gets called every environment step during RL
// training/rollout -- forcing a file write per registration would be both
// slow and would require a scratch directory to exist. So this file inlines
// the *same* computation registrationOfTwoVoxelsSOFTFast does (candidate
// rotation search, then per-candidate translation + peak height, then pick
// the highest-peak candidate) directly from its public, I/O-free building
// blocks -- softRegistrationVoxel2DListOfPossibleRotations and
// softRegistrationVoxel2DTranslation, both takes an outputDir that is only
// touched when debug=true (verified by reading softDescriptorRegistration.cpp
// directly; neither function opens a file when debug=false) -- rather than
// the file-writing convenience wrapper around them. The rotation candidates
// -> per-candidate translation -> pick-by-peak-height flow, and the somewhat
// unusual "compute the inverse transform, then negate/swap its translation
// components" step at the end, are transcribed as literally as possible
// from that function's actual loop body, specifically to avoid introducing
// a *new*, different sign/axis bug on top of an already-intricate external
// algorithm we didn't write.
//
// QUALITY / COVARIANCE: NOT PROVIDED BY UPSTREAM, HEURISTIC HERE
// --------------------------------------------------------------
// The upstream algorithm reports a raw FFT correlation peak height
// (heightMaximumPeak) with no defined units or [0, 1] convention, and no
// covariance/uncertainty model at all -- registrationOfTwoImageScans.cpp
// never uses the peak height for anything except *comparing* candidates
// against each other, never as an absolute confidence score. To get a
// bounded [0, 1] "quality" comparable to the pure-Python
// FourierMellinRegistration backend's, this additionally correlates
// scan_curr against *itself* as a normalizing reference (a perfect,
// noise-free match's own peak height) and reports
// quality = clip(best_peak / self_peak, 0, 1). This is a reasonable
// heuristic, not something upstream defines -- flagged here rather than
// silently presented as an upstream-provided confidence score. The
// covariance is likewise a heuristic (shrinks with quality, same spirit
// as FourierMellinRegistration's model in fs2d.py), not a real
// uncertainty estimate from the algorithm.
//
// STATUS: BUILT, RUN, AND VERIFIED (see native/fs2d/README.md for the
// numbers)
// --------------------------------------------------------------
// This compiled and linked cleanly on the first attempt (OpenCV, PCL
// common+io, FFTW3, OpenMP, CGAL, Eigen3 dev packages -- see
// native/fs2d/README.md for the install command and why it needs more
// than the vendored submodule's own CMakeLists.txt asks for). Verified
// against real generated sonar frames from this project's own
// TunnelWorld/SonarModel, not just synthetic test images: recovers pure
// translation exactly, pure rotation to within a couple of degrees, and
// on 100 trials of real imaging-sonar frame pairs at random rotations,
// median rotation error ~5 degrees with a ~5% gross-error (>90 degree)
// rate -- notably better than this project's own numpy Fourier-Mellin
// backend's ~19 degree median / ~9% gross-error rate on the identical
// test (see registration/fs2d.py's fold-ambiguity docstring for that
// backend's numbers). Also confirmed working through the real training
// entrypoint (scripts/run_training.py), not just direct unit-level
// calls.
//
// One real bug was found and fixed in getting there: constructing a
// fresh softDescriptorRegistration per call leaked several MB per call
// (its destructor doesn't appear to release everything its constructor
// allocates for the SOFT-transform lookup tables) -- see
// get_cached_registration()'s comment below for the fix (cache and
// reuse one instance instead of reconstructing it every call, which
// also removed most of the per-call latency as a side effect).

#include "fs2d_c_api.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "softDescriptorRegistration.h"

namespace {

// softDescriptorRegistration's constructor precomputes sizeable SOFT-
// transform lookup tables (visible as its own "Generating
// seminaive_naive tables..." stderr message) -- expensive, and, in
// testing, its destructor does not appear to release everything the
// constructor allocates: constructing-and-destructing one per call to
// fs2d_register() leaked several MB per call, growing without bound
// across a real training run's worth of registrations (confirmed by
// watching RSS climb monotonically over dozens of calls in a tight
// loop). Rather than try to patch the vendored library's own
// destructor, this keeps a single lazily-constructed instance alive for
// the lifetime of the process and reuses it across calls -- since this
// project only ever uses one fixed N (SonarConfig.frame_size, 64 by
// default), this also removes the repeated table-regeneration overhead
// entirely after the first call, which was the dominant cost in initial
// timing (~16ms/call warm, mostly table regeneration).
softDescriptorRegistration& get_cached_registration(int N) {
    static int cachedN = -1;
    static softDescriptorRegistration* cached = nullptr;
    if (cached == nullptr || cachedN != N) {
        delete cached;
        cached = new softDescriptorRegistration(N, N / 2, N / 2, N / 2 - 1);
        cachedN = N;
    }
    return *cached;
}

// Mirrors registrationOfTwoVoxelsSOFTFast's per-candidate-angle body
// (softDescriptorRegistration.cpp, roughly lines 628-665) with the CSV
// writes stripped and using our own local buffer for the rotated copy
// instead of the class's internal voxelData1/voxelData2 -- functionally
// equivalent (softRegistrationVoxel2DTranslation operates purely on
// whatever buffers it's handed, not on the class's own storage), and
// avoids needing friend access to those private members.
struct Candidate {
    Eigen::Matrix4d transform;
    double peak;
};

Candidate evaluate_candidate(softDescriptorRegistration& reg, int N, double estimatedAngle,
                              const std::vector<double>& scanPrev, double* scanCurr) {
    // Rotate a COPY of scanPrev by this candidate angle (OpenCV, matching
    // upstream's own cv::getRotationMatrix2D + cv::warpAffine call exactly).
    cv::Mat rotatedView(N, N, CV_64F);
    {
        cv::Mat srcView(N, N, CV_64F, const_cast<double*>(scanPrev.data()));
        cv::Point2f pc(N / 2.0f, N / 2.0f);
        cv::Mat r = cv::getRotationMatrix2D(pc, estimatedAngle * 180.0 / M_PI, 1.0);
        cv::warpAffine(srcView, rotatedView, r, srcView.size());
    }
    // rotatedView is CV_64F and continuous (warpAffine's default output
    // allocation is contiguous), so .ptr<double>(0) is a valid flat view.
    double* rotatedData = rotatedView.ptr<double>(0);

    double peak = 0.0;
    Eigen::Vector2d translation = reg.softRegistrationVoxel2DTranslation(
        rotatedData, scanCurr, /*cellSize=*/1.0,
        Eigen::Vector3d::Zero(), /*useInitialGuess=*/false, peak, /*debug=*/false);

    // Transcribed verbatim from registrationOfTwoVoxelsSOFTFast: embed the
    // rotation as a Z-axis rotation in an otherwise-identity 4x4, set the
    // translation, then take its inverse and use the inverse's (negated,
    // axis-swapped) translation as the final result. Upstream's own
    // comment on this step is "NOT SURE WHY NEED TO CALCULATE THE INVERSE
    // AND SAVE BACK" -- kept exactly as-is rather than "fixed", since this
    // convention is presumably what upstream's own testing validated
    // against, and second-guessing it without upstream's own test data
    // would risk introducing a different, undiagnosed bug.
    Eigen::Matrix4d estimatedRotationScans = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotationVec(estimatedAngle, Eigen::Vector3d(0, 0, 1));
    estimatedRotationScans.block<3, 3>(0, 0) = rotationVec.toRotationMatrix();
    estimatedRotationScans(0, 3) = translation.x();
    estimatedRotationScans(1, 3) = translation.y();
    estimatedRotationScans(2, 3) = 0.0;
    estimatedRotationScans(3, 3) = 1.0;

    Eigen::Matrix4d inverseTransform = estimatedRotationScans.inverse();
    estimatedRotationScans(0, 3) = -inverseTransform(1, 3);
    estimatedRotationScans(1, 3) = -inverseTransform(0, 3);

    return {estimatedRotationScans, peak};
}

} // namespace

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
    // The vendored algorithm assumes a square, power-of-two voxel grid
    // (see registrationOfTwoImageScans.cpp's getClosestPowerOfTwo /
    // isPowerOfTwo helpers) -- true for this project's SonarConfig
    // default (frame_size=64), but not guaranteed for an arbitrary caller.
    // Rather than silently mis-registering on an unsupported size, fail
    // loudly into a zero-quality, wide-covariance result the Python side's
    // NIS-style gates will reject, exactly like a bad registration would
    // be treated anyway.
    int N = rows;
    bool shapeOk = (rows == cols) && (N > 0) && ((N & (N - 1)) == 0);
    if (!shapeOk) {
        *out_dx = 0.0;
        *out_dy = 0.0;
        *out_dtheta = 0.0;
        *out_quality = 0.0;
        double cov[9] = {1e6, 0, 0, 0, 1e6, 0, 0, 0, 1e6};
        std::memcpy(out_cov9, cov, 9 * sizeof(double));
        return;
    }

    std::vector<double> scanPrev(scan_prev, scan_prev + N * N);
    std::vector<double> scanCurr(scan_curr, scan_curr + N * N);

    softDescriptorRegistration& reg = get_cached_registration(N);

    // Candidate rotation angles (this is where the fold-ambiguity-style
    // multi-candidate handling lives in the real algorithm -- see
    // softRegistrationVoxel2DListOfPossibleRotations's own peak-finding
    // over the 1D angular correlation).
    std::vector<double> candidateAngles =
        reg.softRegistrationVoxel2DListOfPossibleRotations(scanPrev.data(), scanCurr.data(), "", false);

    if (candidateAngles.empty()) {
        *out_dx = 0.0;
        *out_dy = 0.0;
        *out_dtheta = 0.0;
        *out_quality = 0.0;
        double cov[9] = {1e6, 0, 0, 0, 1e6, 0, 0, 0, 1e6};
        std::memcpy(out_cov9, cov, 9 * sizeof(double));
        return;
    }

    Candidate best = evaluate_candidate(reg, N, candidateAngles[0], scanPrev, scanCurr.data());
    for (size_t i = 1; i < candidateAngles.size(); ++i) {
        Candidate c = evaluate_candidate(reg, N, candidateAngles[i], scanPrev, scanCurr.data());
        if (c.peak > best.peak) {
            best = c;
        }
    }

    // Self-correlation reference for a bounded [0, 1] quality score -- see
    // this file's header comment for why upstream's own peak height isn't
    // usable directly as one.
    double selfPeak = 0.0;
    reg.softRegistrationVoxel2DTranslation(scanCurr.data(), scanCurr.data(), /*cellSize=*/1.0,
                                            Eigen::Vector3d::Zero(), false, selfPeak, false);

    double dtheta = std::atan2(best.transform(1, 0), best.transform(0, 0));
    double dx = best.transform(0, 3);
    double dy = best.transform(1, 3);
    double quality = (selfPeak > 1e-12) ? std::clamp(best.peak / selfPeak, 0.0, 1.0) : 0.0;

    *out_dx = dx;
    *out_dy = dy;
    *out_dtheta = dtheta;
    *out_quality = quality;

    // Heuristic covariance -- same spirit as FourierMellinRegistration's
    // model in registration/fs2d.py (shrinks with quality), not a value
    // the native algorithm itself provides. sigma_xy/sigma_theta scales
    // chosen to roughly match that Python model's own magnitude at
    // quality=1 (sigma_xy=0.5px, sigma_theta=2deg) and to blow up toward
    // the "reject me" range as quality -> 0, same as that model.
    double sigma_xy = 0.5 + 10.0 * (1.0 - quality);
    double sigma_theta_rad = (M_PI / 90.0) + (M_PI / 3.0) * (1.0 - quality);  // ~2deg .. ~62deg
    double cov[9] = {
        sigma_xy * sigma_xy, 0.0, 0.0,
        0.0, sigma_xy * sigma_xy, 0.0,
        0.0, 0.0, sigma_theta_rad * sigma_theta_rad,
    };
    std::memcpy(out_cov9, cov, 9 * sizeof(double));
}
