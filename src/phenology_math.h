#ifndef PHENOLOGY_MATH_H
#define PHENOLOGY_MATH_H

#include <vector>
#include <optional>
#include <Eigen/Core>

// Struct to represent a growing season segment
struct SeasonSegment {
    int start_idx;
    int peak_idx;
    int end_idx;
};

// Fast Whittaker smoother using Eigen
// y: raw time series
// weights: optional weights for each point
// lambda: smoothing parameter
std::vector<double> eigen_whittaker(
    const std::vector<double>& y, 
    const std::optional<std::vector<double>>& weights, 
    double lambda);

// Savitzky-Golay smoother using Eigen
// y: raw time series
// window_size: size of the smoothing window (must be odd)
// poly_order: polynomial order
std::vector<double> eigen_savgol(
    const std::vector<double>& y, 
    int window_size, 
    int poly_order);

// Split growing seasons based on local minima and maxima
// smoothed_y: smoothed time series data
// dates: calendar date/DOY for each point in smoothed_y (same length, same
//   indexing) - used to measure season duration in real elapsed time rather
//   than observation count, so results don't depend on the sensor's revisit
//   cadence (daily, 8-day, 16-day, irregular, ...).
// min_season_length: minimum elapsed calendar duration (dates[end_idx] -
//   dates[start_idx]) between the start and end of a season, in the same
//   units as `dates` (days, if `dates` holds day-of-year/day-since-epoch
//   values).
// min_amplitude: minimum value difference between peak and bounding minima
// rtrough_max: threshold for keeping a trough (val - min_y <= rtrough_max * A)
// r_min_filter: minimum threshold for merging adjacent seasons (false positive gap)
// retry_on_empty: mirrors phenofit's season_mov behaviour (see R/season_mov.R,
//   "if have no brks, try to decrease r_max"): when the first pass finds zero
//   raw seasons, retry once with a relaxed (more permissive) rtrough_max before
//   giving up on the pixel.
// returns: a vector of SeasonSegments containing start, peak, and end indices
std::vector<SeasonSegment> split_growing_seasons(
    const std::vector<double>& smoothed_y,
    const std::vector<double>& dates,
    int min_season_length = 0,
    double min_amplitude = 0.0,
    double rtrough_max = 0.6,
    double r_min_filter = 0.02,
    bool retry_on_empty = true);

// HANTS (Harmonic Analysis of Time Series) smoother using Eigen
// y: raw time series
// num_frequencies: number of harmonic frequencies to include
// threshold: outlier rejection threshold
// initial_weights: optional per-observation reliability weights in [0, 1]
//   (e.g. derived from a QC/QA band). Points with weight 0 start already
//   excluded from the harmonic fit instead of only being down-weighted by
//   the residual-based rejection.
std::vector<double> eigen_hants(
    const std::vector<double>& y,
    const std::vector<double>& t,
    int num_frequencies,
    double threshold,
    const std::optional<std::vector<double>>& initial_weights = std::nullopt);

#endif // PHENOLOGY_MATH_H

// wTSM: Weight updating method in TIMESAT
Eigen::VectorXd eigen_wTSM(
    const Eigen::VectorXd& y, 
    const Eigen::VectorXd& yfit, 
    const Eigen::VectorXd& w, 
    int iter, 
    int nptperyear, 
    double wfact);

