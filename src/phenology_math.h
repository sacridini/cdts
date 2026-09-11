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
// min_season_length: minimum number of points between start and end
// min_amplitude: minimum value difference between peak and bounding minima
// rtrough_max: threshold for keeping a trough (val - min_y <= rtrough_max * A)
// r_min_filter: minimum threshold for merging adjacent seasons (false positive gap)
// returns: a vector of SeasonSegments containing start, peak, and end indices
std::vector<SeasonSegment> split_growing_seasons(
    const std::vector<double>& smoothed_y,
    int min_season_length = 0,
    double min_amplitude = 0.0,
    double rtrough_max = 0.6,
    double r_min_filter = 0.02);

// HANTS (Harmonic Analysis of Time Series) smoother using Eigen
// y: raw time series
// num_frequencies: number of harmonic frequencies to include
// threshold: outlier rejection threshold
std::vector<double> eigen_hants(
    const std::vector<double>& y, 
    const std::vector<double>& t, 
    int num_frequencies, 
    double threshold);

#endif // PHENOLOGY_MATH_H

// wTSM: Weight updating method in TIMESAT
Eigen::VectorXd eigen_wTSM(
    const Eigen::VectorXd& y, 
    const Eigen::VectorXd& yfit, 
    const Eigen::VectorXd& w, 
    int iter, 
    int nptperyear, 
    double wfact);

