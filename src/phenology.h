#ifndef PHENOLOGY_H
#define PHENOLOGY_H

#include <vector>
#include <tuple>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "phenology_math.h"
#include "phenology_curves.h"

namespace phenology {

enum class ExtractionMethod {
    THRESHOLD,
    DERIVATIVE,
    GU,
    KLOSTERMAN
};

struct PhenologyMetrics {
    double trs2_sos, trs2_eos;
    double trs5_sos, trs5_eos;
    double trs6_sos, trs6_eos;
    double der_sos, der_pos, der_eos;
    double gu_ud, gu_sd, gu_dd, gu_rd;
    double zhang_greenup, zhang_maturity, zhang_senescence, zhang_dormancy;
    double los;
    double pop;
};

// Evaluate the fitted curve to find metrics
PhenologyMetrics extract_metrics(const Eigen::VectorXd& params, CurveType type, const Eigen::VectorXd& t_segment);

// Weighted R^2 and RMSE of the fitted curve against the observed segment,
// using the final iterative-reweighting weights (w_seg) so that points the
// fit already treated as outliers/low-quality don't dominate the score.
// Mirrors phenofit's get_GOF() diagnostic (R/get_GOF.R), which is used there
// to flag unreliable curve fits instead of silently trusting every season.
struct GoodnessOfFit {
    double r2;
    double rmse;
};
GoodnessOfFit compute_gof(const Eigen::VectorXd& y_seg, const Eigen::VectorXd& yfit_seg, const Eigen::VectorXd& w_seg);

// Returns: A 3D numpy array: (21 metrics, n_pixels, max_seasons)
// Metrics 0-18 are the phenology dates/derived values (see PhenologyMetrics);
// metric 19 is R2 and metric 20 is RMSE of the fitted curve for that season.
pybind11::array_t<double> fit_phenology_batch(
    pybind11::array_t<double> values_array, // 2D: [n_pixels, n_time]
    pybind11::array_t<double> dates_array,  // 1D: [n_time]
    int curve_type_int,
    int extraction_method,
    int max_seasons = 2,
    double whittaker_lambda = 10.0,
    bool apply_whittaker = true,
    bool apply_hants = false,
    int hants_frequencies = 3,
    double hants_threshold = 0.1,
    int min_season_length = 0,
    double min_amplitude = 0.0,
    double min_pixel_amplitude = 0.0,
    double rtrough_max = 0.6,
    double r_min_filter = 0.02,
    int n_jobs = -1,
    // Optional 2D [n_pixels, n_time] array of per-observation reliability
    // weights in [0, 1] (e.g. decoded from a MODIS/Sentinel-2 QC band via
    // cdts.qc). When provided, feeds the Whittaker/HANTS smoothers and
    // seeds the iterative wTSM curve-fit reweighting, instead of every
    // observation starting at an implicit weight of 1. Pass None/omit to
    // keep the previous behaviour.
    pybind11::object weights_array = pybind11::none(),
    // See split_growing_seasons()'s retry_on_empty.
    bool season_retry = true
);

// Thin, directly-testable wrapper around split_growing_seasons() that returns
// plain (start_idx, peak_idx, end_idx) tuples instead of requiring a full
// fit_phenology_batch() run (smoothing + curve fitting) just to inspect
// season-boundary detection. Exposed to Python for unit tests only.
// dates: optional calendar date/DOY per point in `y` (same length/indexing);
//   when omitted, defaults to 0, 1, 2, ... (i.e. min_season_length behaves as
//   a plain observation count, matching tests that don't care about real
//   calendar spacing).
std::vector<std::tuple<int, int, int>> debug_split_seasons(
    std::vector<double> y,
    std::optional<std::vector<double>> dates = std::nullopt,
    int min_season_length = 0,
    double min_amplitude = 0.0,
    double rtrough_max = 0.6,
    double r_min_filter = 0.02,
    bool retry_on_empty = true
);

}

#endif // PHENOLOGY_H
