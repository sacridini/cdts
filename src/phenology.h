#ifndef PHENOLOGY_H
#define PHENOLOGY_H

#include <vector>
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

// Returns: A 3D numpy array: (19 metrics, n_pixels, max_seasons)
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
    int n_jobs = -1
);

}

#endif // PHENOLOGY_H
