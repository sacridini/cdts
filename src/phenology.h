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
    double sos;
    double eos;
    double los;
    double pop; // peak of season
};

// Evaluate the fitted curve to find metrics
PhenologyMetrics extract_metrics(const Eigen::VectorXd& params, CurveType type, const Eigen::VectorXd& t_segment, ExtractionMethod method = ExtractionMethod::THRESHOLD);

// Batch processing of phenology parameters extraction
// Returns: A tuple of 4 2D numpy arrays: (sos_arr, eos_arr, los_arr, pop_arr)
// Each array has shape [n_pixels, max_seasons]
pybind11::tuple fit_phenology_batch(
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
    int n_jobs = -1
);

}

#endif // PHENOLOGY_H
