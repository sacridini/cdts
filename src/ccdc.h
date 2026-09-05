#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace cdts {
namespace ccdc {

// Represents a fitted CCDC harmonic model segment
struct CCDCSegment {
    int t_start;         // Start date (ordinal day)
    int t_end;           // End date
    int t_break;         // Break date (when change was detected)
    std::vector<double> rmse;
    std::vector<std::vector<double>> coefs; // [band][6 coefs]
    std::vector<double> magnitude;          // Magnitude of change per band
};

struct CCDCParams {
    int min_obs = 12;
    int conseq_anom = 3;  // CCDC defaults to 3, COLD to 6
    double chi2_prob_threshold = 0.99;
};

// Main CCDC logic that operates on multiple bands simultaneously
std::vector<CCDCSegment> fit_ccdc(const std::vector<int>& dates,
                                  const std::vector<std::vector<double>>& bands,
                                  const std::vector<int>& qa,
                                  CCDCParams params = CCDCParams());

// Batch fit function for arrays
pybind11::tuple fit_ccdc_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Bands, Time]
    pybind11::array_t<int> qa_array,        // Shape: [Y, X, Time]
    pybind11::array_t<int> dates_array,     // Shape: [Time]
    CCDCParams params,
    int max_segments = 6,
    bool return_coefs = true,
    int n_jobs = -1);

} // namespace ccdc
} // namespace cdts
