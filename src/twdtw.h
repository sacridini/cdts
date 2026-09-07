#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace cdts {
namespace twdtw {

struct TWDTWParams {
    double alpha = 0.1;
    double beta = 0.05;
    double gamma = 50.0;
};

// Calculates the TWDTW distance between a time series and a pattern
double fit_twdtw(const std::vector<double>& ts_values,
                 const std::vector<int>& ts_dates,
                 const std::vector<double>& pattern_values,
                 const std::vector<int>& pattern_dates,
                 const TWDTWParams& params = TWDTWParams());

// Batch version for applying TWDTW to a 3D array (cube)
pybind11::array_t<double> fit_twdtw_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Time]
    pybind11::array_t<int> dates_array,     // Shape: [Time]
    pybind11::array_t<double> pattern_values_array, // Shape: [PatternTime]
    pybind11::array_t<int> pattern_dates_array,     // Shape: [PatternTime]
    const TWDTWParams& params,
    int n_jobs = -1);

} // namespace twdtw
} // namespace cdts
