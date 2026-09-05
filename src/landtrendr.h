#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace cdts {
namespace landtrendr {

// Struct to hold parameters for LandTrendr
struct LandTrendrParams {
    int max_segments = 6;
    double pval_threshold = 0.05;
    bool prevent_fast_recovery = true;     // Reject biologically impossible rapid recoveries
    double recovery_threshold = 0.25;      // Max recovery rate per year
};

// Struct to hold output vertices
struct Vertex {
    int year;
    double value;
};

// Core function to run LandTrendr on a single pixel time series
std::vector<Vertex> fit_trajectory(const std::vector<int>& years, 
                                   const std::vector<double>& values, 
                                   const LandTrendrParams& params);

// Desawtooth function
std::vector<double> desawtooth(const std::vector<double>& vals, double stopat = 0.9);

// New batch fit function
pybind11::tuple fit_trajectory_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Time]
    pybind11::array_t<int> years_array,     // Shape: [Time]
    LandTrendrParams params,
    double no_data_value = -9999.0);

} // namespace landtrendr
} // namespace cdts
