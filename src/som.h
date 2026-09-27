#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

namespace cdts {
namespace som {

// Codes shared with cdts/ai/som.py (keep in sync).
// Learning-rate decay: 0 asymptotic_decay, 1 inverse_decay_to_zero, 2 linear_decay_to_zero
// Sigma decay:         0 asymptotic_decay, 1 inverse_decay_to_one,  2 linear_decay_to_one
// Neighborhood:        0 gaussian, 1 mexican_hat, 2 bubble, 3 triangle

// Online (sample-by-sample) SOM, a line-by-line port of MiniSom.train().
// `weights` (x, y, D) is updated in place. `order` holds the sample index
// picked at each step (length num_iteration), or one epoch's ordering
// (length N) when use_epochs is true.
void train_online(
    pybind11::array_t<double, pybind11::array::c_style> weights_array,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> data_array,
    pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast> order_array,
    int num_iteration,
    bool use_epochs,
    double learning_rate,
    double sigma,
    int lr_decay,
    int sigma_decay,
    int neighborhood,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> xx_array,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> yy_array);

// Batch SOM (Kohonen, 2013), port of MiniSom.train_batch_offline().
// `weights` (x, y, D) is updated in place.
void train_batch(
    pybind11::array_t<double, pybind11::array::c_style> weights_array,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> data_array,
    int num_iteration,
    double learning_rate,
    double sigma,
    int lr_decay,
    int sigma_decay,
    int neighborhood,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> xx_array,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> yy_array,
    int n_jobs);

// Flat index (i * y + j) of the best matching unit of every sample.
pybind11::array_t<int> predict_bmus(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> data_array,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> weights_array,
    int n_jobs);

} // namespace som
} // namespace cdts
