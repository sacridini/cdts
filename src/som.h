#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace cdts {
namespace som {

pybind11::array_t<double> train_som_batch(
    pybind11::array_t<double> data_array,
    int x, int y, 
    int num_iters, 
    double initial_sigma,
    int n_jobs,
    int random_seed);

pybind11::array_t<int> predict_bmus(
    pybind11::array_t<double> data_array,
    pybind11::array_t<double> weights_array,
    int n_jobs);

} // namespace som
} // namespace cdts
