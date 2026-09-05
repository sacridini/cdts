#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace cdts {
namespace utils {

py::array_t<double> compute_medoid(py::array_t<double> input_array, double no_data_value);

} // namespace utils
} // namespace cdts
