#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "landtrendr.h"
#include "ccdc.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ backend for Change Detection Python (cdts)";

    // LandTrendr sub-module
    py::module_ lt = m.def_submodule("landtrendr", "LandTrendr algorithms");

    py::class_<cdts::landtrendr::LandTrendrParams>(lt, "LandTrendrParams")
        .def(py::init<>())
        .def_readwrite("max_segments", &cdts::landtrendr::LandTrendrParams::max_segments)
        .def_readwrite("pval_threshold", &cdts::landtrendr::LandTrendrParams::pval_threshold)
        .def_readwrite("prevent_fast_recovery", &cdts::landtrendr::LandTrendrParams::prevent_fast_recovery)
        .def_readwrite("recovery_threshold", &cdts::landtrendr::LandTrendrParams::recovery_threshold);

    py::class_<cdts::landtrendr::Vertex>(lt, "Vertex")
        .def(py::init<int, double>())
        .def_readwrite("year", &cdts::landtrendr::Vertex::year)
        .def_readwrite("value", &cdts::landtrendr::Vertex::value);

    // Expose the fit_trajectory function to Python
    lt.def("fit_trajectory", &cdts::landtrendr::fit_trajectory, 
           "Run LandTrendr on a single pixel time series",
           py::arg("years"), py::arg("values"), py::arg("params"));

    // Expose the fit_trajectory_batch function to Python
    lt.def("fit_trajectory_batch", &cdts::landtrendr::fit_trajectory_batch, 
           "Run LandTrendr on a batch of pixels (3D array: [Y, X, Time]) with OpenMP",
           py::arg("values_array"), py::arg("years_array"), py::arg("params"), py::arg("no_data_value") = -9999.0, py::arg("n_jobs") = -1);

    // Expose desawtooth function for testing
    lt.def("desawtooth", &cdts::landtrendr::desawtooth,
           "Remove spikes from a time series",
           py::arg("vals"), py::arg("stopat") = 0.9);

    // CCDC sub-module
    py::module_ mc = m.def_submodule("ccdc", "CCDC algorithms");

    py::class_<cdts::ccdc::CCDCParams>(mc, "CCDCParams")
        .def(py::init<>())
        .def_readwrite("min_obs", &cdts::ccdc::CCDCParams::min_obs)
        .def_readwrite("conseq_anom", &cdts::ccdc::CCDCParams::conseq_anom)
        .def_readwrite("chi2_prob_threshold", &cdts::ccdc::CCDCParams::chi2_prob_threshold);

    py::class_<cdts::ccdc::CCDCSegment>(mc, "CCDCSegment")
        .def(py::init<>())
        .def_readwrite("t_start", &cdts::ccdc::CCDCSegment::t_start)
        .def_readwrite("t_end", &cdts::ccdc::CCDCSegment::t_end)
        .def_readwrite("t_break", &cdts::ccdc::CCDCSegment::t_break)
        .def_readwrite("coefs", &cdts::ccdc::CCDCSegment::coefs)
        .def_readwrite("rmse", &cdts::ccdc::CCDCSegment::rmse)
        .def_readwrite("magnitude", &cdts::ccdc::CCDCSegment::magnitude);

    mc.def("fit_ccdc", &cdts::ccdc::fit_ccdc,
           "Run CCDC on a single pixel time series",
           py::arg("dates"), py::arg("values"), py::arg("qa"),
           py::arg("params") = cdts::ccdc::CCDCParams());

    // Utilities sub-module
    py::module_ utils = m.def_submodule("utils", "Geospatial utilities and processing");
    
    utils.def("compute_medoid", &cdts::utils::compute_medoid,
           "Computes the multidimensional medoid composite over the time axis",
           py::arg("input_array"), py::arg("no_data_value") = -9999.0);
}

