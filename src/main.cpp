#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "landtrendr.h"
#include "ccdc.h"
#include "utils.h"
#include "twdtw.h"
#include "som.h"
#include "phenology.h"

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

    // Expose the fit_ccdc_batch function to Python
    mc.def("fit_ccdc_batch", &cdts::ccdc::fit_ccdc_batch, "Run CCDC on a batch of pixels with OpenMP",
             py::arg("values_array"), py::arg("qa_array"), py::arg("dates_array"), 
             py::arg("params"), py::arg("max_segments") = 6, py::arg("return_coefs") = true, py::arg("n_jobs") = -1);

    // Utilities sub-module
    py::module_ utils = m.def_submodule("utils", "Geospatial utilities and processing");
    
    utils.def("compute_medoid", &cdts::utils::compute_medoid,
           "Computes the multidimensional medoid composite over the time axis",
           py::arg("input_array"), py::arg("no_data_value") = -9999.0);

    // TWDTW sub-module
    py::module_ tw = m.def_submodule("twdtw", "TWDTW algorithms");

    py::class_<cdts::twdtw::TWDTWParams>(tw, "TWDTWParams")
        .def(py::init<>())
        .def_readwrite("alpha", &cdts::twdtw::TWDTWParams::alpha)
        .def_readwrite("beta", &cdts::twdtw::TWDTWParams::beta)
        .def_readwrite("gamma", &cdts::twdtw::TWDTWParams::gamma)
        .def_readwrite("max_time_warp", &cdts::twdtw::TWDTWParams::max_time_warp)
        .def_readwrite("subsequence_matching", &cdts::twdtw::TWDTWParams::subsequence_matching);

    py::class_<cdts::twdtw::TWDTWResult>(tw, "TWDTWResult")
        .def(py::init<>())
        .def_readwrite("distance", &cdts::twdtw::TWDTWResult::distance)
        .def_readwrite("path", &cdts::twdtw::TWDTWResult::path);

    tw.def("fit_twdtw", &cdts::twdtw::fit_twdtw,
           "Run TWDTW on a single time series against a pattern",
           py::arg("ts_values"), py::arg("ts_dates"), 
           py::arg("pattern_values"), py::arg("pattern_dates"),
           py::arg("num_bands") = 1,
           py::arg("params") = cdts::twdtw::TWDTWParams(),
           py::arg("abort_threshold") = std::numeric_limits<double>::infinity(),
           py::arg("return_path") = false);

    tw.def("fit_twdtw_batch", &cdts::twdtw::fit_twdtw_batch,
           "Run TWDTW on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("dates_array"),
           py::arg("pattern_values_array"), py::arg("pattern_dates_array"),
           py::arg("params"), 
           py::arg("abort_threshold") = std::numeric_limits<double>::infinity(),
           py::arg("n_jobs") = -1);

    // SOM submodule
    py::module_ som = m.def_submodule("som", "SOM C++ implementations");
    som.def("train_som_batch", &cdts::som::train_som_batch, "Train a Batch SOM");
    som.def("predict_bmus", &cdts::som::predict_bmus, "Find BMU for samples");

    // Phenology sub-module
    py::module_ ph = m.def_submodule("phenology", "Phenology extraction");
    
    py::enum_<phenology::CurveType>(ph, "CurveType")
        .value("BECK", phenology::CurveType::BECK)
        .value("ELMORE", phenology::CurveType::ELMORE)
        .value("GU", phenology::CurveType::GU)
        .value("KLOS", phenology::CurveType::KLOS)
        .value("ZHANG", phenology::CurveType::ZHANG)
        .value("AG", phenology::CurveType::AG)
        .value("DL", phenology::CurveType::DL)
        .export_values();

    py::enum_<phenology::ExtractionMethod>(ph, "ExtractionMethod")
        .value("THRESHOLD", phenology::ExtractionMethod::THRESHOLD)
        .value("DERIVATIVE", phenology::ExtractionMethod::DERIVATIVE)
        .value("GU", phenology::ExtractionMethod::GU)
        .value("KLOSTERMAN", phenology::ExtractionMethod::KLOSTERMAN)
        .export_values();

    ph.def("fit_phenology_batch", &phenology::fit_phenology_batch,
           "Run phenology extraction on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("dates_array"),
           py::arg("curve_type"), 
           py::arg("extraction_method") = 0,
           py::arg("max_seasons") = 2,
           py::arg("whittaker_lambda") = 10.0,
           py::arg("apply_whittaker") = true,
           py::arg("apply_hants") = false,
           py::arg("hants_frequencies") = 3,
           py::arg("hants_threshold") = 0.1,
           py::arg("min_season_length") = 0,
           py::arg("min_amplitude") = 0.0,
           py::arg("min_pixel_amplitude") = 0.1,
           py::arg("rtrough_max") = 0.6,
           py::arg("r_min_filter") = 0.02,
           py::arg("n_jobs") = -1);
}
