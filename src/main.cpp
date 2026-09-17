#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "landtrendr.h"
#include "ccdc.h"
#include "utils.h"
#include "twdtw.h"
#include "som.h"
#include "phenology.h"
#include "mann_kendall.h"
#include "bfast_monitor.h"
#include "bfast_lite.h"
#include "bfast.h"

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
           py::arg("n_jobs") = -1,
           py::arg("weights_array") = py::none(),
           py::arg("season_retry") = true);

    ph.def("debug_split_seasons", &phenology::debug_split_seasons,
           "Directly run the season-boundary detector (no smoothing/curve fitting) for unit testing",
           py::arg("y"), py::arg("dates") = py::none(), py::arg("min_season_length") = 0, py::arg("min_amplitude") = 0.0,
           py::arg("rtrough_max") = 0.6, py::arg("r_min_filter") = 0.02, py::arg("retry_on_empty") = true);

    // Mann-Kendall sub-module
    py::module_ mkmod = m.def_submodule("mannkendall", "Mann-Kendall trend test family + Sen's slope");

    py::enum_<cdts::mannkendall::MKMethod>(mkmod, "MKMethod")
        .value("ORIGINAL", cdts::mannkendall::MKMethod::ORIGINAL)
        .value("HAMED_RAO", cdts::mannkendall::MKMethod::HAMED_RAO)
        .value("YUE_WANG", cdts::mannkendall::MKMethod::YUE_WANG)
        .value("SEASONAL", cdts::mannkendall::MKMethod::SEASONAL)
        .export_values();

    mkmod.def("fit_mann_kendall_batch", &cdts::mannkendall::fit_mann_kendall_batch,
           "Pixel-wise Mann-Kendall trend test + Sen's slope on a batch of time series with OpenMP",
           py::arg("values_array"), py::arg("method") = 1, py::arg("alpha") = 0.05,
           py::arg("lag") = -1, py::arg("period") = 1, py::arg("min_valid") = 4, py::arg("n_jobs") = -1);

    mkmod.def("mk_test_single", &cdts::mannkendall::mk_test_single,
           "Run the Mann-Kendall test on a single time series (unit-testing helper)",
           py::arg("y"), py::arg("method") = 1, py::arg("alpha") = 0.05,
           py::arg("lag") = -1, py::arg("period") = 1);

    // bfastmonitor sub-module
    py::module_ bfm = m.def_submodule("bfastmonitor", "bfastmonitor: near-real-time structural change monitoring");

    py::class_<cdts::bfastmonitor::BFMResult>(bfm, "BFMResult")
        .def(py::init<>())
        .def_readwrite("breakpoint", &cdts::bfastmonitor::BFMResult::breakpoint)
        .def_readwrite("breakpoint_idx", &cdts::bfastmonitor::BFMResult::breakpoint_idx)
        .def_readwrite("magnitude", &cdts::bfastmonitor::BFMResult::magnitude)
        .def_readwrite("sigma", &cdts::bfastmonitor::BFMResult::sigma)
        .def_readwrite("n_history", &cdts::bfastmonitor::BFMResult::n_history)
        .def_readwrite("has_break", &cdts::bfastmonitor::BFMResult::has_break)
        .def_readwrite("valid", &cdts::bfastmonitor::BFMResult::valid);

    bfm.def("bfast_monitor", &cdts::bfastmonitor::bfast_monitor,
           "Run bfastmonitor on a single pixel time series (unit-testing helper)",
           py::arg("y"), py::arg("start_time"), py::arg("monitor_start_time"),
           py::arg("frequency"), py::arg("order") = 3, py::arg("h") = 0.25,
           py::arg("period") = 10, py::arg("alpha") = 0.05);

    bfm.def("fit_bfast_monitor_batch", &cdts::bfastmonitor::fit_bfast_monitor_batch,
           "Run bfastmonitor on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("start_time"), py::arg("monitor_start_time"),
           py::arg("frequency"), py::arg("order") = 3, py::arg("h") = 0.25,
           py::arg("period") = 10, py::arg("alpha") = 0.05, py::arg("min_valid") = 10,
           py::arg("n_jobs") = -1);

    // bfastlite sub-module
    py::module_ bfl = m.def_submodule("bfastlite", "bfastlite: single-pass multiple-breakpoint detection");

    py::class_<cdts::bfastlite::BFLResult>(bfl, "BFLResult")
        .def(py::init<>())
        .def_readwrite("n_breaks", &cdts::bfastlite::BFLResult::n_breaks)
        .def_readwrite("rss", &cdts::bfastlite::BFLResult::rss)
        .def_readwrite("lwz", &cdts::bfastlite::BFLResult::lwz)
        .def_readwrite("n_valid", &cdts::bfastlite::BFLResult::n_valid)
        .def_readwrite("valid", &cdts::bfastlite::BFLResult::valid)
        .def_readwrite("breakpoint_idx", &cdts::bfastlite::BFLResult::breakpoint_idx);

    bfl.def("bfast_lite", &cdts::bfastlite::bfast_lite,
           "Run bfastlite on a single pixel time series (unit-testing helper)",
           py::arg("y"), py::arg("start_time"), py::arg("frequency"),
           py::arg("order") = 3, py::arg("h") = 0.15, py::arg("max_breaks_output") = 5);

    bfl.def("fit_bfast_lite_batch", &cdts::bfastlite::fit_bfast_lite_batch,
           "Run bfastlite on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("start_time"), py::arg("frequency"),
           py::arg("order") = 3, py::arg("h") = 0.15, py::arg("max_breaks_output") = 5,
           py::arg("min_valid") = 20, py::arg("n_jobs") = -1);

    // bfast sub-module (the classic iterative trend+season break detection)
    py::module_ bf = m.def_submodule("bfast", "bfast: classic iterative trend+season break detection");

    py::class_<cdts::bfast::BFResult>(bf, "BFResult")
        .def(py::init<>())
        .def_readwrite("n_trend_breaks", &cdts::bfast::BFResult::n_trend_breaks)
        .def_readwrite("n_season_breaks", &cdts::bfast::BFResult::n_season_breaks)
        .def_readwrite("magnitude", &cdts::bfast::BFResult::magnitude)
        .def_readwrite("time", &cdts::bfast::BFResult::time)
        .def_readwrite("n_iter", &cdts::bfast::BFResult::n_iter)
        .def_readwrite("n_valid", &cdts::bfast::BFResult::n_valid)
        .def_readwrite("valid", &cdts::bfast::BFResult::valid)
        .def_readwrite("trend_breakpoint_idx", &cdts::bfast::BFResult::trend_breakpoint_idx)
        .def_readwrite("season_breakpoint_idx", &cdts::bfast::BFResult::season_breakpoint_idx);

    bf.def("bfast", &cdts::bfast::bfast,
           "Run bfast on a single pixel time series (unit-testing helper)",
           py::arg("y"), py::arg("start_time"), py::arg("frequency"),
           py::arg("order") = 3, py::arg("h") = 0.15,
           py::arg("max_breaks_trend") = 5, py::arg("max_breaks_season") = 5,
           py::arg("max_iter") = 10, py::arg("level") = 0.05);

    bf.def("fit_bfast_batch", &cdts::bfast::fit_bfast_batch,
           "Run bfast on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("start_time"), py::arg("frequency"),
           py::arg("order") = 3, py::arg("h") = 0.15,
           py::arg("max_breaks_trend") = 5, py::arg("max_breaks_season") = 5,
           py::arg("max_iter") = 10, py::arg("level") = 0.05,
           py::arg("min_valid") = 20, py::arg("n_jobs") = -1);
}
