#pragma once
#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Mann-Kendall trend test family + Theil-Sen slope estimator, ported from
// pymannkendall (Hussain & Mahmud, 2019, JOSS, doi:10.21105/joss.01556) to
// C++/OpenMP for per-pixel batch throughput over EO time series cubes.

namespace cdts {
namespace mannkendall {

enum class MKMethod {
    ORIGINAL = 0,   // classic Mann-Kendall (Mann 1945, Kendall 1975)
    HAMED_RAO = 1,  // Hamed & Rao (1998) autocorrelation-corrected variance
    YUE_WANG = 2,   // Yue & Wang (2004) autocorrelation-corrected variance
    SEASONAL = 3    // Hirsch & Slack (1984), pools per-season MK scores
};

struct MKResult {
    int trend = 0;      // -1 decreasing, 0 no trend, 1 increasing
    bool h = false;     // true if the trend is statistically significant at alpha
    double p = std::nan("");
    double z = std::nan("");
    double tau = std::nan("");
    double s = std::nan("");
    double var_s = std::nan("");
    // slope/intercept: units of the series per TIME STEP (index spacing),
    // not calendar time - pass one sample per year for a directly
    // interpretable per-year trend. For method=SEASONAL, slope/intercept
    // are per full `period` cycle instead (see seasonal_sens_slope).
    double slope = std::nan("");
    double intercept = std::nan("");
};

// Run the Mann-Kendall test on a single time series. y may contain NaNs
// (skipped, mirroring pymannkendall's missing_values_analysis(method='skip')).
// period is only used when method == SEASONAL (e.g. 23 for MODIS 16-day
// annual cycles, 12 for monthly data).
MKResult mk_test(
    const std::vector<double>& y,
    MKMethod method = MKMethod::HAMED_RAO,
    double alpha = 0.05,
    int lag = -1,       // -1 == None in pymannkendall (uses the full series length)
    int period = 1);

// Directly-testable single-series wrapper exposed to Python for unit tests,
// returning a plain tuple instead of requiring a full batch array.
pybind11::tuple mk_test_single(
    std::vector<double> y,
    int method = 1,
    double alpha = 0.05,
    int lag = -1,
    int period = 1);

// Batch entry point: values_array [n_pixels, n_time] -> out [9, n_pixels].
// Row order: trend, h, p, z, tau, s, var_s, slope, intercept.
// Pixels with fewer than min_valid non-NaN observations are left as NaN.
pybind11::array_t<double> fit_mann_kendall_batch(
    pybind11::array_t<double> values_array,
    int method = 1,
    double alpha = 0.05,
    int lag = -1,
    int period = 1,
    int min_valid = 4,
    int n_jobs = -1);

} // namespace mannkendall
} // namespace cdts
