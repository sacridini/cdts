#include "twdtw.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace cdts {
namespace twdtw {

double time_weight(int t1, int t2, const TWDTWParams& params) {
    double dt = std::abs(t1 - t2);
    return params.alpha / (1.0 + std::exp(-params.beta * (dt - params.gamma)));
}

double fit_twdtw(const std::vector<double>& ts_values,
                 const std::vector<int>& ts_dates,
                 const std::vector<double>& pattern_values,
                 const std::vector<int>& pattern_dates,
                 const TWDTWParams& params) {
    
    int n = ts_values.size();
    int m = pattern_values.size();

    if (n == 0 || m == 0) return std::numeric_limits<double>::infinity();

    std::vector<std::vector<double>> d(n + 1, std::vector<double>(m + 1, std::numeric_limits<double>::infinity()));
    d[0][0] = 0.0;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            double spatial_dist = std::abs(ts_values[i - 1] - pattern_values[j - 1]);
            double temp_penalty = time_weight(ts_dates[i - 1], pattern_dates[j - 1], params);
            double cvalue = spatial_dist + temp_penalty;

            d[i][j] = cvalue + std::min({
                d[i - 1][j - 1],
                d[i - 1][j],
                d[i][j - 1]
            });
        }
    }
    
    return d[n][m];
}

pybind11::array_t<double> fit_twdtw_batch(
    pybind11::array_t<double> values_array,
    pybind11::array_t<int> dates_array,
    pybind11::array_t<double> pattern_values_array,
    pybind11::array_t<int> pattern_dates_array,
    const TWDTWParams& params,
    int n_jobs) {
    
    // Request buffer information
    auto buf_values = values_array.request();
    auto buf_dates = dates_array.request();
    auto buf_pat_values = pattern_values_array.request();
    auto buf_pat_dates = pattern_dates_array.request();

    if (buf_values.ndim != 3) throw std::runtime_error("values_array must be 3D [Y, X, Time]");
    if (buf_dates.ndim != 1) throw std::runtime_error("dates_array must be 1D [Time]");
    if (buf_pat_values.ndim != 1) throw std::runtime_error("pattern_values_array must be 1D");
    if (buf_pat_dates.ndim != 1) throw std::runtime_error("pattern_dates_array must be 1D");

    int Y = buf_values.shape[0];
    int X = buf_values.shape[1];
    int T = buf_values.shape[2];
    int P = buf_pat_values.shape[0];

    if (buf_dates.shape[0] != T) throw std::runtime_error("dates_array length must match Time dimension of values_array");
    if (buf_pat_dates.shape[0] != P) throw std::runtime_error("pattern_dates_array length must match pattern_values_array");

    const double* values_ptr = static_cast<double*>(buf_values.ptr);
    const int* dates_ptr = static_cast<int*>(buf_dates.ptr);
    const double* pat_values_ptr = static_cast<double*>(buf_pat_values.ptr);
    const int* pat_dates_ptr = static_cast<int*>(buf_pat_dates.ptr);

    // Prepare result array: [Y, X]
    auto result_array = pybind11::array_t<double>({Y, X});
    auto buf_result = result_array.request();
    double* result_ptr = static_cast<double*>(buf_result.ptr);

    // Convert patterns to vector once
    std::vector<double> pat_vals(pat_values_ptr, pat_values_ptr + P);
    std::vector<int> pat_dates(pat_dates_ptr, pat_dates_ptr + P);
    std::vector<int> ts_dates(dates_ptr, dates_ptr + T);

    #pragma omp parallel for collapse(2) num_threads(n_jobs > 0 ? n_jobs : omp_get_max_threads())
    for (int y = 0; y < Y; ++y) {
        for (int x = 0; x < X; ++x) {
            std::vector<double> ts_vals(T);
            for (int t = 0; t < T; ++t) {
                ts_vals[t] = values_ptr[y * X * T + x * T + t];
            }
            
            double dist = fit_twdtw(ts_vals, ts_dates, pat_vals, pat_dates, params);
            result_ptr[y * X + x] = dist;
        }
    }

    return result_array;
}

} // namespace twdtw
} // namespace cdts
