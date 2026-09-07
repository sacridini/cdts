#include "twdtw.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <Eigen/Dense> // For SIMD operations

namespace cdts {
namespace twdtw {

// 1. LUT (Look-Up Table) Optimization for Logistic Time Penalty
class TWDTW_LUT {
public:
    std::vector<double> lut;
    TWDTW_LUT(const TWDTWParams& p, int max_diff = 2000) {
        lut.resize(max_diff + 1);
        for(int i = 0; i <= max_diff; ++i) {
            lut[i] = p.alpha / (1.0 + std::exp(-p.beta * (i - p.gamma)));
        }
    }
    
    inline double get(int t1, int t2) const {
        int dt = std::abs(t1 - t2);
        if (dt >= lut.size()) return lut.back();
        return lut[dt];
    }
};

double fit_twdtw(const std::vector<double>& ts_values,
                 const std::vector<int>& ts_dates,
                 const std::vector<double>& pattern_values,
                 const std::vector<int>& pattern_dates,
                 const TWDTWParams& params,
                 double abort_threshold) {
    
    int n = ts_values.size();
    int m = pattern_values.size();

    if (n == 0 || m == 0) return std::numeric_limits<double>::infinity();

    TWDTW_LUT lut(params);

    // 2. Memory Locality Optimization (O(M) space instead of O(N*M))
    std::vector<double> prev_row(m + 1, std::numeric_limits<double>::infinity());
    std::vector<double> curr_row(m + 1, std::numeric_limits<double>::infinity());
    prev_row[0] = 0.0;

    for (int i = 1; i <= n; ++i) {
        curr_row[0] = std::numeric_limits<double>::infinity();
        double min_in_row = std::numeric_limits<double>::infinity();

        for (int j = 1; j <= m; ++j) {
            // 3. Sakoe-Chiba Band (Time window constraint)
            if (std::abs(ts_dates[i - 1] - pattern_dates[j - 1]) > params.max_time_warp) {
                curr_row[j] = std::numeric_limits<double>::infinity();
                continue;
            }

            // Here we use scalar absolute. For multidimensional multi-band images,
            // we would map Eigen::VectorXd and compute SIMD spatial_dist = (v1 - v2).norm();
            double spatial_dist = std::abs(ts_values[i - 1] - pattern_values[j - 1]);
            
            // Fast LUT access instead of std::exp
            double temp_penalty = lut.get(ts_dates[i - 1], pattern_dates[j - 1]);
            double cvalue = spatial_dist + temp_penalty;

            curr_row[j] = cvalue + std::min({
                prev_row[j - 1],
                prev_row[j],
                curr_row[j - 1]
            });
            
            min_in_row = std::min(min_in_row, curr_row[j]);
        }
        
        // 4. Early Abandonment
        if (min_in_row > abort_threshold) {
            return std::numeric_limits<double>::infinity();
        }

        std::swap(prev_row, curr_row);
    }
    
    return prev_row[m];
}

pybind11::array_t<double> fit_twdtw_batch(
    pybind11::array_t<double> values_array,
    pybind11::array_t<int> dates_array,
    pybind11::array_t<double> pattern_values_array,
    pybind11::array_t<int> pattern_dates_array,
    const TWDTWParams& params,
    double abort_threshold,
    int n_jobs) {
    
    auto buf_values = values_array.request();
    auto buf_dates = dates_array.request();
    auto buf_pat_values = pattern_values_array.request();
    auto buf_pat_dates = pattern_dates_array.request();

    int Y = buf_values.shape[0];
    int X = buf_values.shape[1];
    int T = buf_values.shape[2];
    int P = buf_pat_values.shape[0];

    const double* values_ptr = static_cast<double*>(buf_values.ptr);
    const int* dates_ptr = static_cast<int*>(buf_dates.ptr);
    const double* pat_values_ptr = static_cast<double*>(buf_pat_values.ptr);
    const int* pat_dates_ptr = static_cast<int*>(buf_pat_dates.ptr);

    auto result_array = pybind11::array_t<double>({Y, X});
    auto buf_result = result_array.request();
    double* result_ptr = static_cast<double*>(buf_result.ptr);

    std::vector<double> pat_vals(pat_values_ptr, pat_values_ptr + P);
    std::vector<int> pat_dates(pat_dates_ptr, pat_dates_ptr + P);
    std::vector<int> ts_dates(dates_ptr, dates_ptr + T);

    // Precompute LUT once for the entire batch
    TWDTW_LUT lut(params);

    #pragma omp parallel for collapse(2) num_threads(n_jobs > 0 ? n_jobs : omp_get_max_threads())
    for (int y = 0; y < Y; ++y) {
        for (int x = 0; x < X; ++x) {
            std::vector<double> ts_vals(T);
            
            // SIMD compatible loop for extracting memory contiguous data
            
            for (int t = 0; t < T; ++t) {
                ts_vals[t] = values_ptr[y * X * T + x * T + t];
            }
            
            // Local DP execution using optimized implementation
            int n = ts_vals.size();
            int m = pat_vals.size();
            
            if (n == 0 || m == 0) {
                result_ptr[y * X + x] = std::numeric_limits<double>::infinity();
                continue;
            }

            std::vector<double> prev_row(m + 1, std::numeric_limits<double>::infinity());
            std::vector<double> curr_row(m + 1, std::numeric_limits<double>::infinity());
            prev_row[0] = 0.0;
            
            bool aborted = false;

            for (int i = 1; i <= n; ++i) {
                curr_row[0] = std::numeric_limits<double>::infinity();
                double min_in_row = std::numeric_limits<double>::infinity();

                for (int j = 1; j <= m; ++j) {
                    if (std::abs(ts_dates[i - 1] - pat_dates[j - 1]) > params.max_time_warp) {
                        curr_row[j] = std::numeric_limits<double>::infinity();
                        continue;
                    }

                    double spatial_dist = std::abs(ts_vals[i - 1] - pat_vals[j - 1]);
                    double temp_penalty = lut.get(ts_dates[i - 1], pat_dates[j - 1]);
                    double cvalue = spatial_dist + temp_penalty;

                    curr_row[j] = cvalue + std::min({
                        prev_row[j - 1],
                        prev_row[j],
                        curr_row[j - 1]
                    });
                    
                    min_in_row = std::min(min_in_row, curr_row[j]);
                }
                
                if (min_in_row > abort_threshold) {
                    aborted = true;
                    break;
                }
                std::swap(prev_row, curr_row);
            }

            result_ptr[y * X + x] = aborted ? std::numeric_limits<double>::infinity() : prev_row[m];
        }
    }

    return result_array;
}

} // namespace twdtw
} // namespace cdts
