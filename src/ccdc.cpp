#include "ccdc.h"
#include <cmath>
#include <Eigen/Dense>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cdts {
namespace ccdc {

const double PI = 3.14159265358979323846;
const double W = 2.0 * PI / 365.25;

// Incomplete Gamma for Chi-Square distribution
double gammln_ccdc(double xx) {
    double x, y, tmp, ser;
    static double cof[6] = {76.18009172947146, -86.50532032941677,
                            24.01409824083091, -1.231739572450155,
                            0.1208650973866179e-2, -0.5395239384953e-5};
    y = x = xx;
    tmp = x + 5.5;
    tmp -= (x + 0.5) * std::log(tmp);
    ser = 1.000000000190015;
    for (int j = 0; j <= 5; j++) ser += cof[j] / ++y;
    return -tmp + std::log(2.5066282746310005 * ser / x);
}

void gser(double* gamser, double a, double x, double* gln) {
    int n;
    double sum, del, ap;
    *gln = gammln_ccdc(a);
    if (x <= 0.0) {
        *gamser = 0.0;
        return;
    }
    ap = a;
    del = sum = 1.0 / a;
    for (n = 1; n <= 100; n++) {
        ++ap;
        del *= x / ap;
        sum += del;
        if (std::abs(del) < std::abs(sum) * 3.0e-7) {
            *gamser = sum * std::exp(-x + a * std::log(x) - (*gln));
            return;
        }
    }
}

void gcf(double* gammcf, double a, double x, double* gln) {
    int i;
    double an, b, c, d, del, h;
    *gln = gammln_ccdc(a);
    b = x + 1.0 - a;
    c = 1.0 / 1.0e-30;
    d = 1.0 / b;
    h = d;
    for (i = 1; i <= 100; i++) {
        an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < 1.0e-30) d = 1.0e-30;
        c = b + an / c;
        if (std::abs(c) < 1.0e-30) c = 1.0e-30;
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 3.0e-7) break;
    }
    *gammcf = std::exp(-x + a * std::log(x) - (*gln)) * h;
}

double gammap(double a, double x) {
    double gamser, gammcf, gln;
    if (x < 0.0 || a <= 0.0) return 0.0;
    if (x < (a + 1.0)) {
        gser(&gamser, a, x, &gln);
        return gamser;
    } else {
        gcf(&gammcf, a, x, &gln);
        return 1.0 - gammcf;
    }
}

double chi2_cdf(double x, double df) {
    return gammap(df / 2.0, x / 2.0);
}

// Helper to fit Harmonic OLS with IRLS (Robust Fit) for a SINGLE band
bool fit_harmonic_robust(const std::vector<int>& dates, 
                         const std::vector<double>& values,
                         int start_idx, int end_idx,
                         Eigen::VectorXd& beta_out, double& rmse_out) {
    int n = end_idx - start_idx + 1;
    int k = 6; 
    if (n < k) return false;
    
    Eigen::MatrixXd X(n, k);
    Eigen::VectorXd Y(n);
    
    for (int i = 0; i < n; ++i) {
        double t = dates[start_idx + i];
        Y(i) = values[start_idx + i];
        
        X(i, 0) = 1.0;
        X(i, 1) = t;
        X(i, 2) = std::cos(W * t);
        X(i, 3) = std::sin(W * t);
        X(i, 4) = std::cos(2.0 * W * t);
        X(i, 5) = std::sin(2.0 * W * t);
    }
    
    // IRLS Loop (3 iterations is usually enough for CCDC initialization)
    Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
    for (int iter = 0; iter < 4; ++iter) {
        Eigen::MatrixXd W_mat = w.asDiagonal();
        Eigen::MatrixXd Xw = W_mat * X;
        Eigen::VectorXd Yw = W_mat * Y;
        
        beta_out = (Xw.transpose() * Xw).ldlt().solve(Xw.transpose() * Yw);
        
        Eigen::VectorXd residuals = Y - X * beta_out;
        
        // Calculate Median Absolute Deviation (MAD)
        std::vector<double> abs_res(n);
        for(int i=0; i<n; ++i) abs_res[i] = std::abs(residuals(i));
        std::sort(abs_res.begin(), abs_res.end());
        double mad = abs_res[n/2];
        if (mad < 1e-6) mad = 1e-6;
        
        // Bisquare weights
        double tune = 4.685 * mad / 0.6745;
        for (int i = 0; i < n; ++i) {
            double r = std::abs(residuals(i)) / tune;
            if (r < 1.0) {
                w(i) = (1.0 - r*r) * (1.0 - r*r);
            } else {
                w(i) = 0.0;
            }
        }
    }
    
    Eigen::VectorXd residuals = Y - X * beta_out;
    double sse = residuals.squaredNorm();
    rmse_out = std::sqrt(sse / (n - k));
    return true;
}

std::vector<CCDCSegment> fit_ccdc(const std::vector<int>& dates, 
                                  const std::vector<std::vector<double>>& band_values,
                                  const std::vector<int>& qa,
                                  CCDCParams params) {
    std::vector<CCDCSegment> segments;
    int num_bands = band_values.size();
    if (num_bands == 0) return segments;
    int n_total = dates.size();
    
    // 1. Filter out QA pixels (clouds, shadows)
    std::vector<int> valid_dates;
    std::vector<std::vector<double>> valid_bands(num_bands);
    
    for (int i = 0; i < n_total; ++i) {
        if (qa[i] == 0) { // Assuming 0 means clear
            valid_dates.push_back(dates[i]);
            for (int b = 0; b < num_bands; ++b) {
                valid_bands[b].push_back(band_values[b][i]);
            }
        }
    }
    
    int n = valid_dates.size();
    if (n < params.min_obs) return segments;
    
    int start_idx = 0;
    while (start_idx < n) {
        // We need at least min_obs to initialize a model
        if (n - start_idx < params.min_obs) break;
        
        int end_idx = start_idx + params.min_obs - 1;
        
        std::vector<Eigen::VectorXd> betas(num_bands);
        std::vector<double> rmses(num_bands, 0.0);
        bool init_success = true;
        
        // Fit initial model for ALL bands
        for (int b = 0; b < num_bands; ++b) {
            if (!fit_harmonic_robust(valid_dates, valid_bands[b], start_idx, end_idx, betas[b], rmses[b])) {
                init_success = false;
                break;
            }
            if (rmses[b] < 1e-4) rmses[b] = 1e-4; // avoid div by zero in normalized metrics
        }
        
        if (!init_success) break;
        
        int break_idx = -1;
        int anom_count = 0;
        
        // 2. Moving Window over time
        for (int i = end_idx + 1; i < n; ++i) {
            double t = valid_dates[i];
            
            // Calculate a unified Change Metric across all bands
            double change_metric = 0.0;
            
            for (int b = 0; b < num_bands; ++b) {
                double actual = valid_bands[b][i];
                double pred = betas[b](0) + betas[b](1)*t + 
                              betas[b](2)*std::cos(W*t) + betas[b](3)*std::sin(W*t) + 
                              betas[b](4)*std::cos(2.0*W*t) + betas[b](5)*std::sin(2.0*W*t);
                              
                double residual = std::abs(actual - pred);
                
                // Normalize by RMSE
                double norm_res = residual / rmses[b];
                change_metric += norm_res * norm_res;
            }
            
            // Dynamic threshold using exact Chi-Square CDF
            double prob_change = chi2_cdf(change_metric, num_bands);
            
            if (prob_change > params.chi2_prob_threshold) {
                anom_count++;
                if (anom_count == params.conseq_anom) {
                    bool valid_break = true;
                    // COLD Angle logic: check if residuals have consistent signs (direction) over the anomalous period
                    if (params.conseq_anom >= 6) {
                        for (int b = 0; b < num_bands; ++b) {
                            int pos_count = 0;
                            int neg_count = 0;
                            for (int k = i - params.conseq_anom + 1; k <= i; ++k) {
                                double t_anom = valid_dates[k];
                                double actual = valid_bands[b][k];
                                double pred = betas[b](0) + betas[b](1)*t_anom + 
                                              betas[b](2)*std::cos(W*t_anom) + betas[b](3)*std::sin(W*t_anom) + 
                                              betas[b](4)*std::cos(2.0*W*t_anom) + betas[b](5)*std::sin(2.0*W*t_anom);
                                if (actual > pred) pos_count++;
                                else neg_count++;
                            }
                            // If a band's residuals bounce around (+ and -), it's noise, not a consistent break trajectory
                            if (pos_count > 1 && neg_count > 1) {
                                valid_break = false;
                                break;
                            }
                        }
                    }
                    
                    if (valid_break) {
                        break_idx = i - params.conseq_anom + 1; // Mark the start of the anomalies
                        break;
                    } else {
                        // False alarm, reset anomaly counter
                        anom_count = 0;
                        end_idx = i;
                    }
                }
            } else {
                anom_count = 0; 
                end_idx = i;
                
                // Dynamic Model Updating: update coefficients every 24 new observations
                if ((end_idx - start_idx + 1) % 24 == 0) {
                    for (int b = 0; b < num_bands; ++b) {
                        fit_harmonic_robust(valid_dates, valid_bands[b], start_idx, end_idx, betas[b], rmses[b]);
                        if (rmses[b] < 1e-4) rmses[b] = 1e-4;
                    }
                }
            }
        }
        
        // 3. Finalize Segment
        // Refit all bands over the stable period
        CCDCSegment seg;
        seg.t_start = valid_dates[start_idx];
        seg.t_end = valid_dates[end_idx];
        seg.t_break = (break_idx != -1) ? valid_dates[break_idx] : 0;
        
        for (int b = 0; b < num_bands; ++b) {
            fit_harmonic_robust(valid_dates, valid_bands[b], start_idx, end_idx, betas[b], rmses[b]);
            
            seg.rmse.push_back(rmses[b]);
            
            std::vector<double> band_coefs(6);
            for (int c = 0; c < 6; ++c) {
                band_coefs[c] = betas[b](c);
            }
            seg.coefs.push_back(band_coefs);
            
            double mag = 0.0;
            if (break_idx != -1) {
                double t = valid_dates[break_idx];
                double pred = betas[b](0) + betas[b](1)*t + 
                              betas[b](2)*std::cos(W*t) + betas[b](3)*std::sin(W*t) + 
                              betas[b](4)*std::cos(2.0*W*t) + betas[b](5)*std::sin(2.0*W*t);
                mag = valid_bands[b][break_idx] - pred;
            }
            seg.magnitude.push_back(mag);
        }
        
        segments.push_back(seg);
        
        // 4. Move forward
        if (break_idx != -1) {
            start_idx = break_idx;
        } else {
            break;
        }
    }
    
    return segments;
}

pybind11::tuple fit_ccdc_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Bands, Time]
    pybind11::array_t<int> qa_array,        // Shape: [Y, X, Time]
    pybind11::array_t<int> dates_array,     // Shape: [Time]
    CCDCParams params,
    int max_segments,
    bool return_coefs,
    int n_jobs) 
{
    auto val_buf = values_array.request();
    auto qa_buf = qa_array.request();
    auto dates_buf = dates_array.request();
    
    int height = val_buf.shape[0];
    int width = val_buf.shape[1];
    int num_bands = val_buf.shape[2];
    int times = val_buf.shape[3];
    int num_pixels = height * width;
    
    double* val_ptr = static_cast<double*>(val_buf.ptr);
    int* qa_ptr = static_cast<int*>(qa_buf.ptr);
    int* dates_ptr = static_cast<int*>(dates_buf.ptr);
    
    std::vector<int> dates(dates_ptr, dates_ptr + times);
    
    int params_per_segment = return_coefs ? (3 + num_bands * 7) : 1;
    
    // Output arrays
    pybind11::array_t<double> segments_out({num_pixels, max_segments, params_per_segment});
    auto seg_ptr = static_cast<double*>(segments_out.request().ptr);
    
    pybind11::array_t<int> counts_out(num_pixels);
    auto counts_ptr = static_cast<int*>(counts_out.request().ptr);
    
    std::fill(seg_ptr, seg_ptr + (num_pixels * max_segments * params_per_segment), 0.0);
    std::fill(counts_ptr, counts_ptr + num_pixels, 0);
    
    #ifdef _OPENMP
    if (n_jobs > 0) {
        omp_set_num_threads(n_jobs);
    }
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int p = 0; p < num_pixels; ++p) {
        std::vector<std::vector<double>> pixel_bands(num_bands, std::vector<double>(times));
        std::vector<int> pixel_qa(times);
        
        bool all_nan = true;
        for (int t = 0; t < times; ++t) {
            pixel_qa[t] = qa_ptr[p * times + t];
            bool has_nan_in_time = false;
            for (int b = 0; b < num_bands; ++b) {
                double v = val_ptr[p * num_bands * times + b * times + t];
                pixel_bands[b][t] = v;
                if (std::isnan(v)) {
                    has_nan_in_time = true;
                } else if (v != 0.0) {
                    all_nan = false;
                }
            }
            if (has_nan_in_time) {
                pixel_qa[t] = 1; // Mask this date if any band is NaN
            }
        }
        
        if (all_nan) {
            continue;
        }
        
        std::vector<CCDCSegment> segs;
        try {
            segs = fit_ccdc(dates, pixel_bands, pixel_qa, params);
        } catch (...) {
            // Ignore errors for individual pixels
        }
        
        int n_segs = std::min((int)segs.size(), max_segments);
        counts_ptr[p] = n_segs;
        
        for (int i = 0; i < n_segs; ++i) {
            const auto& seg = segs[i];
            int base_idx = p * max_segments * params_per_segment + i * params_per_segment;
            
            if (return_coefs) {
                seg_ptr[base_idx + 0] = seg.t_start;
                seg_ptr[base_idx + 1] = seg.t_end;
                seg_ptr[base_idx + 2] = seg.t_break > 0 ? seg.t_break : 0;
                
                int idx = 3;
                for (int b = 0; b < num_bands; ++b) {
                    seg_ptr[base_idx + idx++] = seg.rmse[b];
                    for (int c = 0; c < 6; ++c) {
                        seg_ptr[base_idx + idx++] = seg.coefs[b][c];
                    }
                }
            } else {
                seg_ptr[base_idx + 0] = seg.t_break > 0 ? seg.t_break : 0;
            }
        }
    }
    
    return pybind11::make_tuple(segments_out, counts_out);
}

} // namespace ccdc
} // namespace cdts
