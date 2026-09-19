#include "landtrendr.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <numeric>

#ifdef _OPENMP
#ifdef _OPENMP
#ifdef _OPENMP
#include <omp.h>
#endif
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x) (void)(x)
#endif
#endif

// -------------------------------------------------------------
// Math Functions for Statistical Significance (P-value / F-Stat)
// -------------------------------------------------------------
double gammln(double xx) {
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

double betacf(double a, double b, double x) {
    int m, m2;
    double aa, c, d, del, h, qab, qam, qap;
    qab = a + b;
    qap = a + 1.0;
    qam = a - 1.0;
    c = 1.0;
    d = 1.0 - qab * x / qap;
    if (std::abs(d) < 1.0e-30) d = 1.0e-30;
    d = 1.0 / d;
    h = d;
    for (m = 1; m <= 100; m++) {
        m2 = 2 * m;
        aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < 1.0e-30) d = 1.0e-30;
        c = 1.0 + aa / c;
        if (std::abs(c) < 1.0e-30) c = 1.0e-30;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < 1.0e-30) d = 1.0e-30;
        c = 1.0 + aa / c;
        if (std::abs(c) < 1.0e-30) c = 1.0e-30;
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 3.0e-7) break;
    }
    return h;
}

double betai(double a, double b, double x) {
    double bt;
    if (x == 0.0 || x == 1.0) bt = 0.0;
    else bt = std::exp(gammln(a + b) - gammln(a) - gammln(b) + a * std::log(x) + b * std::log(1.0 - x));
    
    if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
    else return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double f_pval(double f_stat, double df1, double df2) {
    if (f_stat < 0.0) return 1.0;
    if (df1 <= 0 || df2 <= 0) return 1.0;
    double x = df2 / (df2 + df1 * f_stat);
    return betai(df2 / 2.0, df1 / 2.0, x);
}

// -------------------------------------------------------------
// Core Algorithm Logic
// -------------------------------------------------------------

namespace cdts {
namespace landtrendr {

// Internal function to calculate desawtooth corrections
struct DesawtoothCorrection {
    std::vector<double> correction;
    std::vector<double> prop_correction;
};

DesawtoothCorrection find_correction(const std::vector<double>& vals) {
    int n = vals.size();
    DesawtoothCorrection res;
    res.correction.assign(n, 0.0);
    res.prop_correction.assign(n, 0.0);

    for (int i = 1; i < n - 1; ++i) {
        double diff_2 = std::abs(vals[i - 1] - vals[i + 1]);
        double diff_minus1 = std::abs(vals[i] - vals[i + 1]);
        double diff_plus1 = std::abs(vals[i] - vals[i - 1]);
        
        double md = std::max(diff_minus1, diff_plus1);
        if (md == 0.0) {
            md = diff_2; // avoid division by zero. If md is 0, diff_2 is 0. prop_correction will be 0.
        }

        if (md > 0.0) {
            res.prop_correction[i] = 1.0 - (diff_2 / md);
        } else {
            res.prop_correction[i] = 0.0;
        }

        res.correction[i] = res.prop_correction[i] * (((vals[i - 1] + vals[i + 1]) / 2.0) - vals[i]);
    }

    return res;
}

std::vector<double> desawtooth(const std::vector<double>& vals, double stopat) {
    std::vector<double> v = vals;
    double prop = 1.0;

    while (prop > stopat) {
        DesawtoothCorrection c = find_correction(v);
        
        prop = 0.0;
        int wh_max = -1;
        
        // Find max prop_correction
        for (size_t i = 0; i < c.prop_correction.size(); ++i) {
            if (c.prop_correction[i] > prop) {
                prop = c.prop_correction[i];
                wh_max = i;
            }
        }
        
        if (prop > stopat && wh_max != -1) {
            v[wh_max] = v[wh_max] + c.correction[wh_max];
        } else {
            break;
        }
    }

    return v;
}

double angle_diff(double x0, double x1, double x2,
                  double y0, double y1, double y2,
                  double yrange, double distweightfactor) {
    double ydiff2 = y2 - y1;
    double ydiff1 = y1 - y0;

    double angle1 = std::atan(ydiff1 / (x1 - x0));
    double angle2 = std::atan(ydiff2 / (x2 - x1));

    double scaler = std::max(0.0, (ydiff2 * distweightfactor) / yrange) + 1.0;
    double diff = std::max(std::abs(angle1), std::abs(angle2)) * scaler;
    return diff;
}

std::vector<int> vet_verts(const std::vector<int>& x, const std::vector<double>& y, 
                           const std::vector<int>& vertices, int desired_count, 
                           double distweightfactor = 2.0) {
    int n_verts = vertices.size();
    int n_to_remove = n_verts - desired_count;

    if (n_to_remove <= 0 || n_verts <= 3) {
        return vertices;
    }

    // Min and max for y
    double min_y = *std::min_element(y.begin(), y.end());
    double max_y = *std::max_element(y.begin(), y.end());
    double yr = max_y - min_y;
    if (yr == 0.0) yr = 1.0; // avoid division by zero

    double range_x = x.back() - x.front();
    if (range_x == 0.0) range_x = 1.0;

    // Scale y
    std::vector<double> yscale(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        yscale[i] = ((y[i] - min_y) / yr) * range_x;
    }

    double sc_yr = *std::max_element(yscale.begin(), yscale.end()) - *std::min_element(yscale.begin(), yscale.end());
    if (sc_yr == 0.0) sc_yr = 1.0;

    std::vector<int> v = vertices;
    std::vector<double> slope_ratios(n_verts - 2);

    for (int i = 1; i < n_verts - 1; ++i) {
        slope_ratios[i - 1] = angle_diff(x[v[i - 1]], x[v[i]], x[v[i + 1]],
                                         yscale[v[i - 1]], yscale[v[i]], yscale[v[i + 1]],
                                         sc_yr, distweightfactor);
    }

    int count = n_verts;

    for (int step = 0; step < n_to_remove; ++step) {
        // Find minimum slope ratio
        double min_val = std::numeric_limits<double>::max();
        int worst = -1;
        for (int i = 0; i < count - 2; ++i) {
            if (slope_ratios[i] < min_val) {
                min_val = slope_ratios[i];
                worst = i;
            }
        }

        if (worst == -1) break; // Should not happen

        int worst_idx = worst + 1; // Index in vertex array

        // Remove the vertex
        v.erase(v.begin() + worst_idx);
        
        // Remove the corresponding slope ratio
        slope_ratios.erase(slope_ratios.begin() + worst);

        count--;

        // Recalculate neighbors
        if (worst_idx != 1) { // has left neighbor to recalculate
            int left_idx = worst_idx - 1;
            slope_ratios[left_idx - 1] = angle_diff(x[v[left_idx - 1]], x[v[left_idx]], x[v[left_idx + 1]],
                                                    yscale[v[left_idx - 1]], yscale[v[left_idx]], yscale[v[left_idx + 1]],
                                                    sc_yr, distweightfactor);
        }

        if (worst_idx != count - 1) { // has right neighbor to recalculate (which shifted to worst_idx)
            int right_idx = worst_idx;
            slope_ratios[right_idx - 1] = angle_diff(x[v[right_idx - 1]], x[v[right_idx]], x[v[right_idx + 1]],
                                                     yscale[v[right_idx - 1]], yscale[v[right_idx]], yscale[v[right_idx + 1]],
                                                     sc_yr, distweightfactor);
        }
    }

    return v;
}

// Simple matrix inversion for small matrices using Gauss-Jordan
bool invert_matrix(std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) I[i][i] = 1.0;

    for (int i = 0; i < n; ++i) {
        // Find pivot
        double max_el = std::abs(A[i][i]);
        int pivot = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > max_el) {
                max_el = std::abs(A[k][i]);
                pivot = k;
            }
        }
        if (max_el == 0.0) return false; // Singular

        // Swap rows
        if (pivot != i) {
            std::swap(A[i], A[pivot]);
            std::swap(I[i], I[pivot]);
        }

        // Scale row
        double diag = A[i][i];
        for (int j = 0; j < n; ++j) {
            A[i][j] /= diag;
            I[i][j] /= diag;
        }

        // Eliminate column
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = 0; j < n; ++j) {
                    A[k][j] -= factor * A[i][j];
                    I[k][j] -= factor * I[i][j];
                }
            }
        }
    }
    A = I;
    return true;
}

// Piecewise linear OLS fit with fixed breakpoints
std::vector<double> fit_piecewise_ols(const std::vector<int>& x, const std::vector<double>& y, const std::vector<int>& verts) {
    int n = x.size();
    int k = verts.size();
    
    // Build design matrix X_mat (n x k)
    std::vector<std::vector<double>> X_mat(n, std::vector<double>(k, 0.0));
    for (int i = 0; i < n; ++i) {
        int xi = x[i];
        for (int j = 0; j < k; ++j) {
            int vj = x[verts[j]];
            if (j > 0 && xi >= x[verts[j-1]] && xi <= vj) {
                int v_prev = x[verts[j-1]];
                if (vj > v_prev) {
                    X_mat[i][j] = static_cast<double>(xi - v_prev) / (vj - v_prev);
                }
            } else if (j < k - 1 && xi >= vj && xi <= x[verts[j+1]]) {
                int v_next = x[verts[j+1]];
                if (v_next > vj) {
                    X_mat[i][j] = static_cast<double>(v_next - xi) / (v_next - vj);
                }
            } else if (xi == vj) {
                X_mat[i][j] = 1.0;
            }
        }
    }

    // X^T * X
    std::vector<std::vector<double>> XtX(k, std::vector<double>(k, 0.0));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int r = 0; r < n; ++r) {
                XtX[i][j] += X_mat[r][i] * X_mat[r][j];
            }
        }
    }

    // Invert (X^T * X)
    if (!invert_matrix(XtX)) {
        // Fallback: just return the original Y values at vertices
        std::vector<double> fallback(k);
        for(int i=0; i<k; ++i) fallback[i] = y[verts[i]];
        return fallback;
    }

    // X^T * Y
    std::vector<double> XtY(k, 0.0);
    for (int i = 0; i < k; ++i) {
        for (int r = 0; r < n; ++r) {
            XtY[i] += X_mat[r][i] * y[r];
        }
    }

    // Beta = (X^T * X)^-1 * X^T * Y
    std::vector<double> beta(k, 0.0);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            beta[i] += XtX[i][j] * XtY[j];
        }
    }

    return beta;
}

// One candidate model in the vertex-count ladder (see fit_trajectory).
struct CandidateModel {
    std::vector<int> verts;
    std::vector<double> fitted;
    double pval;
    bool recovery_ok;
};

std::vector<Vertex> fit_trajectory(const std::vector<int>& years,
                                   const std::vector<double>& values,
                                   const LandTrendrParams& params) {
    std::vector<Vertex> vertices;
    int n = years.size();
    if (n == 0 || values.empty() || n != values.size()) {
        return vertices;
    }

    // Too few observations to justify fitting/simplifying at all (LT-GEE's
    // minObservationsNeeded) -- pass the raw trajectory through unsegmented.
    if (n < std::max(2, params.min_observations_needed)) {
        for (int i = 0; i < n; ++i) {
            vertices.push_back({years[i], values[i]});
        }
        return vertices;
    }

    // 1. Remove spikes / desawtoothing
    std::vector<double> filtered_values = desawtooth(values, params.spike_threshold);

    // 2. Initial candidate vertices (for now, all indices)
    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    // 3. Vet vertices to max_segments + 1 + vertexCountOvershoot, giving the
    // initial candidate pool that much slack, then prune the overshoot away
    // before the scored model-selection ladder (step 4) begins -- LT-GEE's
    // vertexCountOvershoot. These intermediate overshoot vertex counts are
    // never themselves candidate models.
    int final_count = params.max_segments + 1;
    if (final_count > n) final_count = n;
    int overshoot_count = final_count + std::max(0, params.vertex_count_overshoot);
    if (overshoot_count > n) overshoot_count = n;

    std::vector<int> current_verts = vet_verts(years, filtered_values, all_indices, overshoot_count, 2.0);
    while (static_cast<int>(current_verts.size()) > final_count) {
        current_verts = vet_verts(years, filtered_values, all_indices, current_verts.size() - 1, 2.0);
    }

    // 4. Build the full ladder of candidate models, from max_segments+1 vertices
    // down to 2, removing the weakest (smallest-angle) vertex at each step -- same
    // vertex-pruning cdts always did, just no longer stopping at the first hit.
    // Every candidate is fit and scored so best_model_proportion (step 5) can
    // choose among the whole ladder instead of only the first "good enough" one.
    std::vector<CandidateModel> ladder;

    // Null model (mean of values) SSE, shared by every candidate's F-test.
    double mean_y = 0.0;
    for (double v : values) mean_y += v;
    mean_y /= n;
    double sse_null = 0.0;
    for (double v : values) {
        double err = v - mean_y;
        sse_null += err * err;
    }

    while (current_verts.size() >= 2) {
        std::vector<double> fitted = fit_piecewise_ols(years, values, current_verts);

        double sse = 0.0;
        for (int i = 0; i < n; ++i) {
            double interp_y = 0.0;
            int xi = years[i];
            for (size_t j = 0; j < current_verts.size() - 1; ++j) {
                int x0 = years[current_verts[j]];
                int x1 = years[current_verts[j+1]];
                if (xi >= x0 && xi <= x1) {
                    double y0 = fitted[j];
                    double y1 = fitted[j+1];
                    interp_y = y0 + (y1 - y0) * static_cast<double>(xi - x0) / (x1 - x0);
                    break;
                }
            }
            double err = values[i] - interp_y;
            sse += err * err;
        }

        int df_full = static_cast<int>(current_verts.size());
        int df_reduced = 1;
        double mse_full = sse / std::max(1, n - df_full);
        double f_stat = ((sse_null - sse) / (df_full - df_reduced)) / (mse_full > 0 ? mse_full : 1e-6);
        double pval = (df_full > df_reduced) ? f_pval(f_stat, df_full - df_reduced, n - df_full) : 1.0;

        // Recovery enforcement logic: a candidate whose fitted segments imply a
        // biologically-impossible fast green-up is not eligible for selection.
        bool recovery_ok = true;
        if (params.prevent_fast_recovery) {
            for (size_t i = 0; i + 1 < current_verts.size(); ++i) {
                double val_diff = fitted[i+1] - fitted[i];
                double yr_diff = static_cast<double>(years[current_verts[i+1]] - years[current_verts[i]]);
                if (val_diff > 0.0 && yr_diff > 0.0) {
                    double rate = val_diff / yr_diff;
                    if (rate > params.recovery_threshold) {
                        recovery_ok = false;
                        break;
                    }
                }
            }
        }

        ladder.push_back({current_verts, fitted, pval, recovery_ok});

        if (current_verts.size() <= 2) break;
        current_verts = vet_verts(years, filtered_values, all_indices, current_verts.size() - 1, 2.0);
    }

    if (ladder.empty()) {
        return vertices;
    }

    // 5. Model selection.
    // Eligible = statistically significant on its own (pval_threshold) AND not
    // recovery-violating. Among eligible candidates, LT-GEE's bestModelProportion
    // picks the one with the MOST vertices whose p-value is still within
    // best_model_proportion times the lowest p-value found among them --
    // i.e. prefer detail, but only where it's "nearly as good" a fit as the
    // single best-fitting candidate. If nothing clears pval_threshold, fall back
    // to the simplest (fewest-vertex) recovery-valid candidate, and if even that
    // is empty (every candidate violates the recovery constraint), fall back to
    // the very simplest candidate in the ladder regardless of recovery.
    std::vector<const CandidateModel*> eligible;
    for (const auto& c : ladder) {
        if (c.recovery_ok && c.pval <= params.pval_threshold) eligible.push_back(&c);
    }

    const CandidateModel* chosen = nullptr;
    if (!eligible.empty()) {
        double min_pval = std::numeric_limits<double>::max();
        for (auto* c : eligible) min_pval = std::min(min_pval, c->pval);
        double threshold = min_pval * params.best_model_proportion;

        size_t most_verts = 0;
        for (auto* c : eligible) {
            if (c->pval <= threshold && c->verts.size() > most_verts) {
                most_verts = c->verts.size();
                chosen = c;
            }
        }
    }

    if (chosen == nullptr) {
        // Nothing passed pval_threshold: fall back to the simplest recovery-valid
        // candidate (last one appended to the ladder), else the simplest overall.
        for (auto it = ladder.rbegin(); it != ladder.rend(); ++it) {
            if (it->recovery_ok) { chosen = &(*it); break; }
        }
        if (chosen == nullptr) chosen = &ladder.back();
    }

    for (size_t i = 0; i < chosen->verts.size(); ++i) {
        vertices.push_back({years[chosen->verts[i]], chosen->fitted[i]});
    }

    return vertices;
}

pybind11::tuple fit_trajectory_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Time]
    pybind11::array_t<int> years_array,     // Shape: [Time]
    LandTrendrParams params,
    double no_data_value,
    int n_jobs
) {
    auto val_buf = values_array.request();
    auto year_buf = years_array.request();
    
    int height = val_buf.shape[0];
    int width = val_buf.shape[1];
    int times = val_buf.shape[2];
    int num_pixels = height * width;
    
    double* val_ptr = static_cast<double*>(val_buf.ptr);
    int* year_ptr = static_cast<int*>(year_buf.ptr);
    
    std::vector<int> years(year_ptr, year_ptr + times);
    int max_vertices = params.max_segments + 1;
    
    // Output arrays
    pybind11::array_t<double> vertices_out({num_pixels, max_vertices, 2});
    auto vert_ptr = static_cast<double*>(vertices_out.request().ptr);
    
    pybind11::array_t<int> counts_out(num_pixels);
    auto counts_ptr = static_cast<int*>(counts_out.request().ptr);
    
    std::fill(vert_ptr, vert_ptr + (num_pixels * max_vertices * 2), no_data_value);
    std::fill(counts_ptr, counts_ptr + num_pixels, 0);
    
    #ifdef _OPENMP
    omp_set_num_threads(n_jobs > 0 ? n_jobs : std::max(1, omp_get_max_threads() - 1));
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int p = 0; p < num_pixels; ++p) {
        std::vector<double> pixel_values(times);
        bool has_valid_data = false;
        
        for (int t = 0; t < times; ++t) {
            double v = val_ptr[p * times + t];
            pixel_values[t] = v;
            if (v != no_data_value && !std::isnan(v)) has_valid_data = true;
        }
        
        if (!has_valid_data) continue;
        
        std::vector<Vertex> result = fit_trajectory(years, pixel_values, params);
        
        counts_ptr[p] = result.size();
        for (size_t i = 0; i < result.size() && (int)i < max_vertices; ++i) {
            vert_ptr[p * max_vertices * 2 + i * 2 + 0] = result[i].year;
            vert_ptr[p * max_vertices * 2 + i * 2 + 1] = result[i].value;
        }
    }
    
    return pybind11::make_tuple(vertices_out, counts_out);
}

} // namespace landtrendr
} // namespace cdts

