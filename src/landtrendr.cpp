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

// Single-index version of find_correction.pro's per-point formula -- shared
// by desawtooth()'s initial full pass and its incremental per-step update
// (only a fixed-width window around whichever index was just corrected can
// change, so recomputing just those beats rebuilding the whole array).
inline void correction_at(const std::vector<double>& v, int i,
                           double& correction, double& prop_correction) {
    double diff_2 = std::abs(v[i - 1] - v[i + 1]);
    double diff_minus1 = std::abs(v[i] - v[i + 1]);
    double diff_plus1 = std::abs(v[i] - v[i - 1]);

    double md = std::max(diff_minus1, diff_plus1);
    if (md == 0.0) {
        md = diff_2; // avoid division by zero -- diff_2 is then 0 too, so prop_correction is 0.
    }

    prop_correction = (md > 0.0) ? (1.0 - diff_2 / md) : 0.0;
    correction = prop_correction * (((v[i - 1] + v[i + 1]) / 2.0) - v[i]);
}

std::vector<double> desawtooth(const std::vector<double>& vals, double stopat) {
    std::vector<double> v = vals;
    int n = static_cast<int>(v.size());
    if (n < 3) return v;

    std::vector<double> correction(n, 0.0), prop_correction(n, 0.0);
    for (int i = 1; i < n - 1; ++i) {
        correction_at(v, i, correction[i], prop_correction[i]);
    }

    // desawtooth.pro's while loop checks `prop` *before* each pass using the
    // value left over from the previous pass (seeded at 1.0), then always
    // applies that pass's strongest correction regardless of its own
    // magnitude -- only the *next* pass's entry is gated by it. So the very
    // first correction always happens, however small, and only further
    // corrections are conditional on the threshold. `wh_max` also tracks the
    // true (possibly negative) argmax of prop_correction, not max(0, ...).
    double prop = 1.0;
    while (prop > stopat) {
        int wh_max = 0;
        double max_prop = prop_correction[0];
        for (int i = 1; i < n; ++i) {
            if (prop_correction[i] > max_prop) {
                max_prop = prop_correction[i];
                wh_max = i;
            }
        }

        v[wh_max] = v[wh_max] + correction[wh_max];
        prop = max_prop;

        // Only entries whose formula reads v[wh_max] can have changed: i and
        // i's immediate neighbors read v[i-1..i+1], so indices wh_max-2 ..
        // wh_max+2 are the full blast radius.
        int lo = std::max(1, wh_max - 2);
        int hi = std::min(n - 2, wh_max + 2);
        for (int i = lo; i <= hi; ++i) {
            correction_at(v, i, correction[i], prop_correction[i]);
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

// OLS fit of y over x for the contiguous index range [lo, hi]; returns the
// per-point fitted values and the fit's SSE. Shared by find_vertices() below
// for both scoring a segment (SSE/span) and finding its best split point --
// tbcd_v2.pro's score_segments and the regression inside split_series.
struct SegRegression { std::vector<double> fitted; double sse; };

SegRegression regress_range(const std::vector<int>& x, const std::vector<double>& y, int lo, int hi) {
    int m = hi - lo + 1;
    double x0 = static_cast<double>(x[lo]);
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = lo; i <= hi; ++i) {
        double dx = x[i] - x0;
        sum_x += dx; sum_y += y[i]; sum_xx += dx * dx; sum_xy += dx * y[i];
    }
    double denom = m * sum_xx - sum_x * sum_x;
    double slope = 0.0, intercept = sum_y / m;
    if (std::abs(denom) > 1e-9) {
        slope = (m * sum_xy - sum_x * sum_y) / denom;
        intercept = (sum_y - slope * sum_x) / m;
    }
    SegRegression out;
    out.fitted.resize(m);
    out.sse = 0.0;
    for (int i = lo; i <= hi; ++i) {
        double f = intercept + slope * (x[i] - x0);
        out.fitted[i - lo] = f;
        double err = y[i] - f;
        out.sse += err * err;
    }
    return out;
}

// Finds the best new vertex within segment [lo, hi]: regress the segment and take
// the point with the largest absolute residual, excluding the segment's own
// endpoints (tbcd_v2.pro's split_series). If this is the rightmost segment in the
// current vertex set, the second-to-last point's candidacy is suppressed unless
// the series is still rising there, right at the trailing edge -- this blocks a
// spurious one-year "recovery" vertex from being manufactured at the very end of
// the record. Returns -1 if nothing valid is left to split on.
int split_series(const std::vector<int>& x, const std::vector<double>& y, int lo, int hi,
                  bool is_end_segment, bool disttest) {
    SegRegression reg = regress_range(x, y, lo, hi);
    int m = hi - lo + 1;
    std::vector<double> diff(m);
    for (int i = 0; i < m; ++i) diff[i] = std::abs(y[lo + i] - reg.fitted[i]);
    diff[0] = 0.0;
    diff[m - 1] = 0.0;
    if (disttest && is_end_segment && m >= 3 && !(y[hi] > y[hi - 1])) {
        diff[m - 2] = 0.0;
    }

    int best_idx = 0;
    double best_val = diff[0];
    for (int i = 1; i < m; ++i) {
        if (diff[i] > best_val) { best_val = diff[i]; best_idx = i; }
    }
    if (best_idx == 0) return -1;
    return lo + best_idx;
}

// Regression-based recursive vertex identification (Kennedy et al. 2010, Section
// 2.5.2's first strategy, complementary to vet_verts()'s angle-based culling
// below) -- a faithful port of tbcd_v2.pro's find_vertices. Starts with just the
// first/last observation as vertices, then repeatedly splits whichever current
// segment has the largest SSE/span (not raw SSE -- a long segment with moderate
// error can outrank a short one with more), retrying the next-worst segment if
// split_series rejects the split. Stops once `target_count` vertices are reached
// (LT-GEE's max_segments + 1 + vertexCountOvershoot), no segment has a valid
// split left, or (matching the original's own runaway guard) 20 vertices have
// been added.
std::vector<int> find_vertices(const std::vector<int>& x, const std::vector<double>& y,
                                int target_count, double distweightfactor) {
    int n = static_cast<int>(x.size());
    int m = std::min(target_count, n - 2);
    std::vector<int> verts = {0, n - 1};
    bool disttest = (distweightfactor != 0.0);
    int count = 0;

    while (static_cast<int>(verts.size()) < m) {
        int nseg = static_cast<int>(verts.size()) - 1;
        std::vector<double> mses(nseg, 0.0);
        for (int s = 0; s < nseg; ++s) {
            int lo = verts[s], hi = verts[s + 1];
            double span = static_cast<double>(hi - lo + 1);
            if (span > 2.0) {
                mses[s] = regress_range(x, y, lo, hi).sse / span;
            }
        }

        int split_at = -1;
        while (true) {
            int s = static_cast<int>(std::max_element(mses.begin(), mses.end()) - mses.begin());
            if (mses[s] <= 0.0) { split_at = -1; break; }

            bool is_end_segment = (s == nseg - 1);
            int candidate = split_series(x, y, verts[s], verts[s + 1], is_end_segment, disttest);
            if (candidate != -1) { split_at = candidate; break; }
            mses[s] = 0.0; // rejected -- try the next-worst segment
        }
        if (split_at == -1) break;

        verts.push_back(split_at);
        std::sort(verts.begin(), verts.end());

        if (++count > 20) break;
    }

    return verts;
}

// Simple matrix inversion for small matrices using Gauss-Jordan. Flat row-major
// storage (A[i*n+j]) instead of vector<vector<double>>: one allocation instead
// of n+1, and contiguous memory instead of n separately-heap-allocated rows.
bool invert_matrix(std::vector<double>& A, int n) {
    std::vector<double> I(n * n, 0.0);
    for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;

    for (int i = 0; i < n; ++i) {
        // Find pivot
        double max_el = std::abs(A[i * n + i]);
        int pivot = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(A[k * n + i]) > max_el) {
                max_el = std::abs(A[k * n + i]);
                pivot = k;
            }
        }
        if (max_el == 0.0) return false; // Singular

        // Swap rows
        if (pivot != i) {
            for (int j = 0; j < n; ++j) {
                std::swap(A[i * n + j], A[pivot * n + j]);
                std::swap(I[i * n + j], I[pivot * n + j]);
            }
        }

        // Scale row
        double diag = A[i * n + i];
        for (int j = 0; j < n; ++j) {
            A[i * n + j] /= diag;
            I[i * n + j] /= diag;
        }

        // Eliminate column
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k * n + i];
                for (int j = 0; j < n; ++j) {
                    A[k * n + j] -= factor * A[i * n + j];
                    I[k * n + j] -= factor * I[i * n + j];
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

    // Build design matrix X_mat (n x k), flat row-major.
    std::vector<double> X_mat(n * k, 0.0);
    for (int i = 0; i < n; ++i) {
        int xi = x[i];
        for (int j = 0; j < k; ++j) {
            int vj = x[verts[j]];
            if (j > 0 && xi >= x[verts[j-1]] && xi <= vj) {
                int v_prev = x[verts[j-1]];
                if (vj > v_prev) {
                    X_mat[i * k + j] = static_cast<double>(xi - v_prev) / (vj - v_prev);
                }
            } else if (j < k - 1 && xi >= vj && xi <= x[verts[j+1]]) {
                int v_next = x[verts[j+1]];
                if (v_next > vj) {
                    X_mat[i * k + j] = static_cast<double>(v_next - xi) / (v_next - vj);
                }
            } else if (xi == vj) {
                X_mat[i * k + j] = 1.0;
            }
        }
    }

    // X^T * X, flat k x k
    std::vector<double> XtX(k * k, 0.0);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            double s = 0.0;
            for (int r = 0; r < n; ++r) {
                s += X_mat[r * k + i] * X_mat[r * k + j];
            }
            XtX[i * k + j] = s;
        }
    }

    // Invert (X^T * X)
    if (!invert_matrix(XtX, k)) {
        // Fallback: just return the original Y values at vertices
        std::vector<double> fallback(k);
        for(int i=0; i<k; ++i) fallback[i] = y[verts[i]];
        return fallback;
    }

    // X^T * Y
    std::vector<double> XtY(k, 0.0);
    for (int i = 0; i < k; ++i) {
        double s = 0.0;
        for (int r = 0; r < n; ++r) {
            s += X_mat[r * k + i] * y[r];
        }
        XtY[i] = s;
    }

    // Beta = (X^T * X)^-1 * X^T * Y
    std::vector<double> beta(k, 0.0);
    for (int i = 0; i < k; ++i) {
        double s = 0.0;
        for (int j = 0; j < k; ++j) {
            s += XtX[i * k + j] * XtY[j];
        }
        beta[i] = s;
    }

    return beta;
}

// SSE of a piecewise-linear (verts, fitted) trajectory against the actual
// values, at every observation (not just at the vertices themselves).
//
// `verts` are indices into the sorted `x`/`y` arrays, and vet_verts()/
// take_out_weakest() never touch the first/last vertex, so
// verts.front()==0 and verts.back()==x.size()-1 always -- every observation
// therefore falls in exactly one segment's contiguous index range
// [verts[j], verts[j+1]], with no need to search for it. Each segment starts
// one past the previous one's end vertex so that shared vertex isn't counted
// twice.
double compute_full_sse(const std::vector<int>& x, const std::vector<double>& y,
                         const std::vector<int>& verts, const std::vector<double>& fitted) {
    double sse = 0.0;
    for (size_t j = 0; j + 1 < verts.size(); ++j) {
        int i0 = verts[j];
        int i1 = verts[j + 1];
        int x0 = x[i0];
        double span = static_cast<double>(x[i1] - x0);
        int start = (j == 0) ? i0 : i0 + 1;
        for (int i = start; i <= i1; ++i) {
            double interp_y = (span > 0.0)
                ? fitted[j] + (fitted[j + 1] - fitted[j]) * static_cast<double>(x[i] - x0) / span
                : fitted[j];
            double err = y[i] - interp_y;
            sse += err * err;
        }
    }
    return sse;
}

// Fits vertex y-values early-to-late, choosing per segment between a
// point-to-point line (endpoints pinned to the actual data values) and a
// simple regression line fit over that segment's own observations -- LT-GEE's
// flexible per-segment fitting (Kennedy et al. 2010, Section 2.5.3). For
// segments after the first, the regression is anchored at the already-fixed
// start value so consecutive segments stay connected.
std::vector<double> fit_piecewise_sequential(const std::vector<int>& x, const std::vector<double>& y,
                                              const std::vector<int>& verts) {
    int k = static_cast<int>(verts.size());
    std::vector<double> fitted(k, 0.0);
    if (k < 2) {
        if (k == 1) fitted[0] = y[verts[0]];
        return fitted;
    }

    for (int j = 0; j < k - 1; ++j) {
        int i0 = verts[j];
        int i1 = verts[j + 1];
        int x0 = x[i0];
        int x1 = x[i1];

        // Points in this segment are the contiguous index range [i0, i1] -- x is
        // sorted and verts are indices into it, so the range is iterated directly
        // (three times below) instead of first materializing an index vector.
        double span = static_cast<double>(x1 - x0);

        // Point-to-point candidate: the segment is just the line between the
        // (already-fixed, for j>0) start value and the next vertex's actual value.
        double p2p_y0 = (j == 0) ? y[verts[j]] : fitted[j];
        double p2p_y1 = y[verts[j + 1]];
        double p2p_slope = (span > 0.0) ? (p2p_y1 - p2p_y0) / span : 0.0;
        double p2p_sse = 0.0;
        for (int i = i0; i <= i1; ++i) {
            double err = y[i] - (p2p_y0 + p2p_slope * (x[i] - x0));
            p2p_sse += err * err;
        }

        // Regression candidate: free (2-parameter) OLS for the first segment,
        // anchored (1-parameter, pinned at the fixed start) for later ones.
        double reg_y0, reg_slope;
        if (j == 0) {
            double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
            int m = i1 - i0 + 1;
            for (int i = i0; i <= i1; ++i) {
                double dx = x[i] - x0;
                sum_x += dx; sum_y += y[i]; sum_xx += dx * dx; sum_xy += dx * y[i];
            }
            double denom = m * sum_xx - sum_x * sum_x;
            if (std::abs(denom) > 1e-9) {
                reg_slope = (m * sum_xy - sum_x * sum_y) / denom;
                reg_y0 = (sum_y - reg_slope * sum_x) / m;
            } else {
                reg_slope = p2p_slope;
                reg_y0 = p2p_y0;
            }
        } else {
            reg_y0 = fitted[j];
            double num = 0.0, den = 0.0;
            for (int i = i0; i <= i1; ++i) {
                double dx = x[i] - x0;
                num += dx * (y[i] - reg_y0);
                den += dx * dx;
            }
            reg_slope = (den > 1e-9) ? num / den : p2p_slope;
        }
        double reg_y1 = reg_y0 + reg_slope * span;
        double reg_sse = 0.0;
        for (int i = i0; i <= i1; ++i) {
            double err = y[i] - (reg_y0 + reg_slope * (x[i] - x0));
            reg_sse += err * err;
        }

        if (reg_sse < p2p_sse) {
            fitted[j] = reg_y0;
            fitted[j + 1] = reg_y1;
        } else {
            fitted[j] = p2p_y0;
            fitted[j + 1] = p2p_y1;
        }
    }
    return fitted;
}

// Removes whichever single interior vertex tbcd_v2.pro's take_out_weakest2 would
// remove in its "run_mse" branch. For each candidate, draws a straight line
// directly between its two flanking (already-fitted) vertex values -- skipping
// the candidate -- and scores it by that line's SSE against the actual
// observations in that local window, divided by the window's x-span. This is a
// strictly local computation (unlike refitting the whole trajectory per
// candidate), and it uses the CURRENT level's fitted vertex values as the line's
// endpoints, not a fresh regression.
//
// take_out_weakest2's OTHER branch -- when a segment violates the recovery
// threshold, surgically drop/smooth that specific vertex instead of picking one
// by local MSE -- was ported and tried here too (see find_recovery_violator
// below for the violator search it shared). It measured WORSE against a live-GEE
// baseline on real tile data (more false-positive loss/gain detections, weaker
// magnitude correlation) than just using this function unconditionally, so it
// isn't wired in; find_recovery_violator now only feeds the eligibility check in
// fit_trajectory_impl, not vertex removal.
std::vector<int> take_out_weakest(const std::vector<int>& x, const std::vector<double>& y,
                                   const std::vector<int>& verts, const std::vector<double>& vertvals) {
    int k = static_cast<int>(verts.size());
    if (k <= 2) return verts;

    double best_mse = std::numeric_limits<double>::max();
    int best_remove = -1;
    for (int i = 1; i < k - 1; ++i) {
        int i0 = verts[i - 1];
        int i1 = verts[i + 1];
        double span = static_cast<double>(x[i1] - x[i0]);
        if (span <= 0.0) continue;
        double slope = (vertvals[i + 1] - vertvals[i - 1]) / span;
        double mse = 0.0;
        for (int j = i0; j <= i1; ++j) {
            double fitted_y = vertvals[i - 1] + slope * static_cast<double>(x[j] - x[i0]);
            double err = y[j] - fitted_y;
            mse += err * err;
        }
        mse /= span;
        if (mse < best_mse) {
            best_mse = mse;
            best_remove = i;
        }
    }

    std::vector<int> result = verts;
    if (best_remove != -1) result.erase(result.begin() + best_remove);
    return result;
}

// Finds the worst recovery-threshold violator among a candidate's segments
// (tbcd_v2.pro's check_slopes): among segments whose scaled recovery rate
// exceeds the threshold, returns the position in `verts` of the segment's LATTER
// vertex -- assumed to be the culprit, matching the original's own stated
// assumption (tracking forward through time). Returns -1 if nothing violates.
// Used here only to gate eligibility (fit_trajectory_impl's recovery_ok) -- see
// take_out_weakest's docstring above for why it doesn't also drive removal.
int find_recovery_violator(const std::vector<int>& years, const std::vector<double>& fitted,
                            const std::vector<int>& verts, double recovery_threshold) {
    double range_of_vals = *std::max_element(fitted.begin(), fitted.end())
                          - *std::min_element(fitted.begin(), fitted.end());
    if (range_of_vals <= 0.0) return -1;

    int worst_seg = -1;
    double worst_scaled = -1.0;
    for (size_t i = 0; i + 1 < verts.size(); ++i) {
        double val_diff = fitted[i + 1] - fitted[i];
        double yr_diff = static_cast<double>(years[verts[i + 1]] - years[verts[i]]);
        // check_slopes.pro: "disturbance is always considered to have a positive
        // slope, and recovery a negative slope" -- fitted/mod_values are already
        // in modifier-space (increasing = disturbance), so the segments to
        // scrutinize here are the NEGATIVE-slope (recovery-direction) ones, not
        // positive ones.
        if (val_diff < 0.0 && yr_diff > 0.0) {
            double scaled_slope = std::abs(val_diff / yr_diff) / range_of_vals;
            if (scaled_slope > recovery_threshold && scaled_slope > worst_scaled) {
                worst_scaled = scaled_slope;
                worst_seg = static_cast<int>(i);
            }
        }
    }
    return (worst_seg == -1) ? -1 : (worst_seg + 1);
}

// One candidate model in the vertex-count ladder (see fit_trajectory).
struct CandidateModel {
    std::vector<int> verts;
    std::vector<double> fitted;
    double pval;
    double f_stat;
    bool recovery_ok;
};

TrajectoryResult fit_trajectory_impl(const std::vector<int>& years,
                                      const std::vector<double>& values,
                                      const LandTrendrParams& params) {
    TrajectoryResult out;
    std::vector<Vertex>& vertices = out.vertices;
    int n = years.size();
    if (n == 0 || values.empty() || n != values.size()) {
        return out;
    }

    // Too few observations to justify fitting/simplifying at all (LT-GEE's
    // minObservationsNeeded) -- pass the raw trajectory through unsegmented.
    // No fit was performed, so there's no meaningful RMSE (left at 0).
    if (n < std::max(2, params.min_observations_needed)) {
        for (int i = 0; i < n; ++i) {
            vertices.push_back({years[i], values[i]});
        }
        return out;
    }

    // 1. Remove spikes / desawtoothing
    std::vector<double> filtered_values = desawtooth(values, params.spike_threshold);

    // fit_trajectory_v2.pro then multiplies the desawtoothed series by `modifier`
    // ("this sets everything so disturbance is always positive [increasing]") --
    // desawtooth commutes exactly with a uniform sign flip (its correction/
    // prop_correction math only ever compares magnitudes or differences), so
    // applying modifier here rather than before desawtooth is bit-identical to
    // the original's order, but lets desawtooth() itself stay orientation-
    // agnostic. `mod_values` mirrors the same flip for the raw (non-desawtoothed)
    // series used everywhere below that isn't candidate-vertex discovery.
    std::vector<double> mod_values = values;
    if (params.modifier != 1.0) {
        for (auto& v : filtered_values) v *= params.modifier;
        for (auto& v : mod_values) v *= params.modifier;
    }

    // 2-3. Identify initial candidate vertices with LT-GEE's two complementary
    // strategies (Section 2.5.2): regression-based recursive splitting builds
    // a candidate pool of up to max_segments + 1 + vertexCountOvershoot
    // vertices, then angle-based culling (vet_verts) prunes that overshoot
    // slack back down to max_segments + 1 -- the fixed candidate set that
    // step 4's model-selection ladder starts from. Kennedy et al. note both
    // criteria matter jointly (neither alone reproduces their results).
    int final_count = params.max_segments + 1;
    if (final_count > n) final_count = n;
    int overshoot_count = final_count + std::max(0, params.vertex_count_overshoot);
    if (overshoot_count > n) overshoot_count = n;

    std::vector<int> candidate_verts = find_vertices(years, filtered_values, overshoot_count, 2.0);
    std::vector<int> current_verts = vet_verts(years, filtered_values, candidate_verts, final_count, 2.0);

    // 4. Build the full ladder of candidate models, from max_segments+1 vertices
    // down to 2. Each level's vertex set is fit with fit_piecewise_sequential
    // (point-to-point vs. anchored-regression per segment, Section 2.5.3); if
    // that fit isn't significant at pval_threshold, it's redone with the exact
    // global OLS solve, which -- since the piecewise-linear model is linear in
    // the vertex y-values -- is the closed-form equivalent of LT-GEE's
    // "simultaneous" Levenberg-Marquardt fallback fit, retained regardless of
    // its own p-value. Simplifying to the next level down removes whichever
    // vertex increases SSE the least (Section 2.5.4). Every candidate is fit
    // and scored so best_model_proportion (step 5) can choose among the whole
    // ladder instead of only the first "good enough" one.
    std::vector<CandidateModel> ladder;

    // Null model (mean of values) SSE, shared by every candidate's F-test.
    double mean_y = 0.0;
    for (double v : mod_values) mean_y += v;
    mean_y /= n;
    double sse_null = 0.0;
    for (double v : mod_values) {
        double err = v - mean_y;
        sse_null += err * err;
    }

    // p-of-F for a given (verts, fitted) pair against the shared null model above.
    // Degrees of freedom exactly match calc_fitting_stats3.pro: every call site in
    // tbcd_v2.pro passes n_predictors = 2*(vertex count) - 2, treating each of the
    // V-1 segments as 2 independently-fit parameters (slope+intercept) rather than
    // V shared vertex y-values. This matters a lot -- it roughly doubles df_regr
    // relative to the naive "V parameters" reading, which lowers ms_regr and thus
    // raises p-values (makes it harder to call a candidate significant) throughout
    // the whole ladder, biasing which model best_model_proportion ends up picking.
    struct PvalStat { double pval; double f_stat; };
    auto score_pval = [&](const std::vector<int>& verts, const std::vector<double>& fitted) -> PvalStat {
        double sse = compute_full_sse(years, mod_values, verts, fitted);
        int V = static_cast<int>(verts.size());
        int df_regr = 2 * V - 2;
        int df_resid = n - df_regr - 1;
        if (df_regr <= 0 || df_resid <= 0) return {1.0, 0.0};
        double ms_regr = (sse_null - sse) / df_regr;
        double ms_resid = sse / df_resid;
        // calc_fitting_stats3.pro: avoid a division glitch when ms_regr underflows.
        double f_stat = (ms_regr < 0.00001) ? 0.00001 : ms_regr / (ms_resid > 0 ? ms_resid : 1e-6);
        return {f_pval(f_stat, df_regr, df_resid), f_stat};
    };

    // n_vertices_orig: tbcd_v2.pro's `n_vertices`, the vertex count right after
    // vet_verts3 -- bounds the pick_best_model6/check_slopes retry loop below
    // (`increment gt n_vertices`), independent of how many candidates the
    // ladder itself ends up holding.
    int n_vertices_orig = static_cast<int>(current_verts.size());

    while (current_verts.size() >= 2) {
        // Primary fit: point-to-point/anchored-regression hybrid per segment.
        std::vector<double> fitted = fit_piecewise_sequential(years, mod_values, current_verts);
        PvalStat stat = score_pval(current_verts, fitted);

        // Fallback: if that fit isn't significant, retry with the exact global
        // OLS solve (LT-GEE's "simultaneous" LM fit) and keep it regardless.
        if (stat.pval > params.pval_threshold) {
            fitted = fit_piecewise_ols(years, mod_values, current_verts);
            stat = score_pval(current_verts, fitted);
        }

        // Recovery enforcement logic (tbcd_v2.pro's check_slopes): a candidate
        // whose fitted segments imply a biologically-impossible fast green-up is
        // not eligible for selection. Unconditional (no prevent_fast_recovery
        // gate -- see the note on find_recovery_violator) and scaled by the
        // fitted trajectory's own value range (check_slopes: scaled_slope =
        // abs(slope)/range(yfit)) -- recovery_threshold is a proportion of the
        // pixel's own dynamic range, not an absolute index-units-per-year rate.
        //
        // tbcd_v2.pro's take_out_weakest2 additionally uses this same violator
        // search to pick which vertex to drop when simplifying (surgically
        // removing/smoothing the culprit instead of the generic local-MSE
        // choice). That was tried here too and measured WORSE against a live-GEE
        // baseline on real tile data (more false-positive loss/gain detections,
        // weaker magnitude correlation) than just using take_out_weakest()
        // unconditionally, so it's deliberately not wired in below -- only the
        // eligibility check (this flag) is ported, not the removal targeting.
        bool recovery_ok = (find_recovery_violator(years, fitted, current_verts, params.recovery_threshold) == -1);

        ladder.push_back({current_verts, fitted, stat.pval, stat.f_stat, recovery_ok});

        if (current_verts.size() <= 2) break;
        current_verts = take_out_weakest(years, mod_values, current_verts, fitted);
    }

    if (ladder.empty()) {
        return out;
    }

    // 5. Model selection -- a faithful port of tbcd_v2.pro's actual selection
    // loop (not just pick_best_model6 in isolation): pick_best_model6 chooses
    // among ALL candidates by p-value/best_model_proportion regardless of
    // recovery validity; check_slopes is then run ONLY on that one selection;
    // if it fails, that candidate's working p-value is poisoned to 1.0 (never
    // eligible again) and selection retries -- up to n_vertices_orig times.
    // This is NOT equivalent to "pre-filter to recovery-valid, then pick
    // best": a candidate can still win even though a *different*, better-
    // fitting candidate exists, if every candidate that beats it on p-value
    // gets vetoed first. `work_pval` is ladder[i].pval, mutated by poisoning.
    std::vector<double> work_pval(ladder.size());
    for (size_t i = 0; i < ladder.size(); ++i) work_pval[i] = ladder[i].pval;

    // pick_best_model6 (use_fstat=0, the default path since 2009): threshold =
    // (2 - bestmodelproportion) * min(p_of_f) over the CURRENT (possibly
    // poisoned) working p-values; return the first (= most-vertex, since the
    // ladder is built most-to-least-detailed) candidate within that band, or
    // -1 if none qualifies -- which happens whenever best_model_proportion > 1
    // (the threshold then falls below even the minimum p-value itself), the
    // intended trigger for the min-f_stat fallback below.
    auto pick_best_model6 = [&]() -> int {
        double mn = *std::min_element(work_pval.begin(), work_pval.end());
        double thr = (2.0 - params.best_model_proportion) * mn;
        for (size_t i = 0; i < work_pval.size(); ++i) {
            if (work_pval[i] <= thr) return static_cast<int>(i);
        }
        return -1;
    };

    int best = 0;
    int increment = 0;
    bool notdone = true;
    while (notdone) {
        ++increment;
        int picked = pick_best_model6();
        if (picked != -1) {
            best = picked;
            bool ok = ladder[best].recovery_ok;
            if (!ok) work_pval[best] = 1.0;
            notdone = !ok && !(increment > n_vertices_orig);
        } else {
            // best_model_proportion > 1: fall back to the candidate with the
            // single lowest f_stat across the WHOLE original ladder (matching
            // tbcd_v2.pro exactly -- unconditional, not filtered by recovery).
            best = 0;
            double min_f = ladder[0].f_stat;
            for (size_t i = 1; i < ladder.size(); ++i) {
                if (ladder[i].f_stat < min_f) { min_f = ladder[i].f_stat; best = static_cast<int>(i); }
            }
            notdone = false;
        }
    }

    const CandidateModel* chosen = &ladder[best];

    // fitted values live in modifier-space throughout the ladder (see mod_values
    // above); multiply back by modifier (self-inverse, since it's always +-1.0)
    // so callers always see real, original-scale values -- matching
    // fit_trajectory_v2.pro's `best_model.yfit = best.yfit * modifier`.
    for (size_t i = 0; i < chosen->verts.size(); ++i) {
        vertices.push_back({years[chosen->verts[i]], chosen->fitted[i] * params.modifier});
    }

    // RMSE of the chosen model's fit against every observation -- LT-GEE's
    // per-pixel noise estimate for DSNR (see TrajectoryResult). Computed in
    // modifier-space against mod_values (matching chosen->fitted); the result
    // is identical either way since SSE is invariant to a uniform sign flip.
    double chosen_sse = compute_full_sse(years, mod_values, chosen->verts, chosen->fitted);
    int chosen_df = static_cast<int>(chosen->verts.size());
    out.rmse = std::sqrt(chosen_sse / std::max(1, n - chosen_df));

    return out;
}

// Convenience wrapper for callers that only need the vertices.
std::vector<Vertex> fit_trajectory(const std::vector<int>& years,
                                   const std::vector<double>& values,
                                   const LandTrendrParams& params) {
    return fit_trajectory_impl(years, values, params).vertices;
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

    // Per-pixel RMSE of the selected fit -- LT-GEE's DSNR is magnitude / this.
    pybind11::array_t<double> rmse_out(num_pixels);
    auto rmse_ptr = static_cast<double*>(rmse_out.request().ptr);

    std::fill(vert_ptr, vert_ptr + (num_pixels * max_vertices * 2), no_data_value);
    std::fill(counts_ptr, counts_ptr + num_pixels, 0);
    std::fill(rmse_ptr, rmse_ptr + num_pixels, 0.0);

    #ifdef _OPENMP
    omp_set_num_threads(n_jobs > 0 ? n_jobs : std::max(1, omp_get_max_threads() - 1));
    #pragma omp parallel
    #endif
    {
        // One reusable buffer per thread instead of one heap allocation per
        // pixel (millions of pixels otherwise reallocate this every iteration).
        std::vector<double> pixel_values(times);

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic)
        #endif
        for (int p = 0; p < num_pixels; ++p) {
            bool has_valid_data = false;

            for (int t = 0; t < times; ++t) {
                double v = val_ptr[p * times + t];
                pixel_values[t] = v;
                if (v != no_data_value && !std::isnan(v)) has_valid_data = true;
            }

            if (!has_valid_data) continue;

            TrajectoryResult result = fit_trajectory_impl(years, pixel_values, params);

            counts_ptr[p] = result.vertices.size();
            for (size_t i = 0; i < result.vertices.size() && (int)i < max_vertices; ++i) {
                vert_ptr[p * max_vertices * 2 + i * 2 + 0] = result.vertices[i].year;
                vert_ptr[p * max_vertices * 2 + i * 2 + 1] = result.vertices[i].value;
            }
            rmse_ptr[p] = result.rmse;
        }
    }

    return pybind11::make_tuple(vertices_out, counts_out, rmse_out);
}

} // namespace landtrendr
} // namespace cdts

