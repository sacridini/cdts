#include "mann_kendall.h"

#ifndef _OPENMP
#ifndef OMP_DUMMIES_DEFINED
#define OMP_DUMMIES_DEFINED
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x) (void)(x)
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace cdts {
namespace mannkendall {

namespace {

// ---- small numeric helpers -------------------------------------------

double median(std::vector<double> v) {
    if (v.empty()) return std::nan("");
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double hi = v[mid];
    if (n % 2 == 1) return hi;
    std::nth_element(v.begin(), v.begin() + mid - 1, v.begin() + mid);
    double lo = v[mid - 1];
    return 0.5 * (lo + hi);
}

std::vector<double> drop_nan(const std::vector<double>& x) {
    std::vector<double> out;
    out.reserve(x.size());
    for (double v : x) {
        if (!std::isnan(v)) out.push_back(v);
    }
    return out;
}

double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

// Exact standard normal CDF via std::erfc.
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// Inverse standard normal CDF: Acklam's rational approximation, refined
// with one Newton step against the exact erfc-based CDF above (~1e-15).
double norm_ppf(double p) {
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
        2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
        2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};

    const double p_low = 0.02425;
    double x;
    if (p < p_low) {
        double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= 1.0 - p_low) {
        double q = p - 0.5;
        double r = q*q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }

    double e = norm_cdf(x) - p;
    double u = e / norm_pdf(x);
    x = x - u / (1.0 + x * u / 2.0);
    return x;
}

// ---- Mann-Kendall building blocks (mirror pymannkendall's private fns) --

// __mk_score: sum of signs over all i<j pairs. O(n^2), inherent to the test.
double mk_score(const std::vector<double>& x) {
    int n = (int)x.size();
    double s = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (x[j] > x[i]) s += 1.0;
            else if (x[j] < x[i]) s -= 1.0;
        }
    }
    return s;
}

// __variance_s, with the tie correction.
double variance_s(const std::vector<double>& x) {
    int n = (int)x.size();
    std::vector<double> sorted_x = x;
    std::sort(sorted_x.begin(), sorted_x.end());

    std::vector<double> tie_sizes;
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && sorted_x[j] == sorted_x[i]) ++j;
        tie_sizes.push_back((double)(j - i));
        i = j;
    }

    int g = (int)tie_sizes.size();
    double nd = (double)n;
    if (g == n) {
        return nd*(nd-1.0)*(2.0*nd+5.0)/18.0;
    }
    double tie_sum = 0.0;
    for (double t : tie_sizes) tie_sum += t*(t-1.0)*(2.0*t+5.0);
    return (nd*(nd-1.0)*(2.0*nd+5.0) - tie_sum) / 18.0;
}

double z_score(double s, double var_s) {
    // No var_s guard here, matching pymannkendall exactly: sqrt of a
    // negative var_s (possible after the Hamed-Rao correction, which is not
    // clamped) yields NaN via IEEE754, same as numpy's np.sqrt.
    if (s > 0.0) return (s - 1.0) / std::sqrt(var_s);
    if (s < 0.0) return (s + 1.0) / std::sqrt(var_s);
    return 0.0;
}

struct PValueResult { double p; bool h; int trend; };

PValueResult p_value(double z, double alpha) {
    double p = 2.0 * (1.0 - norm_cdf(std::fabs(z)));
    bool h = std::fabs(z) > norm_ppf(1.0 - alpha / 2.0);
    int trend = 0;
    if (z < 0.0 && h) trend = -1;
    else if (z > 0.0 && h) trend = 1;
    return {p, h, trend};
}

// __acf: autocorrelation function up to nlags (inclusive), matching
// np.correlate(y, y, 'full')[n-1:] / n, normalized by lag-0 autocovariance.
std::vector<double> acf(const std::vector<double>& x, int nlags) {
    int n = (int)x.size();
    nlags = std::max(0, std::min(nlags, n - 1));

    double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = x[i] - mean;

    double acov0 = 0.0;
    for (int i = 0; i < n; ++i) acov0 += y[i] * y[i];
    acov0 /= n;

    std::vector<double> result(nlags + 1, 0.0);
    for (int lag = 0; lag <= nlags; ++lag) {
        double s = 0.0;
        for (int i = 0; i < n - lag; ++i) s += y[i] * y[i + lag];
        s /= n;
        result[lag] = (acov0 != 0.0) ? s / acov0 : s;
    }
    return result;
}

// scipy.stats.rankdata(method='average') equivalent.
std::vector<double> rankdata(const std::vector<double>& x) {
    int n = (int)x.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return x[a] < x[b]; });

    std::vector<double> ranks(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && x[idx[j]] == x[idx[i]]) ++j;
        double avg_rank = (i + j + 1) / 2.0; // 1-indexed average rank
        for (int k = i; k < j; ++k) ranks[idx[k]] = avg_rank;
        i = j;
    }
    return ranks;
}

// sens_slope(x_old): computed on the ORIGINAL (possibly NaN-gapped) series,
// using true positional spacing (j-i) as the time-step denominator, and
// nanmedian-style skipping of any pair touching a NaN - mirrors
// pymannkendall's sens_slope() operating on x_old rather than the
// NaN-compacted array, so missing observations don't collapse the time axis.
void sens_slope_full(const std::vector<double>& x_old, double& slope, double& intercept) {
    int n = (int)x_old.size();
    std::vector<double> pairwise;
    pairwise.reserve((size_t)n * (n - 1) / 2);
    for (int i = 0; i < n - 1; ++i) {
        if (std::isnan(x_old[i])) continue;
        for (int j = i + 1; j < n; ++j) {
            if (std::isnan(x_old[j])) continue;
            pairwise.push_back((x_old[j] - x_old[i]) / (double)(j - i));
        }
    }
    slope = median(pairwise);

    std::vector<double> valid_vals, valid_idx;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(x_old[i])) {
            valid_vals.push_back(x_old[i]);
            valid_idx.push_back((double)i);
        }
    }
    intercept = median(valid_vals) - median(valid_idx) * slope;
}

// Core original/Hamed-Rao/Yue-Wang test (SEASONAL is handled separately).
MKResult mk_test_impl(const std::vector<double>& y_old, MKMethod method, double alpha, int lag) {
    MKResult res;
    std::vector<double> x = drop_nan(y_old);
    int n = (int)x.size();
    if (n < 4) return res;

    double s = mk_score(x);
    double var_s = variance_s(x);
    double tau = s / (0.5 * n * (n - 1));

    double slope, intercept;
    sens_slope_full(y_old, slope, intercept);

    if (method == MKMethod::HAMED_RAO || method == MKMethod::YUE_WANG) {
        int L = (lag < 0) ? n : (lag + 1);
        L = std::max(1, std::min(L, n));

        std::vector<double> x_detrend(n);
        for (int i = 0; i < n; ++i) x_detrend[i] = x[i] - (double)(i + 1) * slope;

        double interval = norm_ppf(1.0 - alpha / 2.0) / std::sqrt((double)n);
        double sni = 0.0;

        if (method == MKMethod::HAMED_RAO) {
            std::vector<double> I = rankdata(x_detrend);
            std::vector<double> acf_1 = acf(I, L - 1);
            for (int i = 1; i < L; ++i) {
                if (!(acf_1[i] <= interval && acf_1[i] >= -interval)) {
                    sni += (double)(n - i) * (n - i - 1) * (n - i - 2) * acf_1[i];
                }
            }
            // No abs() on sni here: matches the current pymannkendall (>=1.4.3)
            // formula, which fixed an earlier abs(sni) bug present in some
            // older releases (negative net autocorrelation should be able to
            // reduce var_s, not just inflate it).
            double n_ns = 1.0 + (2.0 / ((double)n * (n - 1) * (n - 2))) * sni;
            var_s *= n_ns;
        } else {
            std::vector<double> acf_1 = acf(x_detrend, L - 1);
            for (int i = 1; i < L; ++i) {
                sni += (1.0 - (double)i / n) * acf_1[i];
            }
            double n_ns = 1.0 + 2.0 * sni;
            var_s *= n_ns;
        }
    }

    double z = z_score(s, var_s);
    PValueResult pv = p_value(z, alpha);

    res.trend = pv.trend;
    res.h = pv.h;
    res.p = pv.p;
    res.z = z;
    res.tau = tau;
    res.s = s;
    res.var_s = var_s;
    res.slope = slope;
    res.intercept = intercept;
    return res;
}

// Reshape a flat series into `period` season columns, padding the tail with
// NaN (mirrors pymannkendall's seasonal_test reshape).
std::vector<std::vector<double>> reshape_seasonal(const std::vector<double>& x_old, int period) {
    int n = (int)x_old.size();
    int rows = (n + period - 1) / period;
    std::vector<std::vector<double>> cols(period, std::vector<double>(rows, std::nan("")));
    for (int i = 0; i < n; ++i) {
        cols[i % period][i / period] = x_old[i];
    }
    return cols;
}

// seasonal_test == multivariate_test over the reshaped columns, with
// seasonal_sens_slope for the slope/intercept (pymannkendall).
MKResult seasonal_mk_test(const std::vector<double>& y_old, int period, double alpha) {
    MKResult res;
    if (period < 2) period = 1;

    std::vector<std::vector<double>> cols = reshape_seasonal(y_old, period);

    double s = 0.0, var_s = 0.0, denom = 0.0;
    for (int c = 0; c < period; ++c) {
        std::vector<double> col_clean = drop_nan(cols[c]);
        int nc = (int)col_clean.size();
        if (nc < 2) continue;
        s += mk_score(col_clean);
        var_s += variance_s(col_clean);
        denom += 0.5 * nc * (nc - 1);
    }
    if (denom <= 0.0) return res;

    double tau = s / denom;
    double z = z_score(s, var_s);
    PValueResult pv = p_value(z, alpha);

    // Seasonal Sen's slope: pool per-column pairwise slopes, where the
    // denominator is the ROW gap (i.e. in units of one full `period` cycle).
    std::vector<double> pooled;
    for (int c = 0; c < period; ++c) {
        const std::vector<double>& col = cols[c];
        int rows = (int)col.size();
        for (int i = 0; i < rows - 1; ++i) {
            if (std::isnan(col[i])) continue;
            for (int j = i + 1; j < rows; ++j) {
                if (std::isnan(col[j])) continue;
                pooled.push_back((col[j] - col[i]) / (double)(j - i));
            }
        }
    }
    double slope = median(pooled);

    std::vector<double> valid_vals, valid_flat_idx;
    for (int i = 0; i < (int)y_old.size(); ++i) {
        if (!std::isnan(y_old[i])) {
            valid_vals.push_back(y_old[i]);
            valid_flat_idx.push_back((double)i);
        }
    }
    double intercept = median(valid_vals) - (median(valid_flat_idx) / period) * slope;

    res.trend = pv.trend;
    res.h = pv.h;
    res.p = pv.p;
    res.z = z;
    res.tau = tau;
    res.s = s;
    res.var_s = var_s;
    res.slope = slope;
    res.intercept = intercept;
    return res;
}

} // namespace

MKResult mk_test(const std::vector<double>& y, MKMethod method, double alpha, int lag, int period) {
    if (method == MKMethod::SEASONAL) return seasonal_mk_test(y, period, alpha);
    return mk_test_impl(y, method, alpha, lag);
}

pybind11::tuple mk_test_single(std::vector<double> y, int method, double alpha, int lag, int period) {
    MKResult r = mk_test(y, static_cast<MKMethod>(method), alpha, lag, period);
    return pybind11::make_tuple(r.trend, r.h, r.p, r.z, r.tau, r.s, r.var_s, r.slope, r.intercept);
}

pybind11::array_t<double> fit_mann_kendall_batch(
    pybind11::array_t<double> values_array,
    int method_int,
    double alpha,
    int lag,
    int period,
    int min_valid,
    int n_jobs)
{
    auto buf = values_array.request();
    if (buf.ndim != 2) throw std::runtime_error("values_array must be 2D [pixels, time]");

    int n_pixels = (int)buf.shape[0];
    int n_time = (int)buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    const int n_metrics = 9; // trend, h, p, z, tau, s, var_s, slope, intercept
    pybind11::array_t<double> out_arr({n_metrics, n_pixels});
    double* out_ptr = static_cast<double*>(out_arr.request().ptr);
    for (int i = 0; i < n_metrics * n_pixels; ++i) out_ptr[i] = std::nan("");

    if (n_jobs <= 0) n_jobs = omp_get_max_threads();
    MKMethod method = static_cast<MKMethod>(method_int);

    #pragma omp parallel for num_threads(n_jobs)
    for (int p = 0; p < n_pixels; ++p) {
        std::vector<double> y(n_time);
        int valid_count = 0;
        for (int t = 0; t < n_time; ++t) {
            double v = ptr[(size_t)p * n_time + t];
            y[t] = v;
            if (!std::isnan(v)) ++valid_count;
        }
        if (valid_count < min_valid) continue;

        MKResult r = mk_test(y, method, alpha, lag, period);

        out_ptr[0 * n_pixels + p] = (double)r.trend;
        out_ptr[1 * n_pixels + p] = r.h ? 1.0 : 0.0;
        out_ptr[2 * n_pixels + p] = r.p;
        out_ptr[3 * n_pixels + p] = r.z;
        out_ptr[4 * n_pixels + p] = r.tau;
        out_ptr[5 * n_pixels + p] = r.s;
        out_ptr[6 * n_pixels + p] = r.var_s;
        out_ptr[7 * n_pixels + p] = r.slope;
        out_ptr[8 * n_pixels + p] = r.intercept;
    }

    return out_arr;
}

} // namespace mannkendall
} // namespace cdts
