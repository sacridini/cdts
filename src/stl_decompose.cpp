#include "stl_decompose.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace cdts {
namespace stl {

namespace {

int nextodd(double x) {
    int xi = (int)std::lround(x);
    return (xi % 2 == 0) ? xi + 1 : xi;
}

// Local regression (loess) estimate at a single point, matching stl.c's
// `stlest` - tricube-weighted degree-0/1 fit over [nleft, nright] (1-indexed,
// as in the original). Returns false (leaving *ys untouched) if the window
// carries no usable weight, matching the original's fallback-to-raw-value
// behavior at the call site.
bool loess_est(const double* y, int n, int len, int ideg, double xs, double* ys,
                int nleft, int nright, std::vector<double>& w) {
    (void)n;
    double range = (double)(n - 1);
    double h = std::max(xs - (double)nleft, (double)nright - xs);
    if (len > n) h += (double)((len - n) / 2);
    double h9 = h * 0.999, h1 = h * 0.001, a = 0.0;

    for (int j = nleft - 1; j < nright; ++j) {
        double r = std::fabs((double)(j + 1) - xs);
        if (r <= h9) {
            if (r <= h1) {
                w[j] = 1.0;
            } else {
                double u = r / h;
                double v = 1.0 - u * u * u;
                w[j] = v * v * v;
            }
            a += w[j];
        } else {
            w[j] = 0.0;
        }
    }

    if (a <= 0.0) return false;

    for (int j = nleft - 1; j < nright; ++j) w[j] /= a;

    if (h > 0.0 && ideg > 0) {
        double xbar = 0.0;
        for (int j = nleft - 1; j < nright; ++j) xbar += w[j] * (double)(j + 1);
        double b = xs - xbar, c = 0.0;
        for (int j = nleft - 1; j < nright; ++j) {
            double d = (double)(j + 1) - xbar;
            c += w[j] * d * d;
        }
        if (std::sqrt(c) > range * 0.001) {
            b /= c;
            for (int j = nleft - 1; j < nright; ++j) {
                w[j] *= b * ((double)(j + 1) - xbar) + 1.0;
            }
        }
    }

    double ysum = 0.0;
    for (int j = nleft - 1; j < nright; ++j) ysum += w[j] * y[j];
    *ys = ysum;
    return true;
}

// Loess smoother over the whole series (matching stl.c's `stless`): computes
// the fit directly every `njump`-th point and linearly interpolates in
// between (the same speed shortcut R's stl() itself uses by default via
// s.jump/t.jump/l.jump).
void loess_smooth(const double* y, int n, int len, int ideg, int njump, double* ys) {
    if (n < 2) { ys[0] = y[0]; return; }

    std::vector<double> w((size_t)n);
    int newnj = std::min(njump, n - 1);
    int nleft = 1, nright = 1;

    if (len >= n) {
        nleft = 1;
        nright = n;
        for (int i = 0; i < n; i += newnj) {
            if (!loess_est(y, n, len, ideg, (double)(i + 1), &ys[i], nleft, nright, w)) {
                ys[i] = y[i];
            }
        }
    } else if (newnj == 1) {
        int nsh = (len + 1) / 2;
        nleft = 1;
        nright = len;
        for (int i = 0; i < n; ++i) {
            if (i + 1 > nsh && nright != n) { ++nleft; ++nright; }
            if (!loess_est(y, n, len, ideg, (double)(i + 1), &ys[i], nleft, nright, w)) {
                ys[i] = y[i];
            }
        }
    } else {
        int nsh = (len + 1) / 2;
        for (int i = 0; i < n; i += newnj) {
            if (i + 1 < nsh) {
                nleft = 1;
                nright = len;
            } else if (i >= n - nsh) {
                nleft = n - len + 1;
                nright = n;
            } else {
                nleft = i + 1 - nsh + 1;
                nright = len + i + 1 - nsh;
            }
            if (!loess_est(y, n, len, ideg, (double)(i + 1), &ys[i], nleft, nright, w)) {
                ys[i] = y[i];
            }
        }
    }

    if (newnj != 1) {
        for (int i = 0; i < n - newnj; i += newnj) {
            double delta = (ys[i + newnj] - ys[i]) / (double)newnj;
            for (int j = i + 1; j <= i + newnj - 1; ++j) {
                ys[j] = ys[i] + delta * (double)(j - i);
            }
        }
        int k = ((n - 1) / newnj) * newnj; // 0-based
        if (k != n - 1) {
            if (!loess_est(y, n, len, ideg, (double)n, &ys[n - 1], nleft, nright, w)) {
                ys[n - 1] = y[n - 1];
            }
            if (k != n - 1) {
                double delta = (ys[n - 1] - ys[k]) / (double)(n - 1 - k);
                for (int j = k + 1; j < n - 1; ++j) {
                    ys[j] = ys[k] + delta * (double)(j - k);
                }
            }
        }
    }
}

// Running mean of window `len`, matching stl.c's `stlma`.
void moving_average(const double* x, int n, int len, double* ave) {
    double flen = (double)len, v = 0.0;
    for (int i = 0; i < len; ++i) v += x[i];
    ave[0] = v / flen;
    int newn = n - len + 1;
    if (newn > 1) {
        int k = len, m = 0;
        for (int j = 1; j < newn; ++j, ++k, ++m) {
            v += x[k] - x[m];
            ave[j] = v / flen;
        }
    }
}

// Low-pass filter: three moving averages of length np, np, 3, matching
// stl.c's `stlfts`. x has length n; trend (output) has length n - 2*np.
void low_pass_filter(const double* x, int n, int np, double* trend, double* work) {
    moving_average(x, n, np, trend);
    moving_average(trend, n - np + 1, np, work);
    moving_average(work, n - 2 * np + 2, 3, trend);
}

// Cycle-subseries smoothing, matching stl.c's `stlss`: smooths each of the
// `np` subseries (e.g. all Januaries, all Februaries, ...) independently via
// `loess_smooth`, extended by one point on each end. Writes into `season`,
// length n + 2*np, indexed as season[m*np + j].
void subseries_smooth(const double* y, int n, int np, int ns, int isdeg, int nsjump,
                       double* season, std::vector<double>& work1, std::vector<double>& work2,
                       std::vector<double>& work4) {
    for (int j = 0; j < np; ++j) {
        int k = (n - (j + 1)) / np + 1;
        for (int i = 0; i < k; ++i) work1[i] = y[i * np + j];

        loess_smooth(work1.data(), k, ns, isdeg, nsjump, &work2[1]);

        int nright = std::min(ns, k);
        if (!loess_est(work1.data(), k, ns, isdeg, 0.0, &work2[0], 1, nright, work4)) {
            work2[0] = work2[1];
        }
        int nleft = std::max(1, k - ns + 1);
        if (!loess_est(work1.data(), k, ns, isdeg, (double)(k + 1), &work2[k + 1], nleft, k, work4)) {
            work2[k + 1] = work2[k];
        }
        for (int m = 0; m < k + 2; ++m) season[m * np + j] = work2[m];
    }
}

// One full STL "inner loop" pass (matching stl.c's `stlstp`, with the
// robustness-weight plumbing removed since it's always disabled - see
// stl.h's Scope note): repeats the detrend -> subseries-smooth -> low-pass
// -> deseasonalize -> trend-smooth cycle `niter` times.
void inner_loop(const double* y, int n, int np, int ns, int nt, int nl,
                 int isdeg, int itdeg, int ildeg, int nsjump, int ntjump, int nljump,
                 int niter, double* season, double* trend) {
    int n2p = n + 2 * np;
    std::vector<double> work(n2p, 0.0), work2(n2p, 0.0), work3(n2p, 0.0), work4(n2p, 0.0), work5(n2p, 0.0);
    std::vector<double> sub_work1(n2p, 0.0);

    for (int iter = 0; iter < niter; ++iter) {
        for (int i = 0; i < n; ++i) work[i] = y[i] - trend[i];

        subseries_smooth(work.data(), n, np, ns, isdeg, nsjump, work2.data(), sub_work1, work3, work5);
        low_pass_filter(work2.data(), n2p, np, work3.data(), work.data());
        loess_smooth(work3.data(), n, nl, ildeg, nljump, work.data());

        for (int i = 0; i < n; ++i) season[i] = work2[np + i] - work[i];
        for (int i = 0; i < n; ++i) work[i] = y[i] - season[i];
        loess_smooth(work.data(), n, nt, itdeg, ntjump, trend);
    }
}

} // namespace

std::vector<double> periodic_seasonal(const std::vector<double>& y, int period) {
    int n = (int)y.size();
    if (period < 2 || n <= 2 * period) {
        throw std::runtime_error("stl::periodic_seasonal: series is not periodic or has less than two periods");
    }

    // "periodic" case: s.window forced to 10*n+1 (effectively spans the
    // whole subseries) with s.degree = 0 - matches R/stl.R exactly.
    int ns = 10 * n + 1;
    int sdeg = 0;
    int twindow = nextodd(std::ceil(1.5 * (double)period / (1.0 - 1.5 / (double)ns)));
    int tdeg = 1;
    int lwindow = nextodd((double)period);
    int ldeg = tdeg;

    ns = std::max(3, ns); if (ns % 2 == 0) ++ns;
    twindow = std::max(3, twindow); if (twindow % 2 == 0) ++twindow;
    lwindow = std::max(3, lwindow); if (lwindow % 2 == 0) ++lwindow;

    int sjump = std::max(1, (int)std::ceil(ns / 10.0));
    int tjump = std::max(1, (int)std::ceil(twindow / 10.0));
    int ljump = std::max(1, (int)std::ceil(lwindow / 10.0));

    std::vector<double> season(n, 0.0), trend(n, 0.0);
    inner_loop(y.data(), n, period, ns, twindow, lwindow, sdeg, tdeg, ldeg,
               sjump, tjump, ljump, /*niter=*/2, season.data(), trend.data());

    // "periodic" post-processing (matching R/stl.R): force the seasonal
    // component to be exactly periodic by averaging per cycle position -
    // this also means the loess machinery above only needs to be
    // approximately right, not bit-exact, for this specific (huge s.window,
    // degree 0) case: any residual non-uniformity across cycles is washed
    // out here.
    std::vector<double> cycle_sum(period, 0.0);
    std::vector<int> cycle_cnt(period, 0);
    for (int i = 0; i < n; ++i) {
        int pos = i % period;
        cycle_sum[pos] += season[i];
        cycle_cnt[pos] += 1;
    }
    std::vector<double> cycle_mean(period);
    for (int p = 0; p < period; ++p) cycle_mean[p] = cycle_sum[p] / (double)cycle_cnt[p];

    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) result[i] = cycle_mean[i % period];
    return result;
}

} // namespace stl
} // namespace cdts
