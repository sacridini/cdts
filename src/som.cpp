// Self-Organizing Map kernels.
//
// Both trainers are operation-by-operation ports of Python MiniSom 2.3.x
// (G. Vettigli, MIT license): `train_online` reproduces MiniSom.train() and
// `train_batch` reproduces MiniSom.train_batch_offline(). Every floating
// point expression keeps MiniSom/NumPy's evaluation order (numpy pairwise
// summation for the distance norm, argmin over the *square-rooted* distance,
// neighborhood formulas, decay formulas, sample-order accumulation) so that,
// given the same initial weights, the trained codebook matches MiniSom
// bit-for-bit on platforms where NumPy and the C runtime share exp()/pow().
#include "som.h"
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace cdts {
namespace som {

namespace {

// Keep exp()/pow() as scalar libm calls: an auto-vectorized SIMD variant
// (e.g. MSVC's SVML-style __vdecl_exp) can differ from NumPy by an ulp.
#if defined(_MSC_VER)
#define CDTS_NOINLINE __declspec(noinline)
#else
#define CDTS_NOINLINE __attribute__((noinline))
#endif

CDTS_NOINLINE double scalar_exp(double v) { return std::exp(v); }
CDTS_NOINLINE double scalar_square(double v) { return std::pow(v, 2.0); }

// Neurons accumulated together by one thread in the batch update.
constexpr int kNeuronBlock = 8;

int resolve_threads(int n_jobs) {
    return n_jobs > 0 ? n_jobs : std::max(1, omp_get_max_threads() - 1);
}

// numpy's DOUBLE_pairwise_sum (used by np.add.reduce along a contiguous axis).
double pairwise_sum(const double* a, int64_t n) {
    if (n < 8) {
        double res = 0.;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        int64_t i;
        for (i = 8; i < n - (n % 8); i += 8) {
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// Squared euclidean distance with numpy's rounding:
// add.reduce((x - w) * (x - w)) over the feature axis.
inline double squared_distance(const double* x, const double* w, int D, double* buf) {
    if (D < 8) {
        double res = 0.;
        for (int d = 0; d < D; ++d) {
            const double diff = x[d] - w[d];
            res += diff * diff;
        }
        return res;
    }
    if (D <= 128) {  // pairwise_sum's 8-accumulator block, fused with the squares
        double r[8];
        for (int j = 0; j < 8; ++j) {
            const double diff = x[j] - w[j];
            r[j] = diff * diff;
        }
        int d;
        for (d = 8; d < D - (D % 8); d += 8) {
            for (int j = 0; j < 8; ++j) {
                const double diff = x[d + j] - w[d + j];
                r[j] += diff * diff;
            }
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; d < D; ++d) {
            const double diff = x[d] - w[d];
            res += diff * diff;
        }
        return res;
    }
    for (int d = 0; d < D; ++d) {
        const double diff = x[d] - w[d];
        buf[d] = diff * diff;
    }
    return pairwise_sum(buf, D);
}

// argmin over sqrt(squared distance), first index wins ties (numpy argmin).
// sqrt is monotone, so it is only evaluated when the squared distance
// improves: two different squared distances can round to the same sqrt,
// and then the earlier neuron must be kept.
inline int find_bmu(const double* x, const double* w, int K, int D, double* buf) {
    int bmu = 0;
    double best_sq = std::numeric_limits<double>::infinity();
    double best = std::numeric_limits<double>::infinity();
    for (int k = 0; k < K; ++k) {
        const double sq = squared_distance(x, w + static_cast<int64_t>(k) * D, D, buf);
        if (sq < best_sq) {
            const double dist = std::sqrt(sq);
            if (dist < best) {
                best = dist;
                best_sq = sq;
                bmu = k;
            }
        }
    }
    return bmu;
}

double decay_learning_rate(int code, double lr, double t, double max_iter) {
    switch (code) {
        case 1: {  // inverse_decay_to_zero
            const double C = max_iter / 100.0;
            return lr * C / (C + t);
        }
        case 2:  // linear_decay_to_zero
            return lr * (1 - t / max_iter);
        default:  // asymptotic_decay
            return lr / (1 + t / (max_iter / 2));
    }
}

double decay_sigma(int code, double sigma, double t, double max_iter) {
    switch (code) {
        case 1: {  // inverse_decay_to_one
            const double C = (sigma - 1) / max_iter;
            return sigma / (1 + (t * C));
        }
        case 2:  // linear_decay_to_one
            return sigma + (t * (1 - sigma) / max_iter);
        default:  // asymptotic_decay
            return sigma / (1 + t / (max_iter / 2));
    }
}

// Map geometry. xx/yy are MiniSom's euclidean neuron coordinates
// (get_euclidean_coordinates(), flattened in (i, j) order); they encode the
// rectangular or hexagonal topology. The gaussian is separable, so exp() is
// only evaluated once per distinct coordinate value.
struct Grid {
    int x, y, K;
    std::vector<double> xx, yy;
    std::vector<double> ux, uy;   // distinct coordinate values
    std::vector<int> ix, iy;      // neuron -> index into ux / uy

    Grid(int x_, int y_, const double* xx_, const double* yy_)
        : x(x_), y(y_), K(x_ * y_), xx(xx_, xx_ + x_ * y_), yy(yy_, yy_ + x_ * y_) {
        index_unique(xx, ux, ix);
        index_unique(yy, uy, iy);
    }

    static void index_unique(const std::vector<double>& v, std::vector<double>& u, std::vector<int>& idx) {
        u = v;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        idx.resize(v.size());
        for (size_t k = 0; k < v.size(); ++k) {
            idx[k] = static_cast<int>(std::lower_bound(u.begin(), u.end(), v[k]) - u.begin());
        }
    }
};

// Neighborhood of winner c (flat index) with spread sigma, written to h[K].
// ax/ay are scratch buffers of size max(ux.size(), x) / max(uy.size(), y).
void neighborhood(int code, const Grid& g, int c, double sigma,
                  double* h, std::vector<double>& ax, std::vector<double>& ay) {
    const int ci = c / g.y;
    const int cj = c % g.y;
    switch (code) {
        case 1: {  // mexican_hat
            const double d = 2 * sigma * sigma;
            for (int k = 0; k < g.K; ++k) {
                const double p = scalar_square(g.xx[k] - g.xx[c]) + scalar_square(g.yy[k] - g.yy[c]);
                h[k] = scalar_exp(-p / d) * (1 - 2 / d * p);
            }
            break;
        }
        case 2: {  // bubble
            for (int i = 0; i < g.x; ++i)
                ax[i] = (i > ci - sigma && i < ci + sigma) ? 1.0 : 0.0;
            for (int j = 0; j < g.y; ++j)
                ay[j] = (j > cj - sigma && j < cj + sigma) ? 1.0 : 0.0;
            for (int i = 0; i < g.x; ++i)
                for (int j = 0; j < g.y; ++j) h[i * g.y + j] = ax[i] * ay[j];
            break;
        }
        case 3: {  // triangle
            for (int i = 0; i < g.x; ++i) {
                const double v = static_cast<double>(-std::abs(ci - i)) + sigma;
                ax[i] = v < 0 ? 0. : v;
            }
            for (int j = 0; j < g.y; ++j) {
                const double v = static_cast<double>(-std::abs(cj - j)) + sigma;
                ay[j] = v < 0 ? 0. : v;
            }
            for (int i = 0; i < g.x; ++i)
                for (int j = 0; j < g.y; ++j) h[i * g.y + j] = ax[i] * ay[j];
            break;
        }
        default: {  // gaussian
            const double d = 2 * sigma * sigma;
            const double xc = g.xx[c];
            const double yc = g.yy[c];
            for (size_t m = 0; m < g.ux.size(); ++m) ax[m] = scalar_exp(-scalar_square(g.ux[m] - xc) / d);
            for (size_t m = 0; m < g.uy.size(); ++m) ay[m] = scalar_exp(-scalar_square(g.uy[m] - yc) / d);
            for (int k = 0; k < g.K; ++k) h[k] = ax[g.ix[k]] * ay[g.iy[k]];
            break;
        }
    }
}

struct Inputs {
    double* w;
    const double* data;
    int64_t N;
    int D;
};

Inputs check_inputs(py::array_t<double, py::array::c_style>& weights_array,
                    py::array_t<double, py::array::c_style | py::array::forcecast>& data_array,
                    int x, int y) {
    if (weights_array.ndim() != 3 || weights_array.shape(0) != x || weights_array.shape(1) != y)
        throw std::invalid_argument("weights must have shape (x, y, input_len)");
    if (data_array.ndim() != 2 || data_array.shape(1) != weights_array.shape(2))
        throw std::invalid_argument("data must have shape (n_samples, input_len)");
    if (data_array.shape(0) < 1)
        throw std::invalid_argument("data must contain at least one sample");
    return {weights_array.mutable_data(), data_array.data(),
            static_cast<int64_t>(data_array.shape(0)), static_cast<int>(data_array.shape(1))};
}

Grid make_grid(py::array_t<double, py::array::c_style>& weights_array,
               py::array_t<double, py::array::c_style | py::array::forcecast>& xx_array,
               py::array_t<double, py::array::c_style | py::array::forcecast>& yy_array) {
    const int x = static_cast<int>(weights_array.shape(0));
    const int y = static_cast<int>(weights_array.shape(1));
    if (xx_array.size() != static_cast<py::ssize_t>(x) * y || yy_array.size() != static_cast<py::ssize_t>(x) * y)
        throw std::invalid_argument("xx/yy must contain one coordinate per neuron");
    return Grid(x, y, xx_array.data(), yy_array.data());
}

} // namespace

void train_online(
    py::array_t<double, py::array::c_style> weights_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> data_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> order_array,
    int num_iteration,
    bool use_epochs,
    double learning_rate,
    double sigma,
    int lr_decay,
    int sigma_decay,
    int neighborhood_code,
    py::array_t<double, py::array::c_style | py::array::forcecast> xx_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> yy_array)
{
    if (weights_array.ndim() != 3) throw std::invalid_argument("weights must have shape (x, y, input_len)");
    const Grid grid = make_grid(weights_array, xx_array, yy_array);
    const Inputs in = check_inputs(weights_array, data_array, grid.x, grid.y);
    const int64_t* order = order_array.data();
    const int64_t L = order_array.size();
    for (int64_t s = 0; s < L; ++s) {
        if (order[s] < 0 || order[s] >= in.N) throw std::out_of_range("sample index out of range");
    }
    if (num_iteration < 1) throw std::invalid_argument("num_iteration must be >= 1");

    const int K = grid.K;
    const int D = in.D;
    const int64_t epochs = use_epochs ? num_iteration : 1;
    const double max_iter = static_cast<double>(num_iteration);

    py::gil_scoped_release release;

    std::vector<double> buf(D), h(K);
    std::vector<double> ax(std::max<size_t>(grid.ux.size(), grid.x));
    std::vector<double> ay(std::max<size_t>(grid.uy.size(), grid.y));
    double* w = in.w;

    for (int64_t e = 0; e < epochs; ++e) {
        for (int64_t s = 0; s < L; ++s) {
            const double* xs = in.data + order[s] * D;
            const int bmu = find_bmu(xs, w, K, D, buf.data());
            // MiniSom's decay "t": the step index, or the epoch with use_epochs.
            const double t = static_cast<double>(use_epochs ? e : s);
            const double eta = decay_learning_rate(lr_decay, learning_rate, t, max_iter);
            const double sig = decay_sigma(sigma_decay, sigma, t, max_iter);
            neighborhood(neighborhood_code, grid, bmu, sig, h.data(), ax, ay);
            for (int k = 0; k < K; ++k) {
                const double gk = h[k] * eta;
                if (gk == 0.0) continue;  // w += 0 * (x - w) leaves w unchanged
                double* wk = w + static_cast<int64_t>(k) * D;
                for (int d = 0; d < D; ++d) wk[d] += gk * (xs[d] - wk[d]);
            }
        }
    }
}

void train_batch(
    py::array_t<double, py::array::c_style> weights_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> data_array,
    int num_iteration,
    double learning_rate,
    double sigma,
    int lr_decay,
    int sigma_decay,
    int neighborhood_code,
    py::array_t<double, py::array::c_style | py::array::forcecast> xx_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> yy_array,
    int n_jobs)
{
    if (weights_array.ndim() != 3) throw std::invalid_argument("weights must have shape (x, y, input_len)");
    const Grid grid = make_grid(weights_array, xx_array, yy_array);
    const Inputs in = check_inputs(weights_array, data_array, grid.x, grid.y);
    if (num_iteration < 1) throw std::invalid_argument("num_iteration must be >= 1");

    const int K = grid.K;
    const int D = in.D;
    const int64_t N = in.N;
    const int threads = resolve_threads(n_jobs);
    const double max_iter = static_cast<double>(num_iteration);
    double* w = in.w;

    py::gil_scoped_release release;

    std::vector<int> bmu(N);
    // GT[k * K + c]: neighborhood value at neuron k when c is the winner
    // (only rows of winning neurons are filled and read).
    std::vector<double> GT(static_cast<size_t>(K) * K);
    std::vector<char> is_winner(K);
    std::vector<double> h(K);
    std::vector<double> num(static_cast<size_t>(K) * D), den(K);
    const int n_blocks = (K + kNeuronBlock - 1) / kNeuronBlock;

    for (int it = 0; it < num_iteration; ++it) {
        const double t = static_cast<double>(it);
        const double lr = decay_learning_rate(lr_decay, learning_rate, t, max_iter);
        const double sig = decay_sigma(sigma_decay, sigma, t, max_iter);

        #pragma omp parallel num_threads(threads)
        {
            std::vector<double> buf(D);
            #pragma omp for schedule(dynamic, 512)
            for (int64_t i = 0; i < N; ++i) {
                bmu[i] = find_bmu(in.data + i * D, w, K, D, buf.data());
            }
        }

        std::fill(is_winner.begin(), is_winner.end(), 0);
        for (int64_t i = 0; i < N; ++i) is_winner[bmu[i]] = 1;
        {
            std::vector<double> ax(std::max<size_t>(grid.ux.size(), grid.x));
            std::vector<double> ay(std::max<size_t>(grid.uy.size(), grid.y));
            for (int c = 0; c < K; ++c) {
                if (!is_winner[c]) continue;
                neighborhood(neighborhood_code, grid, c, sig, h.data(), ax, ay);
                for (int k = 0; k < K; ++k) GT[static_cast<size_t>(k) * K + c] = h[k];
            }
        }

        // numerator += g * sample; denominator += g, in sample order (as
        // MiniSom does) so each neuron's sums round identically and do not
        // depend on the thread count. Neurons are independent: blocks of them
        // are accumulated in thread-local buffers.
        #pragma omp parallel num_threads(threads)
        {
            std::vector<double> acc(static_cast<size_t>(kNeuronBlock) * D);
            double acc_den[kNeuronBlock];
            #pragma omp for schedule(dynamic, 1)
            for (int b = 0; b < n_blocks; ++b) {
                const int k0 = b * kNeuronBlock;
                const int nk = std::min(kNeuronBlock, K - k0);
                std::fill(acc.begin(), acc.end(), 0.0);
                std::fill(acc_den, acc_den + kNeuronBlock, 0.0);
                for (int64_t i = 0; i < N; ++i) {
                    const double* xs = in.data + i * D;
                    const int c = bmu[i];
                    for (int j = 0; j < nk; ++j) {
                        const double g = GT[static_cast<size_t>(k0 + j) * K + c];
                        if (g == 0.0) continue;  // adds exact zeros
                        double* a = acc.data() + static_cast<size_t>(j) * D;
                        for (int d = 0; d < D; ++d) a[d] += g * xs[d];
                        acc_den[j] += g;
                    }
                }
                std::copy(acc.begin(), acc.begin() + static_cast<size_t>(nk) * D,
                          num.begin() + static_cast<size_t>(k0) * D);
                std::copy(acc_den, acc_den + nk, den.begin() + k0);
            }
        }

        for (int k = 0; k < K; ++k) {
            if (!(den[k] > 0)) continue;
            double* wk = w + static_cast<int64_t>(k) * D;
            const double* nk = num.data() + static_cast<size_t>(k) * D;
            for (int d = 0; d < D; ++d) {
                const double new_w = nk[d] / den[k];
                wk[d] = (1 - lr) * wk[d] + lr * new_w;
            }
        }
    }
}

py::array_t<int> predict_bmus(
    py::array_t<double, py::array::c_style | py::array::forcecast> data_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights_array,
    int n_jobs)
{
    if (weights_array.ndim() != 3) throw std::invalid_argument("weights must have shape (x, y, input_len)");
    if (data_array.ndim() != 2 || data_array.shape(1) != weights_array.shape(2))
        throw std::invalid_argument("data must have shape (n_samples, input_len)");

    const int64_t N = data_array.shape(0);
    const int D = static_cast<int>(data_array.shape(1));
    const int K = static_cast<int>(weights_array.shape(0) * weights_array.shape(1));
    const double* ptr_d = data_array.data();
    const double* ptr_w = weights_array.data();

    auto result = py::array_t<int>(N);
    int* res_ptr = result.mutable_data();
    const int threads = resolve_threads(n_jobs);

    py::gil_scoped_release release;
    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> buf(D);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            res_ptr[i] = find_bmu(ptr_d + i * D, ptr_w, K, D, buf.data());
        }
    }
    return result;
}

} // namespace som
} // namespace cdts
