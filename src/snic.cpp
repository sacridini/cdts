#include "snic.h"

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

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

namespace cdts {
namespace snic {

namespace {

// Heap node: pixel index, cluster label, distance.
struct Node {
    std::uint32_t i;
    std::uint32_t k;
    double d;
};

// 1-based binary min-heap: push sifts up while the parent is strictly
// greater, pop sifts the last node down through the strictly smaller child
// (the left one on ties). With equal distances this order decides which
// cluster wins a pixel both reach at the same cost; it matches the reference
// implementation's.
class Heap {
public:
    explicit Heap(std::size_t capacity) : nodes_(capacity + 1) {}

    bool empty() const { return len_ == 0; }

    void push(std::uint32_t ind, std::uint32_t klab, double dist) {
        if (len_ + 1 >= nodes_.size()) nodes_.resize(nodes_.size() * 2);
        std::size_t i = ++len_;
        std::size_t j = i / 2;
        while (i > 1 && nodes_[j].d > dist) {
            nodes_[i] = nodes_[j];
            i = j;
            j /= 2;
        }
        nodes_[i] = Node{ind, klab, dist};
    }

    Node pop() {
        const Node top = nodes_[1];
        const Node last = nodes_[len_];
        --len_;
        std::size_t i = 1;
        while (true) {
            const std::size_t j = 2 * i;
            std::size_t k = 0;  // 0 == `last` stays here
            double best = last.d;
            if (j <= len_ && nodes_[j].d < best) { k = j; best = nodes_[j].d; }
            if (j + 1 <= len_ && nodes_[j + 1].d < best) k = j + 1;
            if (k == 0) break;
            nodes_[i] = nodes_[k];
            i = k;
        }
        nodes_[i] = last;
        return top;
    }

private:
    std::vector<Node> nodes_;
    std::size_t len_ = 0;
};

constexpr std::int32_t UNLABELED = -1;
constexpr std::int32_t MASKED = -2;

struct Tile {
    int r0, c0, h, w;
    std::vector<std::int32_t> seed_ids;  // global seed indices, input order
};

// Per-seed accumulators shared by all tiles (tiles own disjoint seeds).
struct Accumulators {
    double* sums;        // [K, F]
    double* sum_row;     // [K]
    double* sum_col;     // [K]
    std::int64_t* size;  // [K]
};

template <typename T>
void segment_tile(const T* data, int H, int W, int F, const Tile& tile,
                  const std::int32_t* seeds, double compactness,
                  bool parallel_gather, int n_jobs,
                  std::int32_t* labels_out, Accumulators acc)
{
    using VecT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using VecD = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    const int w = tile.w;
    const int h = tile.h;
    const std::size_t n = static_cast<std::size_t>(w) * h;
    const std::size_t plane = static_cast<std::size_t>(H) * W;

    // Gather the tile pixel-major ([pixel, feature]) so every distance reads
    // one contiguous run of F values instead of F planes N apart, and mask
    // pixels with any NaN channel.
    std::vector<T> px(n * F);
    std::vector<std::int32_t> labels(n, UNLABELED);
    #pragma omp parallel for num_threads(n_jobs) if (parallel_gather)
    for (int r = 0; r < h; ++r) {
        T* row = px.data() + static_cast<std::size_t>(r) * w * F;
        const std::size_t src_row = static_cast<std::size_t>(tile.r0 + r) * W + tile.c0;
        for (int f = 0; f < F; ++f) {
            const T* src = data + f * plane + src_row;
            for (int c = 0; c < w; ++c) row[static_cast<std::size_t>(c) * F + f] = src[c];
        }
        for (int c = 0; c < w; ++c) {
            const T* v = row + static_cast<std::size_t>(c) * F;
            for (int f = 0; f < F; ++f) {
                if (std::isnan(v[f])) { labels[static_cast<std::size_t>(r) * w + c] = MASKED; break; }
            }
        }
    }
    std::size_t n_valid = 0;
    for (std::size_t i = 0; i < n; ++i) n_valid += labels[i] == UNLABELED;

    // Seeds on masked pixels never enter the heap (their segment stays empty).
    const std::size_t K = tile.seed_ids.size();
    Heap heap(n);
    std::size_t k_valid = 0;
    for (std::size_t k = 0; k < K; ++k) {
        const std::int32_t s = tile.seed_ids[k];
        const int r = seeds[2 * s] - tile.r0;
        const int c = seeds[2 * s + 1] - tile.c0;
        const std::uint32_t i = static_cast<std::uint32_t>(r * w + c);
        if (labels[i] == MASKED) continue;
        heap.push(i, static_cast<std::uint32_t>(k), 0.0);
        ++k_valid;
    }
    if (n_valid == 0 || k_valid == 0) {
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                labels_out[static_cast<std::size_t>(tile.r0 + r) * W + tile.c0 + c] = -1;
        return;
    }

    // Local accumulators: running sums of features and coordinates.
    std::vector<double> ksum(K * F, 0.0);
    std::vector<double> kx(K, 0.0), ky(K, 0.0), ksize(K, 0.0);

    // Spatial weight M^2 * K / N = (M / S)^2, S = sqrt(N / K) the grid step.
    const double invwt = (compactness * compactness * static_cast<double>(k_valid)) /
                         static_cast<double>(n_valid);

    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    std::size_t pixelcount = 0;
    while (pixelcount < n_valid && !heap.empty()) {
        const Node node = heap.pop();
        const std::uint32_t k = node.k;
        const std::uint32_t i = node.i;
        if (labels[i] != UNLABELED) continue;

        const int x = static_cast<int>(i % w);
        const int y = static_cast<int>(i / w);
        labels[i] = static_cast<std::int32_t>(k);
        ++pixelcount;

        Eigen::Map<VecD> kc(ksum.data() + static_cast<std::size_t>(k) * F, F);
        kc += Eigen::Map<const VecT>(px.data() + static_cast<std::size_t>(i) * F, F).template cast<double>();
        kx[k] += x;
        ky[k] += y;
        ksize[k] += 1.0;
        const double nk = ksize[k];

        for (int p = 0; p < 4; ++p) {
            const int xx = x + dx4[p];
            const int yy = y + dy4[p];
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
            const std::uint32_t ii = static_cast<std::uint32_t>(yy * w + xx);
            if (labels[ii] != UNLABELED) continue;

            const double colordist =
                (kc - nk * Eigen::Map<const VecT>(px.data() + static_cast<std::size_t>(ii) * F, F)
                               .template cast<double>()).squaredNorm();
            const double xdiff = kx[k] - xx * nk;
            const double ydiff = ky[k] - yy * nk;
            const double xydist = xdiff * xdiff + ydiff * ydiff;
            // normalised by n^2 once, instead of dividing every sum by n
            heap.push(ii, k, (colordist + xydist * invwt) / (nk * nk));
        }
    }

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            const std::int32_t l = labels[static_cast<std::size_t>(r) * w + c];
            labels_out[static_cast<std::size_t>(tile.r0 + r) * W + tile.c0 + c] =
                l >= 0 ? tile.seed_ids[l] : -1;
        }
    }
    for (std::size_t k = 0; k < K; ++k) {
        const std::int32_t s = tile.seed_ids[k];
        std::copy(ksum.begin() + k * F, ksum.begin() + (k + 1) * F, acc.sums + static_cast<std::size_t>(s) * F);
        acc.sum_row[s] = ky[k] + tile.r0 * ksize[k];
        acc.sum_col[s] = kx[k] + tile.c0 * ksize[k];
        acc.size[s] = static_cast<std::int64_t>(ksize[k]);
    }
}

template <typename T>
py::tuple segment_impl(py::array_t<T, py::array::c_style | py::array::forcecast> data,
                       py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> seeds,
                       double compactness, int tile_height, int tile_width, int n_jobs)
{
    if (data.ndim() != 3) throw std::invalid_argument("data must be 3D [features, rows, cols]");
    if (seeds.ndim() != 2 || seeds.shape(1) != 2) throw std::invalid_argument("seeds must be [n_seeds, 2] (row, col)");
    if (!(compactness >= 0.0) || !std::isfinite(compactness)) throw std::invalid_argument("compactness must be finite and >= 0");

    const int F = static_cast<int>(data.shape(0));
    const int H = static_cast<int>(data.shape(1));
    const int W = static_cast<int>(data.shape(2));
    const std::int64_t K = seeds.shape(0);
    if (F < 1 || H < 1 || W < 1) throw std::invalid_argument("data must have at least one feature, row and column");
    if (K < 1) throw std::invalid_argument("at least one seed is required");
    if (K > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("too many seeds");

    const int th = tile_height > 0 ? std::min(tile_height, H) : H;
    const int tw = tile_width > 0 ? std::min(tile_width, W) : W;
    if (static_cast<std::uint64_t>(th) * tw >= std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("tile too large (>= 2^32 pixels); set tile_height/tile_width");
    if (n_jobs <= 0) n_jobs = std::max(1, omp_get_max_threads() - 1);

    const std::int32_t* sp = seeds.data();
    const int ntr = (H + th - 1) / th;
    const int ntc = (W + tw - 1) / tw;
    std::vector<Tile> tiles(static_cast<std::size_t>(ntr) * ntc);
    for (int a = 0; a < ntr; ++a)
        for (int b = 0; b < ntc; ++b) {
            Tile& t = tiles[static_cast<std::size_t>(a) * ntc + b];
            t.r0 = a * th;
            t.c0 = b * tw;
            t.h = std::min(th, H - t.r0);
            t.w = std::min(tw, W - t.c0);
        }
    for (std::int32_t s = 0; s < K; ++s) {
        const int r = sp[2 * s], c = sp[2 * s + 1];
        if (r < 0 || r >= H || c < 0 || c >= W) throw std::invalid_argument("seed outside the image");
        tiles[static_cast<std::size_t>(r / th) * ntc + c / tw].seed_ids.push_back(s);
    }

    py::array_t<std::int32_t> labels({H, W});
    py::array_t<double> means({static_cast<py::ssize_t>(K), static_cast<py::ssize_t>(F)});
    py::array_t<double> centroids({static_cast<py::ssize_t>(K), static_cast<py::ssize_t>(2)});
    py::array_t<std::int64_t> sizes(K);

    const T* dp = data.data();
    std::int32_t* lp = labels.mutable_data();
    double* mp = means.mutable_data();
    double* cp = centroids.mutable_data();
    std::int64_t* szp = sizes.mutable_data();
    std::vector<double> sum_row(K, 0.0), sum_col(K, 0.0);
    std::fill(mp, mp + K * F, 0.0);
    std::fill(szp, szp + K, 0);
    Accumulators acc{mp, sum_row.data(), sum_col.data(), szp};

    {
        py::gil_scoped_release release;
        const int n_tiles = static_cast<int>(tiles.size());
        if (n_tiles == 1) {
            segment_tile<T>(dp, H, W, F, tiles[0], sp, compactness, true, n_jobs, lp, acc);
        } else {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(n_jobs)
            for (int t = 0; t < n_tiles; ++t)
                segment_tile<T>(dp, H, W, F, tiles[t], sp, compactness, false, 1, lp, acc);
        }

        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (std::int64_t s = 0; s < K; ++s) {
            const double sz = static_cast<double>(szp[s]);
            if (sz > 0) {
                for (int f = 0; f < F; ++f) mp[s * F + f] /= sz;
                cp[2 * s] = sum_row[s] / sz;
                cp[2 * s + 1] = sum_col[s] / sz;
            } else {
                std::fill(mp + s * F, mp + (s + 1) * F, nan);
                cp[2 * s] = cp[2 * s + 1] = nan;
            }
        }
    }
    return py::make_tuple(labels, means, centroids, sizes);
}

} // namespace

py::tuple snic_segment(
    py::array data,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> seeds,
    double compactness, int tile_height, int tile_width, int n_jobs)
{
    // float32 cubes are segmented without a float64 copy; the accumulators
    // and distances are float64 either way.
    if (data.dtype().is(py::dtype::of<float>()))
        return segment_impl<float>(data, seeds, compactness, tile_height, tile_width, n_jobs);
    return segment_impl<double>(data, seeds, compactness, tile_height, tile_width, n_jobs);
}

} // namespace snic
} // namespace cdts
