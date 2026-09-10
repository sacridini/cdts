#include "som.h"
#include <Eigen/Dense>
#ifdef _OPENMP
#ifdef _OPENMP
#include <omp.h>
#endif
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x) (void)(x)
#endif
#include <cmath>
#include <vector>
#include <random>

namespace cdts {
namespace som {

pybind11::array_t<double> train_som_batch(
    pybind11::array_t<double> data_array,
    int x, int y, 
    int num_iters, 
    double initial_sigma,
    int n_jobs,
    int random_seed) 
{
    auto buf = data_array.request();
    int N = buf.shape[0];
    int D = buf.shape[1];
    const double* ptr = static_cast<double*>(buf.ptr);

    int K = x * y;
    
    std::vector<double> weights(K * D);
    std::mt19937 gen(random_seed);
    std::uniform_int_distribution<> dis(0, N - 1);
    for (int k = 0; k < K; ++k) {
        int idx = dis(gen);
        for(int d = 0; d < D; ++d) {
            weights[k * D + d] = ptr[idx * D + d];
        }
    }

    std::vector<std::pair<int, int>> pos(K);
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            pos[i * y + j] = {i, j};
        }
    }

    int threads = n_jobs > 0 ? n_jobs : std::max(1, omp_get_max_threads() - 1);

    for (int t = 0; t < num_iters; ++t) {
        double sigma = initial_sigma * std::exp(-static_cast<double>(t) / num_iters);
        double sig2 = 2.0 * sigma * sigma + 1e-8; // avoid div by zero

        std::vector<double> S(K * D, 0.0);
        std::vector<double> Counts(K, 0.0);

        #pragma omp parallel num_threads(threads)
        {
            std::vector<double> S_loc(K * D, 0.0);
            std::vector<double> Counts_loc(K, 0.0);

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                Eigen::Map<const Eigen::VectorXd> v(ptr + i * D, D);
                
                int bmu = 0;
                double min_dist = std::numeric_limits<double>::max();
                
                for (int k = 0; k < K; ++k) {
                    Eigen::Map<const Eigen::VectorXd> w(weights.data() + k * D, D);
                    double dist = (v - w).squaredNorm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        bmu = k;
                    }
                }

                Counts_loc[bmu] += 1.0;
                for (int d = 0; d < D; ++d) {
                    S_loc[bmu * D + d] += v[d];
                }
            }

            #pragma omp critical
            {
                for (int k = 0; k < K; ++k) {
                    Counts[k] += Counts_loc[k];
                    for (int d = 0; d < D; ++d) {
                        S[k * D + d] += S_loc[k * D + d];
                    }
                }
            }
        }

        std::vector<double> new_weights(K * D, 0.0);
        
        #pragma omp parallel for num_threads(threads)
        for (int k = 0; k < K; ++k) {
            std::vector<double> num(D, 0.0);
            double den = 0.0;
            
            for (int c = 0; c < K; ++c) {
                if (Counts[c] == 0) continue;
                
                double d_sq = std::pow(pos[c].first - pos[k].first, 2) + 
                              std::pow(pos[c].second - pos[k].second, 2);
                double h = std::exp(-d_sq / sig2);
                
                den += h * Counts[c];
                for (int d = 0; d < D; ++d) {
                    num[d] += h * S[c * D + d];
                }
            }
            
            if (den > 0) {
                for (int d = 0; d < D; ++d) {
                    new_weights[k * D + d] = num[d] / den;
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    new_weights[k * D + d] = weights[k * D + d];
                }
            }
        }
        weights = new_weights;
    }

    auto result = pybind11::array_t<double>({x, y, D});
    auto res_buf = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);
    for(int i = 0; i < K * D; ++i) {
        res_ptr[i] = weights[i];
    }
    return result;
}

pybind11::array_t<int> predict_bmus(
    pybind11::array_t<double> data_array,
    pybind11::array_t<double> weights_array,
    int n_jobs)
{
    auto buf_d = data_array.request();
    auto buf_w = weights_array.request();

    int N = buf_d.shape[0];
    int D = buf_d.shape[1];
    
    int x = buf_w.shape[0];
    int y = buf_w.shape[1];
    int K = x * y;

    const double* ptr_d = static_cast<double*>(buf_d.ptr);
    const double* ptr_w = static_cast<double*>(buf_w.ptr);

    auto result = pybind11::array_t<int>(N);
    auto buf_res = result.request();
    int* res_ptr = static_cast<int*>(buf_res.ptr);

    int threads = n_jobs > 0 ? n_jobs : std::max(1, omp_get_max_threads() - 1);

    #pragma omp parallel for num_threads(threads)
    for(int i = 0; i < N; ++i) {
        Eigen::Map<const Eigen::VectorXd> v(ptr_d + i * D, D);
        int bmu = 0;
        double min_dist = std::numeric_limits<double>::max();
        for(int k = 0; k < K; ++k) {
            Eigen::Map<const Eigen::VectorXd> w(ptr_w + k * D, D);
            double dist = (v - w).squaredNorm();
            if(dist < min_dist) {
                min_dist = dist;
                bmu = k;
            }
        }
        res_ptr[i] = bmu;
    }
    return result;
}

} // namespace som
} // namespace cdts
