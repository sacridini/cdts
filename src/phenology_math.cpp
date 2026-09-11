#include "phenology_math.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>
#include <algorithm>

std::vector<double> eigen_whittaker(
    const std::vector<double>& y, 
    const std::optional<std::vector<double>>& weights, 
    double lambda) 
{
    int n = y.size();
    if (n == 0) return {};

    Eigen::VectorXd y_vec = Eigen::Map<const Eigen::VectorXd>(y.data(), n);
    Eigen::VectorXd w_vec;
    if (weights.has_value()) {
        if (weights->size() != n) throw std::invalid_argument("Weights must have the same size as y.");
        w_vec = Eigen::Map<const Eigen::VectorXd>(weights->data(), n);
    } else {
        w_vec = Eigen::VectorXd::Ones(n);
    }

    int iters = 2;
    int deltaT = 3; 
    double wfact = 0.5;
    
    Eigen::VectorXd yiter = y_vec;
    Eigen::VectorXd z(n);
    
    double ylu_min = y_vec.minCoeff();
    double ylu_max = y_vec.maxCoeff();
    double zc = ylu_min + (ylu_max - ylu_min) * 0.5;

    for (int iter = 0; iter < iters; ++iter) {
        Eigen::SparseMatrix<double> A(n, n);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(n + (n - 2) * 9);
        for (int i = 0; i < n; ++i) triplets.push_back({i, i, w_vec(i)});
        for (int i = 0; i < n - 2; ++i) {
            triplets.push_back({i, i, lambda * 1.0});
            triplets.push_back({i, i+1, lambda * -2.0});
            triplets.push_back({i, i+2, lambda * 1.0});
            triplets.push_back({i+1, i, lambda * -2.0});
            triplets.push_back({i+1, i+1, lambda * 4.0});
            triplets.push_back({i+1, i+2, lambda * -2.0});
            triplets.push_back({i+2, i, lambda * 1.0});
            triplets.push_back({i+2, i+1, lambda * -2.0});
            triplets.push_back({i+2, i+2, lambda * 1.0});
        }
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();

        Eigen::VectorXd b = w_vec.cwiseProduct(yiter);
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(A);
        if (solver.info() != Eigen::Success) throw std::runtime_error("Whittaker decomposition failed");
        z = solver.solve(b);
        if (solver.info() != Eigen::Success) throw std::runtime_error("Whittaker solve failed");

        int m = (w_vec.array() > 0.5).count();
        if (m < 2) m = 2; 
        
        Eigen::VectorXd w_ceil = w_vec.array().ceil();
        double yfitmean = (z.array() * w_ceil.array()).sum() / m;
        double variance = ((z.array() - yfitmean) * w_ceil.array()).square().sum() / (m - 1);
        double yfitstd = std::sqrt(std::max(0.0, variance));
        
        for (int i = 0; i < n; ++i) {
            int m1 = std::max(0, i - deltaT);
            int m2 = std::min(n - 1, i + deltaT);
            double yi_min = 1e9;
            double yi_max = -1e9;
            for (int j = m1; j <= m2; ++j) {
                if (z(j) < yi_min) yi_min = z(j);
                if (z(j) > yi_max) yi_max = z(j);
            }
            if (y_vec(i) < z(i) - 1e-8) {
                if (yi_min > yfitmean || iter < 1) {
                    double ydiff = 0;
                    if (yi_max - yi_min < 0.8 * yfitstd && yfitstd > 1e-8) {
                        ydiff = 2.0 * (z(i) - y_vec(i)) / yfitstd;
                    }
                    w_vec(i) = wfact * w_vec(i) * std::exp(-ydiff * ydiff);
                }
            }
        }
        
        z = z.cwiseMax(ylu_min).cwiseMin(ylu_max);
        for (int i = 0; i < n; ++i) {
            if (z(i) > yiter(i) && z(i) > zc) {
                yiter(i) = z(i);
            }
        }
    }
    
    std::vector<double> result(z.data(), z.data() + n);
    return result;
}

std::vector<double> eigen_savgol(
    const std::vector<double>& y, 
    int window_size, 
    int poly_order)
{
    int n = y.size();
    if (n == 0) return {};
    if (window_size % 2 == 0) {
        throw std::invalid_argument("Window size must be odd.");
    }
    if (window_size > n) {
        throw std::invalid_argument("Window size cannot be larger than the data size.");
    }
    if (poly_order >= window_size) {
        throw std::invalid_argument("Polynomial order must be less than window size.");
    }

    int half_window = window_size / 2;
    
    // Construct the Vandermonde matrix for the local window centered at 0
    Eigen::MatrixXd J(window_size, poly_order + 1);
    for (int i = 0; i < window_size; ++i) {
        double x = i - half_window;
        for (int j = 0; j <= poly_order; ++j) {
            J(i, j) = std::pow(x, j);
        }
    }
    
    // Calculate the hat matrix H = J * (J^T J)^{-1} * J^T
    // This matrix projects the raw data onto the best-fit polynomial space
    Eigen::MatrixXd H = J * (J.transpose() * J).inverse() * J.transpose();
    
    // The filter coefficients for the center point are the center row of H
    Eigen::VectorXd center_coeffs = H.row(half_window);
    
    std::vector<double> result(n);
    
    for (int i = 0; i < n; ++i) {
        if (i < half_window) {
            // Left edge
            Eigen::VectorXd edge_coeffs = H.row(i);
            double sum = 0.0;
            for (int j = 0; j < window_size; ++j) {
                sum += edge_coeffs(j) * y[j];
            }
            result[i] = sum;
        } else if (i >= n - half_window) {
            // Right edge
            int offset = i - (n - window_size);
            Eigen::VectorXd edge_coeffs = H.row(offset);
            double sum = 0.0;
            for (int j = 0; j < window_size; ++j) {
                sum += edge_coeffs(j) * y[n - window_size + j];
            }
            result[i] = sum;
        } else {
            // Center
            double sum = 0.0;
            for (int j = 0; j < window_size; ++j) {
                sum += center_coeffs(j) * y[i - half_window + j];
            }
            result[i] = sum;
        }
    }

    return result;
}

struct PeakTrough {
    int pos;
    int type; // 1 max, -1 min
    double val;
};

std::vector<SeasonSegment> split_growing_seasons(
    const std::vector<double>& smoothed_y,
    int min_season_length,
    double min_amplitude,
    double rtrough_max,
    double r_min_filter) 
{
    int n = smoothed_y.size();
    if (n < 3) return {};

    double min_y = smoothed_y[0], max_y = smoothed_y[0];
    for (double val : smoothed_y) {
        if (val < min_y) min_y = val;
        if (val > max_y) max_y = val;
    }
    double A = max_y - min_y;
    if (A == 0) return {};

    // 1. findpeaks 
    auto find_extremes = [&](int type) {
        std::vector<PeakTrough> extr;
        int dir = 0; 
        int last_change = 0;
        
        for (int i = 1; i < n; ++i) {
            double diff = type == 1 ? (smoothed_y[i] - smoothed_y[i-1]) : (smoothed_y[i-1] - smoothed_y[i]);
            int cur_dir = (diff > 1e-9) ? 1 : (diff < -1e-9 ? -1 : 0);
            
            if (cur_dir == 1) {
                dir = 1;
                last_change = i - 1;
            } else if (cur_dir == -1) {
                if (dir == 1) {
                    int peak_pos = (last_change + 1 + i - 1) / 2;
                    extr.push_back({peak_pos, type, smoothed_y[peak_pos]});
                }
                dir = -1;
            }
        }
        return extr;
    };

    std::vector<PeakTrough> troughs = find_extremes(-1);
    std::vector<PeakTrough> peaks = find_extremes(1);
    
    // 1.1 Filter troughs: val - min_y <= rtrough_max * A
    std::vector<PeakTrough> filtered_troughs;
    for (auto& t : troughs) {
        if (t.val - min_y <= rtrough_max * A) {
            filtered_troughs.push_back(t);
        }
    }
    
    std::vector<PeakTrough> pos;
    pos.insert(pos.end(), filtered_troughs.begin(), filtered_troughs.end());
    pos.insert(pos.end(), peaks.begin(), peaks.end());
    std::sort(pos.begin(), pos.end(), [](const PeakTrough& a, const PeakTrough& b) {
        return a.pos < b.pos;
    });

    if (pos.empty()) return {};

    // 2. removeClosedExtreme
    double y_min = 0.05 * A;
    std::vector<PeakTrough> rm_closed;
    for (size_t i = 0; i < pos.size(); ++i) {
        if (i < pos.size() - 1 && std::abs(pos[i+1].val - pos[i].val) < y_min) {
            if (pos[i].type == 1) {
                rm_closed.push_back(pos[i].val > pos[i+1].val ? pos[i] : pos[i+1]);
            } else {
                rm_closed.push_back(pos[i].val < pos[i+1].val ? pos[i] : pos[i+1]);
            }
            ++i; // skip next
        } else {
            rm_closed.push_back(pos[i]);
        }
    }
    pos = rm_closed;

    // 3. check_GS_HeadTail (inject artificial trough at edges if needed)
    int nptperyear = 23;
    int minlen = nptperyear / 3;
    if (!pos.empty() && pos.back().type == 1 && (n - pos[pos.size()-2].pos) > minlen &&
        std::abs(smoothed_y.back() - pos[pos.size()-2].val) < 0.15 * A) {
        pos.push_back({n - 1, -1, smoothed_y.back()});
    }
    if (!pos.empty() && pos.front().type == 1 && pos[1].pos > minlen && 
        std::abs(smoothed_y.front() - pos[1].val) < 0.15 * A) {
        pos.insert(pos.begin(), {0, -1, smoothed_y.front()});
    }

    // Keep only from first trough to last trough
    int first_trough = -1, last_trough = -1;
    for (size_t i = 0; i < pos.size(); ++i) {
        if (pos[i].type == -1) {
            if (first_trough == -1) first_trough = i;
            last_trough = i;
        }
    }
    if (first_trough == -1 || last_trough == -1 || first_trough == last_trough) return {};
    pos = std::vector<PeakTrough>(pos.begin() + first_trough, pos.begin() + last_trough + 1);

    // Extract raw seasons
    std::vector<SeasonSegment> seasons;
    for (size_t i = 0; i + 2 < pos.size(); ++i) {
        if (pos[i].type == -1 && pos[i+1].type == 1 && pos[i+2].type == -1) {
            seasons.push_back({pos[i].pos, pos[i+1].pos, pos[i+2].pos});
        }
    }

    // 4. rcpp_season_filter (merge adjacent closed seasons)
    int DAYS_maxDIFF = 10; // 150 days / 16 = 9.3 steps
    int DAYS_max2GS = 40;  // 650 days / 16 = 40.6 steps
    
    // Simple pass to merge seasons if they are too close
    for (size_t i = 0; i + 1 < seasons.size(); ++i) {
        int t_diff = seasons[i+1].start_idx - seasons[i].end_idx;
        int T2s = seasons[i+1].end_idx - seasons[i].start_idx;
        
        if (t_diff < 0) {
            seasons[i].end_idx = seasons[i+1].start_idx;
        }
        
        bool is_closed = (t_diff >= 0 && t_diff <= DAYS_maxDIFF);
        if (is_closed && T2s <= DAYS_max2GS) {
            double T1_minVal = std::min(smoothed_y[seasons[i].start_idx], smoothed_y[seasons[i].end_idx]);
            double T2_minVal = std::min(smoothed_y[seasons[i+1].start_idx], smoothed_y[seasons[i+1].end_idx]);
            double max_Y = std::max(smoothed_y[seasons[i].peak_idx], smoothed_y[seasons[i+1].peak_idx]);
            double min_Y = std::min(T1_minVal, T2_minVal);
            double local_A = max_Y - min_Y;
            
            double T2_h_left = smoothed_y[seasons[i+1].peak_idx] - smoothed_y[seasons[i+1].start_idx];
            double T1_h_right = smoothed_y[seasons[i].peak_idx] - smoothed_y[seasons[i].end_idx];
            double trs = local_A * r_min_filter;
            double trs2 = local_A * (r_min_filter + 0.1);
            
            bool con_left = (T1_h_right <= trs && smoothed_y[seasons[i].start_idx] < smoothed_y[seasons[i+1].start_idx] && 
                             (smoothed_y[seasons[i].peak_idx] > trs2 + T1_minVal));
            bool con_right = (T2_h_left <= trs && smoothed_y[seasons[i].end_idx] > smoothed_y[seasons[i+1].end_idx] && 
                              (smoothed_y[seasons[i+1].peak_idx] > trs2 + T2_minVal));
                              
            if ((smoothed_y[seasons[i].end_idx] >= local_A * 0.7 + T1_minVal) || con_right || con_left) {
                // Merge right
                seasons[i].end_idx = seasons[i+1].end_idx;
                if (smoothed_y[seasons[i].peak_idx] < smoothed_y[seasons[i+1].peak_idx]) {
                    seasons[i].peak_idx = seasons[i+1].peak_idx;
                }
                seasons.erase(seasons.begin() + i + 1);
                --i; // recheck
            }
        }
    }
    
    std::vector<SeasonSegment> valid_seasons;
    for (auto& s : seasons) {
        if (s.end_idx - s.start_idx >= min_season_length) {
            double amplitude = smoothed_y[s.peak_idx] - std::max(smoothed_y[s.start_idx], smoothed_y[s.end_idx]);
            if (amplitude >= min_amplitude) {
                valid_seasons.push_back(s);
            }
        }
    }
    
    return valid_seasons;
}

std::vector<double> eigen_hants(
    const std::vector<double>& y, 
    const std::vector<double>& t, 
    int num_frequencies, 
    double threshold)
{
    int n = y.size();
    if (n == 0) return {};

    Eigen::VectorXd y_vec = Eigen::Map<const Eigen::VectorXd>(y.data(), n);
    Eigen::VectorXd w_vec = Eigen::VectorXd::Ones(n);

    // Build design matrix X for harmonic regression
    // 1 base frequency (mean), plus num_frequencies pairs of sine/cosine
    int num_features = 1 + 2 * num_frequencies;
    Eigen::MatrixXd X(n, num_features);
    
    double pi = 3.14159265358979323846;
    for (int i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        for (int j = 1; j <= num_frequencies; ++j) {
            double angle = 2.0 * pi * j * t[i] / 365.25;
            X(i, 2 * j - 1) = std::sin(angle);
            X(i, 2 * j) = std::cos(angle);
        }
    }

    Eigen::VectorXd y_fit;
    int max_iterations = 10;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Construct weighted X and weighted y
        Eigen::MatrixXd W = w_vec.asDiagonal();
        Eigen::MatrixXd X_w = W * X;
        Eigen::VectorXd y_w = W * y_vec;

        // Solve (X_w^T X_w) * beta = X_w^T y_w
        Eigen::MatrixXd XtX = X_w.transpose() * X_w;
        Eigen::VectorXd Xty = X_w.transpose() * y_w;
        
        // Solve for beta
        Eigen::VectorXd beta = XtX.ldlt().solve(Xty);
        
        // Predict
        y_fit = X * beta;
        
        // Update weights based on residuals and threshold
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            double residual = y_vec(i) - y_fit(i);
            // In typical HANTS, negatively biased outliers are rejected (e.g. clouds)
            if (residual < -threshold && w_vec(i) > 0.0) {
                w_vec(i) = 0.0;
                changed = true;
            }
        }
        
        if (!changed) {
            break;
        }
    }
    
    std::vector<double> result(y_fit.data(), y_fit.data() + n);
    return result;
}

Eigen::VectorXd eigen_wTSM(
    const Eigen::VectorXd& y, 
    const Eigen::VectorXd& yfit, 
    const Eigen::VectorXd& w, 
    int iter, 
    int nptperyear, 
    double wfact)
{
    int n = y.size();
    Eigen::VectorXd wnew = w;
    
    double m_count = 0;
    for(int i = 0; i < n; i++) if (w[i] > 0.5) m_count += 1.0;
    if (m_count < 2.0) return wnew;

    double yfitmean = 0.0;
    for(int i = 0; i < n; i++) {
        double w_ceil = std::ceil(w[i]);
        yfitmean += yfit[i] * w_ceil;
    }
    yfitmean /= m_count;

    double variance = 0.0;
    for(int i = 0; i < n; i++) {
        double w_ceil = std::ceil(w[i]);
        variance += std::pow(yfit[i] - yfitmean, 2) * w_ceil;
    }
    double yfitstd = std::sqrt(variance / (m_count - 1.0));

    int deltaT = std::floor(nptperyear / 7.0);

    for (int i = 0; i < n; i++) {
        int m1 = std::max(0, i - deltaT);
        int m2 = std::min(n - 1, i + deltaT);
        
        double yi_min = yfit[m1];
        double yi_max = yfit[m1];
        for (int j = m1; j <= m2; j++) {
            if (yfit[j] < yi_min) yi_min = yfit[j];
            if (yfit[j] > yi_max) yi_max = yfit[j];
        }

        if (y[i] < yfit[i] - 1e-8) {
            if (yi_min > yfitmean || iter < 2) {
                double ydiff = 0.0;
                if (yi_max - yi_min < 0.8 * yfitstd) {
                    ydiff = 2.0 * (yfit[i] - y[i]) / yfitstd;
                }
                wnew[i] = wfact * w[i] * std::exp(-ydiff * ydiff);
            }
        }
    }
    return wnew;
}

