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

std::vector<SeasonSegment> split_growing_seasons(
    const std::vector<double>& smoothed_y,
    int min_season_length,
    double min_amplitude) {
    int n = smoothed_y.size();
    if (n < 3) return {};

    std::vector<int> minima;
    
    // Find local minima
    for (int i = 1; i < n - 1; ++i) {
        if (smoothed_y[i] < smoothed_y[i-1] && smoothed_y[i] < smoothed_y[i+1]) {
            minima.push_back(i);
        }
    }
    
    // Include boundaries as potential minima if they are lower than their neighbor
    if (smoothed_y[0] < smoothed_y[1]) {
        minima.insert(minima.begin(), 0);
    }
    if (smoothed_y[n-1] < smoothed_y[n-2]) {
        minima.push_back(n-1);
    }
    
    std::vector<SeasonSegment> seasons;
    
    // A season is bounded by two consecutive minima, containing a peak
    for (size_t i = 0; i + 1 < minima.size(); ++i) {
        int start = minima[i];
        int end = minima[i+1];
        
        if (end - start < min_season_length) {
            continue; // Skip seasons that are too short
        }
        
        // Find the absolute maximum between start and end
        int peak = -1;
        double max_val = -1e9; // Start with a very small number
        for (int j = start + 1; j < end; ++j) {
            if (smoothed_y[j] > max_val) {
                max_val = smoothed_y[j];
                peak = j;
            }
        }
        
        // If a valid peak was found and it's higher than the boundaries
        if (peak != -1 && max_val > std::max(smoothed_y[start], smoothed_y[end])) {
            double amplitude = max_val - std::max(smoothed_y[start], smoothed_y[end]);
            if (amplitude >= min_amplitude) {
                seasons.push_back({start, peak, end});
            }
        }
    }
    
    return seasons;
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
