#include "phenology.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <iostream>

namespace phenology {

PhenologyMetrics extract_metrics(const Eigen::VectorXd& params, CurveType type, const Eigen::VectorXd& t_segment, ExtractionMethod method) {
    PhenologyMetrics metrics = {std::nan(""), std::nan(""), std::nan(""), std::nan("")};
    
    if (t_segment.size() == 0) return metrics;

    int n_pts = 365;
    double t_min = t_segment.minCoeff();
    double t_max = t_segment.maxCoeff();
    if (t_max - t_min < 1.0) t_max = t_min + 365.0;
    
    Eigen::VectorXd t_eval = Eigen::VectorXd::LinSpaced(n_pts, t_min, t_max);
    Eigen::VectorXd y_dummy = Eigen::VectorXd::Zero(n_pts);
    Eigen::VectorXd fvec(n_pts);
    
    // Create empty bound vectors just for functor evaluation (optimization bounds aren't used here)
    Eigen::VectorXd dummy_bounds(params.size());
    dummy_bounds.setZero();
    
    switch(type) {
        case CurveType::BECK: { BeckFunctor f(t_eval, y_dummy, dummy_bounds, dummy_bounds); f(params, fvec); break; }
        case CurveType::ELMORE: { ElmoreFunctor f(t_eval, y_dummy, dummy_bounds, dummy_bounds); f(params, fvec); break; }
        case CurveType::GU: { GuFunctor f(t_eval, y_dummy, dummy_bounds, dummy_bounds); f(params, fvec); break; }
        case CurveType::KLOS: { KlosFunctor f(t_eval, y_dummy, dummy_bounds, dummy_bounds); f(params, fvec); break; }
        case CurveType::ZHANG: { ZhangFunctor f(t_eval, y_dummy, dummy_bounds, dummy_bounds); f(params, fvec); break; }
        case CurveType::AG: { AGFunctor f(t_eval, y_dummy, dummy_bounds, dummy_bounds); f(params, fvec); break; }
        case CurveType::DL: { DLFunctor f(t_eval, y_dummy, dummy_bounds, dummy_bounds); f(params, fvec); break; }
    }
    
    double min_val = fvec.minCoeff();
    double max_val = fvec.maxCoeff();
    double amplitude = max_val - min_val;
    double threshold = min_val + 0.2 * amplitude;
    
    int pop_idx;
    fvec.maxCoeff(&pop_idx);
    metrics.pop = t_eval(pop_idx);
    
    Eigen::VectorXd d1(n_pts), d2(n_pts), dk(n_pts), k(n_pts);
    if (method == ExtractionMethod::DERIVATIVE || method == ExtractionMethod::GU || method == ExtractionMethod::KLOSTERMAN) {
        for(int i = 0; i < n_pts; ++i) {
            if(i == 0) d1[i] = (fvec[i+1] - fvec[i]) / (t_eval[i+1] - t_eval[i]);
            else if(i == n_pts - 1) d1[i] = (fvec[i] - fvec[i-1]) / (t_eval[i] - t_eval[i-1]);
            else d1[i] = (fvec[i+1] - fvec[i-1]) / (t_eval[i+1] - t_eval[i-1]);
        }
    }
    
    if (method == ExtractionMethod::KLOSTERMAN) {
        for(int i = 0; i < n_pts; ++i) {
            if(i == 0) d2[i] = (d1[i+1] - d1[i]) / (t_eval[i+1] - t_eval[i]);
            else if(i == n_pts - 1) d2[i] = (d1[i] - d1[i-1]) / (t_eval[i] - t_eval[i-1]);
            else d2[i] = (d1[i+1] - d1[i-1]) / (t_eval[i+1] - t_eval[i-1]);
            k[i] = d2[i] / std::pow(1.0 + d1[i]*d1[i], 1.5);
        }
        for(int i = 0; i < n_pts; ++i) {
            if(i == 0) dk[i] = (k[i+1] - k[i]) / (t_eval[i+1] - t_eval[i]);
            else if(i == n_pts - 1) dk[i] = (k[i] - k[i-1]) / (t_eval[i] - t_eval[i-1]);
            else dk[i] = (k[i+1] - k[i-1]) / (t_eval[i+1] - t_eval[i-1]);
        }
    }

    if (method == ExtractionMethod::DERIVATIVE) {
        double max_d1 = -1e9; int sos_idx = 0;
        for(int i = 0; i <= pop_idx; ++i) { if(d1[i] > max_d1) { max_d1 = d1[i]; sos_idx = i; } }
        metrics.sos = t_eval(sos_idx);
        
        double min_d1 = 1e9; int eos_idx = n_pts - 1;
        for(int i = pop_idx; i < n_pts; ++i) { if(d1[i] < min_d1) { min_d1 = d1[i]; eos_idx = i; } }
        metrics.eos = t_eval(eos_idx);
    } else if (method == ExtractionMethod::GU) {
        double max_d1 = -1e9; int sos_idx = 0;
        for(int i = 0; i <= pop_idx; ++i) { if(d1[i] > max_d1) { max_d1 = d1[i]; sos_idx = i; } }
        
        double min_d1 = 1e9; int eos_idx = n_pts - 1;
        for(int i = pop_idx; i < n_pts; ++i) { if(d1[i] < min_d1) { min_d1 = d1[i]; eos_idx = i; } }
        
        double y_min_left = fvec.head(pop_idx + 1).minCoeff();
        double y_min_right = fvec.tail(n_pts - pop_idx).minCoeff();
        
        if(std::abs(max_d1) > 1e-6) metrics.sos = t_eval(sos_idx) + (y_min_left - fvec(sos_idx)) / max_d1;
        else metrics.sos = t_eval(sos_idx);
        
        if(std::abs(min_d1) > 1e-6) metrics.eos = t_eval(eos_idx) + (y_min_right - fvec(eos_idx)) / min_d1;
        else metrics.eos = t_eval(eos_idx);
    } else if (method == ExtractionMethod::KLOSTERMAN) {
        double max_dk = -1e9; int sos_idx = 0;
        for(int i = 0; i <= pop_idx; ++i) { if(dk[i] > max_dk) { max_dk = dk[i]; sos_idx = i; } }
        metrics.sos = t_eval(sos_idx);
        
        double max_dk_right = -1e9; int eos_idx = n_pts - 1;
        for(int i = pop_idx; i < n_pts; ++i) { if(dk[i] > max_dk_right) { max_dk_right = dk[i]; eos_idx = i; } }
        metrics.eos = t_eval(eos_idx);
    } else {
        // THRESHOLD
        for (int i = 0; i < pop_idx; ++i) {
            if (fvec(i) < threshold && fvec(i+1) >= threshold) {
                double w = (threshold - fvec(i)) / (fvec(i+1) - fvec(i));
                metrics.sos = t_eval(i) + w * (t_eval(i+1) - t_eval(i));
                break;
            }
        }
        if (std::isnan(metrics.sos)) metrics.sos = t_eval(0);
        
        for (int i = pop_idx; i < n_pts - 1; ++i) {
            if (fvec(i) >= threshold && fvec(i+1) < threshold) {
                double w = (threshold - fvec(i)) / (fvec(i+1) - fvec(i));
                metrics.eos = t_eval(i) + w * (t_eval(i+1) - t_eval(i));
                break;
            }
        }
        if (std::isnan(metrics.eos)) metrics.eos = t_eval(n_pts - 1);
    }
    
    metrics.los = metrics.eos - metrics.sos;
    return metrics;
}

pybind11::tuple fit_phenology_batch(
    pybind11::array_t<double> values_array,
    pybind11::array_t<double> dates_array,
    int curve_type_int, 
    int extraction_method,
    int max_seasons,
    double whittaker_lambda,
    bool apply_whittaker,
    bool apply_hants,
    int hants_frequencies,
    double hants_threshold,
    int min_season_length,
    double min_amplitude,
    double min_pixel_amplitude,
    int n_jobs)
{
    auto vals_buf = values_array.request();
    auto dates_buf = dates_array.request();
    
    if (vals_buf.ndim != 2) throw std::runtime_error("values_array must be 2D [pixels, time]");
    if (dates_buf.ndim != 1) throw std::runtime_error("dates_array must be 1D [time]");
    
    int n_pixels = vals_buf.shape[0];
    int n_time = vals_buf.shape[1];
    
    if (dates_buf.shape[0] != n_time) throw std::runtime_error("dates_array length must match values_array time dimension");
    
    double* vals_ptr = static_cast<double*>(vals_buf.ptr);
    double* dates_ptr = static_cast<double*>(dates_buf.ptr);
    
    CurveType curve_type = static_cast<CurveType>(curve_type_int);
    
    pybind11::array_t<double> sos_arr({n_pixels, max_seasons});
    pybind11::array_t<double> eos_arr({n_pixels, max_seasons});
    pybind11::array_t<double> los_arr({n_pixels, max_seasons});
    pybind11::array_t<double> pop_arr({n_pixels, max_seasons});
    
    double* sos_ptr = static_cast<double*>(sos_arr.request().ptr);
    double* eos_ptr = static_cast<double*>(eos_arr.request().ptr);
    double* los_ptr = static_cast<double*>(los_arr.request().ptr);
    double* pop_ptr = static_cast<double*>(pop_arr.request().ptr);
    
    for (int i = 0; i < n_pixels * max_seasons; ++i) {
        sos_ptr[i] = std::nan("");
        eos_ptr[i] = std::nan("");
        los_ptr[i] = std::nan("");
        pop_ptr[i] = std::nan("");
    }
    
    if (n_jobs <= 0) n_jobs = omp_get_max_threads();
    
    Eigen::Map<Eigen::VectorXd> t_all(dates_ptr, n_time);

    #pragma omp parallel for num_threads(n_jobs)
    for (int p = 0; p < n_pixels; ++p) {
        std::vector<double> y_raw(n_time);
        
        bool valid_pixel = false;
        double pixel_min = 99999.0;
        double pixel_max = -99999.0;
        
        for (int t = 0; t < n_time; ++t) {
            double v = vals_ptr[p * n_time + t];
            y_raw[t] = v;
            if (!std::isnan(v) && v != -9999.0) {
                valid_pixel = true;
                if (v < pixel_min) pixel_min = v;
                if (v > pixel_max) pixel_max = v;
            }
        }
        
        if (!valid_pixel) continue;
        
        // 1. Skip Logic: Early exit for low amplitude (water, urban, bare soil)
        if (pixel_max - pixel_min < min_pixel_amplitude) continue;
        
        std::vector<double> y_smooth = y_raw;
        if (apply_whittaker) {
            try {
                y_smooth = eigen_whittaker(y_raw, std::nullopt, whittaker_lambda);
            } catch (...) {
                continue;
            }
        } else if (apply_hants) {
            try {
                y_smooth = eigen_hants(y_raw, hants_frequencies, hants_threshold);
            } catch (...) {
                continue;
            }
        }
        
        auto seasons = split_growing_seasons(y_smooth, min_season_length, min_amplitude);
        
        int s_idx = 0;
        for (const auto& season : seasons) {
            if (s_idx >= max_seasons) break;
            
            int start = season.start_idx;
            int end = season.end_idx;
            int len = end - start + 1;
            
            if (len < 4) continue;
            
            Eigen::VectorXd t_seg(len);
            Eigen::VectorXd y_seg(len);
            for (int i = 0; i < len; ++i) {
                t_seg(i) = t_all(start + i);
                y_seg(i) = y_smooth[start + i];
            }
            
            Eigen::VectorXd params;
            switch(curve_type) {
                case CurveType::BECK: params = Eigen::VectorXd::Ones(6); break;
                case CurveType::ELMORE: params = Eigen::VectorXd::Ones(7); break;
                case CurveType::GU: params = Eigen::VectorXd::Ones(9); break;
                case CurveType::KLOS: params = Eigen::VectorXd::Ones(13); break;
                case CurveType::ZHANG: params = Eigen::VectorXd::Ones(7); break;
                case CurveType::AG: params = Eigen::VectorXd::Ones(7); break;
                case CurveType::DL: params = Eigen::VectorXd::Ones(6); break;
            }
            
            bool converged = fit_curve(t_seg, y_seg, params, curve_type);
            
            if (converged) {
                PhenologyMetrics metrics = extract_metrics(params, curve_type, t_seg, static_cast<ExtractionMethod>(extraction_method));
                int out_idx = p * max_seasons + s_idx;
                sos_ptr[out_idx] = metrics.sos;
                eos_ptr[out_idx] = metrics.eos;
                los_ptr[out_idx] = metrics.los;
                pop_ptr[out_idx] = metrics.pop;
                s_idx++;
            }
        }
    }
    
    return pybind11::make_tuple(sos_arr, eos_arr, los_arr, pop_arr);
}

} // namespace phenology
