#include "phenology_curves.h"
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace phenology {

inline double transform_to_bounded(double p, double lb, double ub) {
    return lb + (ub - lb) * 0.5 * (std::sin(p) + 1.0);
}

inline double transform_to_unbounded(double bounded, double lb, double ub) {
    double val = 2.0 * (bounded - lb) / (ub - lb) - 1.0;
    val = std::max(-1.0, std::min(1.0, val));
    return std::asin(val);
}

// Shared prediction functions: used identically by the fitting residual
// (Functor::operator()) and by evaluate_curve(), so fitting and post-fit
// evaluation (phenology extraction, GOF) can never diverge.

inline double predict_beck(double t, double mn, double mx, double sos, double rsp, double eos, double rau) {
    return mn + (mx - mn) * (1.0 / (1.0 + std::exp(-rsp * (t - sos))) +
                              1.0 / (1.0 + std::exp(rau * (t - eos))) - 1.0);
}

inline double predict_elmore(double t, double mn, double mx, double sos, double rsp, double eos, double rau, double m7) {
    return mn + (mx - m7 * t) * (1.0 / (1.0 + std::exp(-rsp * (t - sos))) -
                                  1.0 / (1.0 + std::exp(-rau * (t - eos))));
}

inline double predict_gu(double t, double y0, double a1, double a2, double sos, double rsp, double eos_p, double rau, double c1, double c2) {
    double base1 = std::max(1e-8, 1.0 + std::exp(-rsp * (t - sos)));
    double base2 = std::max(1e-8, 1.0 + std::exp(-rau * (t - eos_p)));
    return y0 + (a1 / std::pow(base1, c1)) - (a2 / std::pow(base2, c2));
}

inline double predict_klos(double t, double a1, double a2, double b1, double b2, double c,
                            double B1, double B2, double m1, double m2, double q1, double q2, double v1, double v2) {
    double base1 = std::max(1e-8, 1.0 + q1 * std::exp(-B1 * (t - m1)));
    double base2 = std::max(1e-8, 1.0 + q2 * std::exp(-B2 * (t - m2)));
    return (a1 * t + b1) + (a2 * t * t + b2 * t + c) * (1.0 / std::pow(base1, v1) - 1.0 / std::pow(base2, v2));
}

inline double predict_zhang(double t, double t0, double mn, double mx, double sos, double rsp, double eos, double rau) {
    if (t <= t0) {
        return mn + (mx - mn) / (1.0 + std::exp(-rsp * (t - sos)));
    }
    return mn + (mx - mn) / (1.0 + std::exp(rau * (t - eos)));
}

inline double predict_ag(double t, double t0, double mn, double mx, double rsp, double a3, double rau, double a5) {
    if (t <= t0) {
        double base = std::max(0.0, (t0 - t) * rsp);
        return mn + (mx - mn) * std::exp(-std::pow(base, a3));
    }
    double base = std::max(0.0, (t - t0) * rau);
    return mn + (mx - mn) * std::exp(-std::pow(base, a5));
}

inline double predict_dl(double t, double mn, double mx, double sos, double rsp, double eos, double rau) {
    return mn + (mx - mn) * (1.0 / (1.0 + std::exp(-rsp * (t - sos))) -
                              1.0 / (1.0 + std::exp(-rau * (t - eos))));
}

int BeckFunctor::operator()(const Eigen::VectorXd &p_unb, Eigen::VectorXd &fvec) const {
    double mn = transform_to_bounded(p_unb[0], lb[0], ub[0]);
    double mx = transform_to_bounded(p_unb[1], lb[1], ub[1]);
    double sos = transform_to_bounded(p_unb[2], lb[2], ub[2]);
    double rsp = transform_to_bounded(p_unb[3], lb[3], ub[3]);
    double eos = transform_to_bounded(p_unb[4], lb[4], ub[4]);
    double rau = transform_to_bounded(p_unb[5], lb[5], ub[5]);
    for(int i = 0; i < t.size(); ++i) {
        double pred = predict_beck(t[i], mn, mx, sos, rsp, eos, rau);
        fvec[i] = (pred - y[i]) * std::sqrt(w[i]);
    }
    return 0;
}

int ElmoreFunctor::operator()(const Eigen::VectorXd &p_unb, Eigen::VectorXd &fvec) const {
    double mn = transform_to_bounded(p_unb[0], lb[0], ub[0]);
    double mx = transform_to_bounded(p_unb[1], lb[1], ub[1]);
    double sos = transform_to_bounded(p_unb[2], lb[2], ub[2]);
    double rsp = transform_to_bounded(p_unb[3], lb[3], ub[3]);
    double eos = transform_to_bounded(p_unb[4], lb[4], ub[4]);
    double rau = transform_to_bounded(p_unb[5], lb[5], ub[5]);
    double m7 = transform_to_bounded(p_unb[6], lb[6], ub[6]);
    for(int i = 0; i < t.size(); ++i) {
        double pred = predict_elmore(t[i], mn, mx, sos, rsp, eos, rau, m7);
        fvec[i] = (pred - y[i]) * std::sqrt(w[i]);
    }
    return 0;
}

int GuFunctor::operator()(const Eigen::VectorXd &p_unb, Eigen::VectorXd &fvec) const {
    double y0 = transform_to_bounded(p_unb[0], lb[0], ub[0]);
    double a1 = transform_to_bounded(p_unb[1], lb[1], ub[1]);
    double a2 = transform_to_bounded(p_unb[2], lb[2], ub[2]);
    double sos = transform_to_bounded(p_unb[3], lb[3], ub[3]);
    double rsp = transform_to_bounded(p_unb[4], lb[4], ub[4]);
    double eos_p = transform_to_bounded(p_unb[5], lb[5], ub[5]);
    double rau = transform_to_bounded(p_unb[6], lb[6], ub[6]);
    double c1 = transform_to_bounded(p_unb[7], lb[7], ub[7]);
    double c2 = transform_to_bounded(p_unb[8], lb[8], ub[8]);
    for(int i = 0; i < t.size(); ++i) {
        double pred = predict_gu(t[i], y0, a1, a2, sos, rsp, eos_p, rau, c1, c2);
        fvec[i] = (pred - y[i]) * std::sqrt(w[i]);
    }
    return 0;
}

int KlosFunctor::operator()(const Eigen::VectorXd &p_unb, Eigen::VectorXd &fvec) const {
    double a1 = transform_to_bounded(p_unb[0], lb[0], ub[0]);
    double a2 = transform_to_bounded(p_unb[1], lb[1], ub[1]);
    double b1 = transform_to_bounded(p_unb[2], lb[2], ub[2]);
    double b2 = transform_to_bounded(p_unb[3], lb[3], ub[3]);
    double c = transform_to_bounded(p_unb[4], lb[4], ub[4]);
    double B1 = transform_to_bounded(p_unb[5], lb[5], ub[5]);
    double B2 = transform_to_bounded(p_unb[6], lb[6], ub[6]);
    double m1 = transform_to_bounded(p_unb[7], lb[7], ub[7]);
    double m2 = transform_to_bounded(p_unb[8], lb[8], ub[8]);
    double q1 = transform_to_bounded(p_unb[9], lb[9], ub[9]);
    double q2 = transform_to_bounded(p_unb[10], lb[10], ub[10]);
    double v1 = transform_to_bounded(p_unb[11], lb[11], ub[11]);
    double v2 = transform_to_bounded(p_unb[12], lb[12], ub[12]);
    for(int i = 0; i < t.size(); ++i) {
        double pred = predict_klos(t[i], a1, a2, b1, b2, c, B1, B2, m1, m2, q1, q2, v1, v2);
        fvec[i] = (pred - y[i]) * std::sqrt(w[i]);
    }
    return 0;
}

int ZhangFunctor::operator()(const Eigen::VectorXd &p_unb, Eigen::VectorXd &fvec) const {
    double t0 = transform_to_bounded(p_unb[0], lb[0], ub[0]);
    double mn = transform_to_bounded(p_unb[1], lb[1], ub[1]);
    double mx = transform_to_bounded(p_unb[2], lb[2], ub[2]);
    double sos = transform_to_bounded(p_unb[3], lb[3], ub[3]);
    double rsp = transform_to_bounded(p_unb[4], lb[4], ub[4]);
    double eos = transform_to_bounded(p_unb[5], lb[5], ub[5]);
    double rau = transform_to_bounded(p_unb[6], lb[6], ub[6]);
    for(int i = 0; i < t.size(); ++i) {
        double pred = predict_zhang(t[i], t0, mn, mx, sos, rsp, eos, rau);
        fvec[i] = (pred - y[i]) * std::sqrt(w[i]);
    }
    return 0;
}

int AGFunctor::operator()(const Eigen::VectorXd &p_unb, Eigen::VectorXd &fvec) const {
    double t0 = transform_to_bounded(p_unb[0], lb[0], ub[0]);
    double mn = transform_to_bounded(p_unb[1], lb[1], ub[1]);
    double mx = transform_to_bounded(p_unb[2], lb[2], ub[2]);
    double rsp = transform_to_bounded(p_unb[3], lb[3], ub[3]);
    double a3 = transform_to_bounded(p_unb[4], lb[4], ub[4]);
    double rau = transform_to_bounded(p_unb[5], lb[5], ub[5]);
    double a5 = transform_to_bounded(p_unb[6], lb[6], ub[6]);
    for(int i = 0; i < t.size(); ++i) {
        double pred = predict_ag(t[i], t0, mn, mx, rsp, a3, rau, a5);
        fvec[i] = (pred - y[i]) * std::sqrt(w[i]);
    }
    return 0;
}

int DLFunctor::operator()(const Eigen::VectorXd &p_unb, Eigen::VectorXd &fvec) const {
    double mn = transform_to_bounded(p_unb[0], lb[0], ub[0]);
    double mx = transform_to_bounded(p_unb[1], lb[1], ub[1]);
    double sos = transform_to_bounded(p_unb[2], lb[2], ub[2]);
    double rsp = transform_to_bounded(p_unb[3], lb[3], ub[3]);
    double eos = transform_to_bounded(p_unb[4], lb[4], ub[4]);
    double rau = transform_to_bounded(p_unb[5], lb[5], ub[5]);
    for(int i = 0; i < t.size(); ++i) {
        double pred = predict_dl(t[i], mn, mx, sos, rsp, eos, rau);
        fvec[i] = (pred - y[i]) * std::sqrt(w[i]);
    }
    return 0;
}

// Helper template for optimization
template<typename FunctorType>
bool optimize_functor(FunctorType& functor, Eigen::VectorXd& params, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub, int max_fev) {
    Eigen::NumericalDiff<FunctorType> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<FunctorType>> lm(numDiff);
    lm.parameters.maxfev = max_fev;
    lm.parameters.ftol = 1e-6;
    lm.parameters.xtol = 1e-6;
    
    // Transform initial params to unbounded space
    Eigen::VectorXd unbounded_params = params;
    for (int i = 0; i < params.size(); ++i) {
        unbounded_params[i] = transform_to_unbounded(params[i], lb[i], ub[i]);
    }
    
    int info = lm.minimize(unbounded_params);
    
    // Transform back to bounded space
    for (int i = 0; i < params.size(); ++i) {
        params[i] = transform_to_bounded(unbounded_params[i], lb[i], ub[i]);
    }
    
    return (info == Eigen::LevenbergMarquardtSpace::RelativeErrorTooSmall || 
            info == Eigen::LevenbergMarquardtSpace::RelativeErrorAndReductionTooSmall ||
            info == Eigen::LevenbergMarquardtSpace::RelativeReductionTooSmall ||
            info == Eigen::LevenbergMarquardtSpace::CosinusTooSmall);
}

bool fit_curve(const Eigen::VectorXd& t, const Eigen::VectorXd& y, const Eigen::VectorXd& w, 
               Eigen::VectorXd& params, CurveType type, int max_fev) {
    if(t.size() != y.size() || t.size() == 0) return false;

    double t_min = t.minCoeff();
    double t_max = t.maxCoeff();
    double y_min = y.minCoeff();
    double y_max = y.maxCoeff();
    double y_amp = std::max(0.01, y_max - y_min);
    
    double half = (t_max - t_min) / 2.0;
    if (half <= 0) half = 100.0;
    double k = 4.0 / half * 2.67;
    double r_min = k / 1.5;
    double r_max = k * 6.0;
    double d_t = half / 4.0;
    
    Eigen::VectorXd lb, ub;

    switch(type) {
        case CurveType::BECK: {
            if(params.size() != 6) params = Eigen::VectorXd::Zero(6);
            lb = Eigen::VectorXd(6); ub = Eigen::VectorXd(6);
            lb << y_min - 0.5*y_amp, y_min, t_min, r_min, t_min, r_min;
            ub << y_min + 0.5*y_amp, y_max + 0.5*y_amp, t_max, r_max, t_max, r_max;
            if((params.array() == 1.0).all() || (params.array() == 0.0).all()) {
                params = (lb + ub) / 2.0;
                params(2) = t_min + (t_max - t_min) * 0.25; // SOS timing
                params(4) = t_min + (t_max - t_min) * 0.75; // EOS timing
                params(3) = k; // rate
                params(5) = k; // rate
            }
            BeckFunctor functor(t, y, w, lb, ub);
            return optimize_functor(functor, params, lb, ub, max_fev);
        }
        case CurveType::ELMORE: {
            if(params.size() != 7) params = Eigen::VectorXd::Zero(7);
            lb = Eigen::VectorXd(7); ub = Eigen::VectorXd(7);
            lb << y_min - 0.5*y_amp, y_min, t_min, r_min, t_min, r_min, -0.1;
            ub << y_min + 0.5*y_amp, y_max + 0.5*y_amp, t_max, r_max, t_max, r_max, 0.1;
            if((params.array() == 1.0).all() || (params.array() == 0.0).all()) {
                params = (lb + ub) / 2.0;
                params(2) = t_min + (t_max - t_min) * 0.25;
                params(4) = t_min + (t_max - t_min) * 0.75;
                params(3) = k;
                params(5) = k;
                params(6) = 0.0;
            }
            ElmoreFunctor functor(t, y, w, lb, ub);
            return optimize_functor(functor, params, lb, ub, max_fev);
        }
        case CurveType::GU: {
            if(params.size() != 9) params = Eigen::VectorXd::Zero(9);
            lb = Eigen::VectorXd(9); ub = Eigen::VectorXd(9);
            lb << y_min - 0.5*y_amp, 0.0, 0.0, t_min - 30, 0.0, t_min - 30, 0.0, 0.1, 0.1;
            ub << y_min + 0.5*y_amp, 5.0*y_amp, 5.0*y_amp, t_max + 30, 50.0, t_max + 30, 50.0, 10.0, 10.0;
            if((params.array() == 1.0).all() || (params.array() == 0.0).all()) {
                params = (lb + ub) / 2.0;
                params(0) = y_min;              // y0 baseline
                params(1) = y_amp;              // a1
                params(2) = y_amp;              // a2
                params(3) = t_min + (t_max - t_min) * 0.25; // sos timing
                params(4) = k;                  // rsp
                params(5) = t_min + (t_max - t_min) * 0.75; // eos timing
                params(6) = k;                  // rau
                params(7) = 1.0;                // c1
                params(8) = 1.0;                // c2
            }
            GuFunctor functor(t, y, w, lb, ub);
            return optimize_functor(functor, params, lb, ub, max_fev);
        }
        case CurveType::KLOS: {
            if(params.size() != 13) params = Eigen::VectorXd::Zero(13);
            lb = Eigen::VectorXd(13); ub = Eigen::VectorXd(13);
            lb << -1.0, -1.0, -1.0, -1.0, y_min - 1.0, 0.0, 0.0, t_min - 30, t_min - 30, -5.0, -5.0, 0.1, 0.1;
            ub <<  1.0,  1.0,  1.0,  1.0, y_max + 1.0, 50.0, 50.0, t_max + 30, t_max + 30,  5.0,  5.0, 10.0, 10.0;
            if((params.array() == 1.0).all() || (params.array() == 0.0).all()) {
                params = (lb + ub) / 2.0;
                // a1,a2,b1,b2 (0-3): flat trend; c (4): amplitude carried by the logistic terms
                params(0) = 0.0; params(1) = 0.0; params(2) = y_min; params(3) = 0.0;
                params(4) = y_amp;              // c
                params(5) = k;                  // B1
                params(6) = k;                  // B2
                params(7) = t_min + (t_max - t_min) * 0.25; // m1 (sos timing)
                params(8) = t_min + (t_max - t_min) * 0.75; // m2 (eos timing)
                params(9) = 1.0;                // q1
                params(10) = 1.0;               // q2
                params(11) = 1.0;               // v1
                params(12) = 1.0;               // v2
            }
            KlosFunctor functor(t, y, w, lb, ub);
            return optimize_functor(functor, params, lb, ub, max_fev);
        }
        case CurveType::ZHANG: {
            if(params.size() != 7) params = Eigen::VectorXd::Zero(7);
            lb = Eigen::VectorXd(7); ub = Eigen::VectorXd(7);
            lb << t_min, y_min - 0.5*y_amp, y_min, t_min - 30, 0.0, t_min - 30, 0.0;
            ub << t_max, y_min + 0.5*y_amp, y_max + 0.5*y_amp, t_max + 30, 50.0, t_max + 30, 50.0;
            if((params.array() == 1.0).all() || (params.array() == 0.0).all()) {
                params = (lb + ub) / 2.0;
                params(0) = t_min + (t_max - t_min) * 0.5;  // t0 (inflection/peak split)
                params(1) = y_min;                          // mn
                params(2) = y_max;                          // mx
                params(3) = t_min + (t_max - t_min) * 0.25; // sos timing
                params(4) = k;                              // rsp
                params(5) = t_min + (t_max - t_min) * 0.75; // eos timing
                params(6) = k;                              // rau
            }
            ZhangFunctor functor(t, y, w, lb, ub);
            return optimize_functor(functor, params, lb, ub, max_fev);
        }
        case CurveType::AG: {
            if(params.size() != 7) params = Eigen::VectorXd::Zero(7);
            lb = Eigen::VectorXd(7); ub = Eigen::VectorXd(7);
            lb << t_min, y_min - 0.5*y_amp, y_min, 0.0, 0.1, 0.0, 0.1;
            ub << t_max, y_min + 0.5*y_amp, y_max + 0.5*y_amp, 50.0, 10.0, 50.0, 10.0;
            if((params.array() == 1.0).all() || (params.array() == 0.0).all()) {
                params = (lb + ub) / 2.0;
                params(0) = t_min + (t_max - t_min) * 0.5; // t0 (peak)
                params(1) = y_min;                         // mn
                params(2) = y_max;                         // mx
                params(3) = 1.0 / d_t;                      // rsp (characteristic half-width ~ d_t)
                params(4) = 1.0;                            // a3
                params(5) = 1.0 / d_t;                      // rau
                params(6) = 1.0;                            // a5
            }
            AGFunctor functor(t, y, w, lb, ub);
            return optimize_functor(functor, params, lb, ub, max_fev);
        }
        case CurveType::DL: {
            if(params.size() != 6) params = Eigen::VectorXd::Zero(6);
            lb = Eigen::VectorXd(6); ub = Eigen::VectorXd(6);
            lb << y_min - 0.5*y_amp, y_min, t_min - 30, 0.0, t_min - 30, 0.0;
            ub << y_min + 0.5*y_amp, y_max + 0.5*y_amp, t_max + 30, 50.0, t_max + 30, 50.0;
            if((params.array() == 1.0).all() || (params.array() == 0.0).all()) {
                params = (lb + ub) / 2.0;
                params(0) = y_min;
                params(1) = y_max;
                params(2) = t_min + (t_max - t_min) * 0.25; // sos timing
                params(3) = k;                              // rsp
                params(4) = t_min + (t_max - t_min) * 0.75; // eos timing
                params(5) = k;                              // rau
            }
            DLFunctor functor(t, y, w, lb, ub);
            return optimize_functor(functor, params, lb, ub, max_fev);
        }
    }
    return false;
}

Eigen::VectorXd evaluate_curve(CurveType type, const Eigen::VectorXd& params, const Eigen::VectorXd& t) {
    Eigen::VectorXd fvec(t.size());
    switch(type) {
        case CurveType::BECK: {
            double mn = params[0], mx = params[1], sos = params[2], rsp = params[3], eos = params[4], rau = params[5];
            for(int i=0; i<t.size(); ++i) {
                fvec[i] = predict_beck(t[i], mn, mx, sos, rsp, eos, rau);
            }
            break;
        }
        case CurveType::ELMORE: {
            double mn = params[0], mx = params[1], sos = params[2], rsp = params[3], eos = params[4], rau = params[5], m7 = params[6];
            for(int i=0; i<t.size(); ++i) {
                fvec[i] = predict_elmore(t[i], mn, mx, sos, rsp, eos, rau, m7);
            }
            break;
        }
        case CurveType::GU: {
            double y0 = params[0], a1 = params[1], a2 = params[2], sos = params[3], rsp = params[4];
            double eos_p = params[5], rau = params[6], c1 = params[7], c2 = params[8];
            for(int i=0; i<t.size(); ++i) {
                fvec[i] = predict_gu(t[i], y0, a1, a2, sos, rsp, eos_p, rau, c1, c2);
            }
            break;
        }
        case CurveType::KLOS: {
            double a1 = params[0], a2 = params[1], b1 = params[2], b2 = params[3], c = params[4];
            double B1 = params[5], B2 = params[6], m1 = params[7], m2 = params[8], q1 = params[9], q2 = params[10], v1 = params[11], v2 = params[12];
            for(int i=0; i<t.size(); ++i) {
                fvec[i] = predict_klos(t[i], a1, a2, b1, b2, c, B1, B2, m1, m2, q1, q2, v1, v2);
            }
            break;
        }
        case CurveType::ZHANG: {
            double t0 = params[0], mn = params[1], mx = params[2], sos = params[3], rsp = params[4], eos = params[5], rau = params[6];
            for(int i=0; i<t.size(); ++i) {
                fvec[i] = predict_zhang(t[i], t0, mn, mx, sos, rsp, eos, rau);
            }
            break;
        }
        case CurveType::AG: {
            double t0 = params[0], mn = params[1], mx = params[2], rsp = params[3], a3 = params[4], rau = params[5], a5 = params[6];
            for(int i=0; i<t.size(); ++i) {
                fvec[i] = predict_ag(t[i], t0, mn, mx, rsp, a3, rau, a5);
            }
            break;
        }
        case CurveType::DL: {
            double mn = params[0], mx = params[1], sos = params[2], rsp = params[3], eos = params[4], rau = params[5];
            for(int i=0; i<t.size(); ++i) {
                fvec[i] = predict_dl(t[i], mn, mx, sos, rsp, eos, rau);
            }
            break;
        }
    }
    return fvec;
}

} // namespace phenology
