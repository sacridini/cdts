#ifndef PHENOLOGY_CURVES_H
#define PHENOLOGY_CURVES_H

#include <Eigen/Core>
#include <vector>

namespace phenology {

/**
 * Base Functor structure required by Eigen's NonLinearOptimization
 * template parameters: 
 *  _Scalar: type of numbers (e.g. double)
 *  NX: number of inputs (parameters to fit)
 *  NY: number of values (number of data points)
 */
template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
struct Functor {
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
};

// 1. Beck Functor (6 parameters)
struct BeckFunctor : Functor<double> {
    const Eigen::VectorXd& t;
    const Eigen::VectorXd& y;
    const Eigen::VectorXd& lb;
    const Eigen::VectorXd& ub;
    BeckFunctor(const Eigen::VectorXd& t_, const Eigen::VectorXd& y_, const Eigen::VectorXd& lb_, const Eigen::VectorXd& ub_) 
      : Functor<double>(6, t_.size()), t(t_), y(y_), lb(lb_), ub(ub_) {}
    int operator()(const Eigen::VectorXd &p, Eigen::VectorXd &fvec) const;
};

// 2. Elmore Functor (7 parameters)
struct ElmoreFunctor : Functor<double> {
    const Eigen::VectorXd& t;
    const Eigen::VectorXd& y;
    const Eigen::VectorXd& lb;
    const Eigen::VectorXd& ub;
    ElmoreFunctor(const Eigen::VectorXd& t_, const Eigen::VectorXd& y_, const Eigen::VectorXd& lb_, const Eigen::VectorXd& ub_) 
      : Functor<double>(7, t_.size()), t(t_), y(y_), lb(lb_), ub(ub_) {}
    int operator()(const Eigen::VectorXd &p, Eigen::VectorXd &fvec) const;
};

// 3. Gu Functor (9 parameters)
struct GuFunctor : Functor<double> {
    const Eigen::VectorXd& t;
    const Eigen::VectorXd& y;
    const Eigen::VectorXd& lb;
    const Eigen::VectorXd& ub;
    GuFunctor(const Eigen::VectorXd& t_, const Eigen::VectorXd& y_, const Eigen::VectorXd& lb_, const Eigen::VectorXd& ub_) 
      : Functor<double>(9, t_.size()), t(t_), y(y_), lb(lb_), ub(ub_) {}
    int operator()(const Eigen::VectorXd &p, Eigen::VectorXd &fvec) const;
};

// 4. Klos Functor (13 parameters)
struct KlosFunctor : Functor<double> {
    const Eigen::VectorXd& t;
    const Eigen::VectorXd& y;
    const Eigen::VectorXd& lb;
    const Eigen::VectorXd& ub;
    KlosFunctor(const Eigen::VectorXd& t_, const Eigen::VectorXd& y_, const Eigen::VectorXd& lb_, const Eigen::VectorXd& ub_) 
      : Functor<double>(13, t_.size()), t(t_), y(y_), lb(lb_), ub(ub_) {}
    int operator()(const Eigen::VectorXd &p, Eigen::VectorXd &fvec) const;
};

// 5. Zhang Functor (7 parameters)
struct ZhangFunctor : Functor<double> {
    const Eigen::VectorXd& t;
    const Eigen::VectorXd& y;
    const Eigen::VectorXd& lb;
    const Eigen::VectorXd& ub;
    ZhangFunctor(const Eigen::VectorXd& t_, const Eigen::VectorXd& y_, const Eigen::VectorXd& lb_, const Eigen::VectorXd& ub_) 
      : Functor<double>(7, t_.size()), t(t_), y(y_), lb(lb_), ub(ub_) {}
    int operator()(const Eigen::VectorXd &p, Eigen::VectorXd &fvec) const;
};

// 6. Asymmetric Gaussian (AG) Functor (7 parameters)
struct AGFunctor : Functor<double> {
    const Eigen::VectorXd& t;
    const Eigen::VectorXd& y;
    const Eigen::VectorXd& lb;
    const Eigen::VectorXd& ub;
    AGFunctor(const Eigen::VectorXd& t_, const Eigen::VectorXd& y_, const Eigen::VectorXd& lb_, const Eigen::VectorXd& ub_) 
      : Functor<double>(7, t_.size()), t(t_), y(y_), lb(lb_), ub(ub_) {}
    int operator()(const Eigen::VectorXd &p, Eigen::VectorXd &fvec) const;
};

// 7. Standard Double Logistic (DL) Functor (6 parameters)
struct DLFunctor : Functor<double> {
    const Eigen::VectorXd& t;
    const Eigen::VectorXd& y;
    const Eigen::VectorXd& lb;
    const Eigen::VectorXd& ub;
    DLFunctor(const Eigen::VectorXd& t_, const Eigen::VectorXd& y_, const Eigen::VectorXd& lb_, const Eigen::VectorXd& ub_) 
      : Functor<double>(6, t_.size()), t(t_), y(y_), lb(lb_), ub(ub_) {}
    int operator()(const Eigen::VectorXd &p, Eigen::VectorXd &fvec) const;
};

enum class CurveType {
    BECK,
    ELMORE,
    GU,
    KLOS,
    ZHANG,
    AG,
    DL
};

/**
 * Wrapper for Levenberg-Marquardt nonlinear curve fitting
 * @param t Time series segment (independent variable, e.g. DOY)
 * @param y Vegetation index segment (dependent variable)
 * @param params Initial guess for parameters, will be overwritten with best fit parameters
 * @param type The curve equation to use
 * @param max_fev Maximum function evaluations
 * @return true if converged, false otherwise
 */
bool fit_curve(const Eigen::VectorXd& t, const Eigen::VectorXd& y, 
               Eigen::VectorXd& params, CurveType type, int max_fev = 2000);

} // namespace phenology

#endif // PHENOLOGY_CURVES_H
