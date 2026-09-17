#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// bfastmonitor: near-real-time structural change monitoring, ported from the
// R package `bfast` (Verbesselt et al.) and its `strucchangeRcpp` dependency
// (Zeileis, Leisch, Kleiber & Hornik's OLS-MOSUM monitoring process - Chu,
// Stinchcombe & White, 1996, "Monitoring Structural Change", Econometrica).
//
// Scope of this port (see docs/tutorials/bfast_monitor.md for the full
// rationale): only the default `type="OLS-MOSUM"` monitoring process and
// `history="all"` (the entire pre-monitoring period as the stable history,
// no ROC/BP auto-trimming of unstable older history) are implemented. The
// classic iterative `bfast()` and `bfastlite()` (which both additionally
// need STL decomposition and/or the Bai-Perron optimal-breakpoint dynamic
// program) are a separate, future port.

namespace cdts {
namespace bfastmonitor {

struct BFMResult {
    double breakpoint = std::nan("");     // fractional-year time of the first detected break, or NaN
    double breakpoint_idx = std::nan(""); // 0-based index into the full series, or NaN
    double magnitude = std::nan("");      // median residual over the monitoring period (from `start` onward)
    double sigma = std::nan("");          // residual standard error of the history fit
    double n_history = std::nan("");      // number of valid (non-NaN) observations used in the history fit
    double has_break = 0.0;               // 1.0 if a break was detected, 0.0 otherwise
    double valid = 0.0;                   // 1.0 if the history period had enough observations to fit at all
};

// Runs bfastmonitor on a single pixel's time series.
// y: response values, may contain NaN (skipped row-wise).
// n_time equally-spaced observations at `start_time + i/frequency`, i=0..n_time-1
// (matching R's `stats::ts`/`time()` semantics - synthetic regular time from
// frequency and start, not real per-observation dates).
BFMResult bfast_monitor(
    const std::vector<double>& y,
    double start_time,
    double monitor_start_time,
    int frequency,
    int order = 3,
    double h = 0.25,
    int period = 10,
    double alpha = 0.05);

// Batch entry point: values_array [n_pixels, n_time] -> out [7, n_pixels].
// Row order: breakpoint, breakpoint_idx, magnitude, sigma, n_history,
// has_break, valid.
pybind11::array_t<double> fit_bfast_monitor_batch(
    pybind11::array_t<double> values_array,
    double start_time,
    double monitor_start_time,
    int frequency,
    int order = 3,
    double h = 0.25,
    int period = 10,
    double alpha = 0.05,
    int min_valid = 10,
    int n_jobs = -1);

} // namespace bfastmonitor
} // namespace cdts
