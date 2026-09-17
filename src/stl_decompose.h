#pragma once
#include <vector>

// STL: Seasonal-Trend decomposition procedure based on Loess (Cleveland,
// Cleveland, McRae & Terpenning, 1990), ported from R's `stats::stl()`
// (src/library/stats/src/stl.c, itself a hand-edited C translation of the
// original netlib Fortran `stl.f`).
//
// Scope: only the "periodic" seasonal window case (`s.window = "periodic"`,
// i.e. `s.degree = 0` with an effectively infinite seasonal-smoothing span)
// with R's non-robust defaults (`robust = FALSE`, i.e. `inner = 2` inner
// iterations, `outer = 0` robustness iterations) is implemented - this is
// the only configuration bfast() itself ever calls (`stl(Yt, "periodic")`,
// R/bfast.R). The general `s.window = <numeric>` case and robustness
// iterations (which need the median-based `stlrwt` reweighting step) are
// NOT ported - a separate follow-up if a standalone general-purpose STL is
// ever needed. The input must be a complete (no-NaN), regularly-spaced
// series - R's own `stl()` refuses missing values for this decomposition
// (its `decomp = "stlplus"` fallback, which tolerates NaN, is a different
// algorithm and not ported either); see bfast.cpp for how the NaN case is
// handled by its caller (linear interpolation before calling in here).

namespace cdts {
namespace stl {

// Matches R's `stl(ts(y, frequency = period), s.window = "periodic")$time.series[, "seasonal"]`.
// y.size() must be > 2*period (R's own minimum-length requirement).
std::vector<double> periodic_seasonal(const std::vector<double>& y, int period);

} // namespace stl
} // namespace cdts
