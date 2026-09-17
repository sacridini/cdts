# BFAST Lite (Multiple Breakpoint Detection)

## 1. Introduction

`bfastlite` answers a different question than [BFAST Monitor](bfast_monitor.md): instead of watching for the *next* disturbance in real time, it retrospectively segments an **entire** time series into the optimal number of pieces, each with its own trend + harmonic model — "how many structural changes happened in this series, and when?". It's the single-pass, modern alternative to the classic iterative `bfast()` (see [Scope](#2-scope) below): no seasonal-trend decomposition (STL) needed, and it handles missing data natively rather than requiring interpolation.

`cdts` provides a pixel-wise, C++/OpenMP-accelerated port of `bfastlite()`'s default configuration from the R package [`bfast`](https://github.com/bfast2/bfast) (Masiliūnas, Verbesselt *et al.*) and its [`strucchangeRcpp`](https://github.com/bfast2/strucchangeRcpp) dependency's `breakpoints()` — the Bai & Perron (2003) optimal multiple-breakpoint dynamic program — with the same Dask distribution strategy as [BFAST Monitor](bfast_monitor.md) and [Mann-Kendall](mann_kendall.md).

## 2. Scope

Ported: the default `breaks="LWZ"` model-selection criterion (Liu, Wu & Zidek, 1997) choosing the optimal number of breaks automatically, over the same `response ~ trend + harmon` design matrix as [BFAST Monitor](bfast_monitor.md#2-background-what-the-model-fits), fit via [Brown-Durbin-Evans recursive residuals](https://en.wikipedia.org/wiki/CUSUM) for an O(n²) (not O(n³)) segment-RSS table.

Not ported here: the classic iterative `bfast()` (which additionally needs STL decomposition of the series into trend/season/remainder, and separately detects breaks in each component across converging iterations) - see the [BFAST tutorial](bfast.md) for that.

## 3. Using bfastlite in CDTS

### Step 3.1: Via the xarray accessor (recommended)

```python
import cdts

# annual_ndvi: (time, y, x) DataArray, 16-day composites (frequency=23/year) from 2010
result = annual_ndvi.cdts.run_bfast_lite(
    start_time=2010.0,
    frequency=23,
    h=0.15,             # minimum segment size, as a fraction of series length
    max_breaks_output=5, # cap on how many breakpoints to report per pixel
)

result = result.compute()
n_breaks = result.sel(metric="n_breaks")
first_break = result.sel(metric="breakpoint_idx_1")   # NaN where n_breaks == 0
```

### Step 3.2: Direct Dask array entry point

```python
from cdts.bfast import run_bfast_lite_dask

# arr: dask.array.Array, shape (time, y, x)
out = run_bfast_lite_dask(arr, start_time=2010.0, frequency=23).compute()
```

## 4. Understanding the Output

Because the number of detected breaks varies per pixel, the output caps how many are reported via `max_breaks_output` (default 5) - extra slots are `NaN`-padded. Row names come from `cdts.bfast.bfl_metric_names(max_breaks_output)`:

| Metric | Meaning |
| :--- | :--- |
| `n_breaks` | Number of breakpoints selected by the LWZ criterion (0 if none). |
| `rss` | Total residual sum of squares at the selected number of breaks. |
| `lwz` | LWZ score at the selected number of breaks (the value that was minimized). |
| `n_valid` | Number of valid (non-NaN) observations used. |
| `valid` | `1.0` if the series had enough observations to fit at all. |
| `breakpoint_idx_1` .. `breakpoint_idx_{max_breaks_output}` | 0-based indices into the (NaN-dropped) series where each breakpoint falls, in chronological order. `NaN` past `n_breaks`. |

## 5. Validation

Verified directly against R's `bfastlite()` (package `bfast`, via `strucchangeRcpp`) across 6 scenarios — a single mid-series break, no break, two candidate breaks (where the LWZ criterion happened to prefer one), NaN gaps, monthly (`frequency=12`) data, and a non-default `h`. All 6 matched `n_breaks` and every breakpoint position exactly, with `rss` differences on the order of 1e-6 to 1e-7 (consistent with the R↔C++ file round trip, not any algorithmic divergence).

Two real numerical bugs were caught and fixed during this validation (see the commit history for `src/bfast_lite.cpp`) - both are worth knowing about if you extend this code:

1. **Ridge-regularizing the recursive-residual recursion at every step biases the result.** The sum of squared recursive residuals is only exactly equal to the segment's OLS RSS when every step is an *exact* least-squares fit; even a tiny ridge term, applied at every one of the ~n steps, measurably shifted the final cumulative sum. The fix refactorizes the Gram matrix from scratch each step (unregularized) instead of propagating a ridge-stabilized inverse.
2. **An off-by-one in the dynamic program's segment count.** An earlier version added one extra trailing segment when extracting the final answer for "m breaks" (a convention copied from `strucchangeRcpp`'s own column-offset indexing, which doesn't apply to this port's differently-indexed DP table) - it silently returned an (m+1)-break answer while reporting `m`. Caught because the reported breakpoint positions didn't match R's at all for any case with an actual break.

## 6. Performance

Benchmarked against R's real `bfastlite()` (300 pixels, 150-observation series, `h=0.15`, `order=3`):

| | per pixel |
| :--- | :--- |
| R `bfastlite()` (single-thread) | 34.5 ms |
| cdts, single-thread | 11.2 ms (**~3.1x**) |

On top of that single-thread gain, `n_jobs=-1` parallelizes across cores — but per the library-wide threading rule (see the [README](https://github.com/sacridini/cdts#architecture--threading-safety)), it always **reserves one CPU core** so the host stays responsive, rather than saturating every logical core. Measured at a production-realistic scale (20,000 pixels, same machine, 16 logical cores so `n_jobs=-1` uses 15 of them): `n_jobs=1` took 249.3s vs. `n_jobs=-1`'s 45.5s — a further **~5.5x** on top of the single-thread number above. This is intentionally short of a full 16x/15x linear scaling: reserving a core trades a bit of raw multi-core throughput for keeping the machine usable while a large batch job runs.

This is a much more modest speedup than [BFAST Monitor](bfast_monitor.md)'s (~384x/~2180x). That's expected, not a regression: `bfastmonitor`'s R-side cost is dominated by per-call interpreter/object overhead (`data.frame` construction, S3 dispatch), which a tight C++ loop mostly eliminates outright. `bfastlite`'s dynamic program is genuinely CPU-bound real computation (an O(n²) segment-RSS table times the number of candidate breaks) on *both* sides, so the C++ port's advantage here reflects actual compute efficiency (cache-friendly small dense linear algebra, no interpreter overhead per arithmetic op, OpenMP) rather than eliminating call overhead - still a solid, meaningful win, just a different profile.

## 7. References

- Masiliūnas, D., Tuck, S. L., Melo, M. R. S., Zeileis, A., & Verbesselt, J. (in prep). BFAST Lite: A lightweight, non-interactive method for time series structural change detection. See the [`bfast` package documentation](https://bfast2.github.io/) for the current reference.
- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. **Journal of Applied Econometrics**, 18(1), 1–22. [https://doi.org/10.1002/jae.659](https://doi.org/10.1002/jae.659)
- Liu, J., Wu, S., & Zidek, J. V. (1997). On segmented multivariate regression. **Statistica Sinica**, 7(2), 497–525.
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. **Journal of the Royal Statistical Society: Series B**, 37(2), 149–163.
- Zeileis, A., Kleiber, C., Krämer, W., & Hornik, K. (2003). Testing and dating of structural changes in practice. **Computational Statistics & Data Analysis**, 44(1–2), 109–123. [https://doi.org/10.1016/S0167-9473(03)00030-6](https://doi.org/10.1016/S0167-9473(03)00030-6)
- R packages [`bfast`](https://github.com/bfast2/bfast) and [`strucchangeRcpp`](https://github.com/bfast2/strucchangeRcpp) (the reference implementations this port was validated against).
