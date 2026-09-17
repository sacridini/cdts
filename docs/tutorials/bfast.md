# BFAST (Classic Iterative Trend + Season Break Detection)

## 1. Introduction

`bfast` is the original, classic algorithm behind [BFAST Monitor](bfast_monitor.md) and [BFAST Lite](bfast_lite.md): it decomposes a time series into trend, seasonal, and remainder components, and *iteratively* re-estimates each one while searching for structural breaks in both - "how many changes happened, are they trend changes or seasonal (phenological) changes, and when?".

`cdts` provides a pixel-wise, C++/OpenMP-accelerated port of `bfast()`'s default configuration from the R package [`bfast`](https://github.com/bfast2/bfast) (Verbesselt *et al.*) and its [`strucchangeRcpp`](https://github.com/bfast2/strucchangeRcpp) dependency's `breakpoints()`, with the same Dask distribution strategy as [BFAST Lite](bfast_lite.md) and [BFAST Monitor](bfast_monitor.md).

## 2. Background: how the algorithm works

Given a series `Yt`, `bfast()` alternates between two segmented regressions until they agree:

1. **Trend step**: fit `Vt = Yt - St ~ trend` (deseasonalized series against a linear trend), allowing the intercept *and* slope to change at an optimal number of breakpoints (the same Bai & Perron, 2003 dynamic program [BFAST Lite](bfast_lite.md) uses). This gives a new trend component `Tt`.
2. **Season step**: fit `Wt = Yt - Tt ~ harmonic` (detrended series against a harmonic regression), again allowing the harmonic coefficients to change at an optimal number of breakpoints. This gives a new seasonal component `St`.

The two steps repeat - each one's residual feeds the other's next fit - until neither breakpoint set changes from the previous iteration (or `max_iter` is reached). Because the trend step needs a seasonal estimate before any iteration has run, `St` is seeded once beforehand via an STL "periodic" decomposition (Cleveland, Cleveland, McRae & Terpenning, 1990) - the same `stl(Yt, "periodic")` call R's own `bfast()` makes.

This is why `bfast()` needs machinery [BFAST Lite](bfast_lite.md) and [BFAST Monitor](bfast_monitor.md) don't: STL decomposition (for the seasonal seed) *and* the Bai-Perron dynamic program applied twice per iteration (trend and season, separately) rather than once.

## 3. Scope

Ported: the iterative trend/season loop described above, using the optimal number of breaks at each step chosen by minimizing **BIC** (`breakpoints()`'s own default when `breaks = NULL`, which is what classic `bfast()` uses - note this differs from [BFAST Lite](bfast_lite.md)'s own default of `breaks = "LWZ"`), and the STL "periodic" seasonal seed.

Not ported (documented deviations from R's `bfast()`):

1. **`season = "harmonic"` only.** R's own default is `season = "dummy"` (a seasonal-factor/dummy-variable model); `"harmonic"` is R's other fully-supported option (used in Verbesselt, Zeileis & Herold, 2012). This port only implements `"harmonic"`, matching [BFAST Lite](bfast_lite.md#2-scope)'s and [BFAST Monitor](bfast_monitor.md#2-background-what-the-model-fits)'s own harmonic-only design matrix, so all three algorithms share the same regressor convention.
2. **The preliminary `sctest(efp(...))` pre-check is ported** (this needed a second p-value approximation table beyond the one [BFAST Monitor](bfast_monitor.md) already ports, since the retrospective and monitoring processes use different asymptotics - `strucchangeRcpp`'s `sc.me` critical-value table for the "Brownian bridge increments" limiting process, dumped directly from the installed R package). Before every breakpoint search, R's `bfast()` runs a structural-stability test (`efp()`/`sctest()`, the retrospective OLS-MOSUM test) and only searches for breaks if it rejects stability at the given `level` (default `0.05`). An earlier version of this port skipped this and always attempted the BIC search directly - this turned out to matter empirically, not just in theory: it measurably over-detected weak/borderline breaks that R's default gate suppresses (see [Validation](#6-validation) below for the concrete case that caught this).
3. **`decomp = "stl"` only, and NaN gaps are interpolated for it.** R's `decomp = "stlplus"` (an NA-tolerant STL variant) is not ported. R's own `decomp = "stl"` path refuses series with any missing values outright; this port instead linearly interpolates internal NaN gaps (and extrapolates flat at the edges) *only* to build a complete, regularly-spaced series for the one-time STL seasonal seed. The iterative trend/season fits themselves still skip NaN rows natively, exactly as [BFAST Lite](bfast_lite.md) and [BFAST Monitor](bfast_monitor.md) do.

## 4. Using bfast in CDTS

### Step 4.1: Via the xarray accessor (recommended)

```python
import cdts

# annual_ndvi: (time, y, x) DataArray, 16-day composites (frequency=23/year) from 2010
result = annual_ndvi.cdts.run_bfast(
    start_time=2010.0,
    frequency=23,
    h=0.15,                # minimum segment size, as a fraction of valid observations
    max_breaks_trend=5,    # cap on how many trend breakpoints to report per pixel
    max_breaks_season=5,   # cap on how many season breakpoints to report per pixel
)

result = result.compute()
n_trend_breaks = result.sel(metric="n_trend_breaks")
magnitude = result.sel(metric="magnitude")          # size of the largest trend jump (0 if none)
time_of_break = result.sel(metric="time")            # fractional-year time of that jump (NaN if none)
first_trend_break = result.sel(metric="trend_breakpoint_idx_1")
```

### Step 4.2: Direct Dask array entry point

```python
from cdts.bfast import run_bfast_dask

# arr: dask.array.Array, shape (time, y, x)
out = run_bfast_dask(arr, start_time=2010.0, frequency=23).compute()
```

## 5. Understanding the Output

Row names come from `cdts.bfast.bf_metric_names(max_breaks_trend, max_breaks_season)`:

| Metric | Meaning |
| :--- | :--- |
| `n_trend_breaks` | Number of trend breakpoints at convergence. |
| `n_season_breaks` | Number of season (phenological) breakpoints at convergence. |
| `magnitude` | Size of the largest trend-component jump (the difference between the two segments' fitted levels at the break), `0` if `n_trend_breaks == 0`. Matches R's `bf$Magnitude`. |
| `time` | Fractional-year time of that largest jump, `NaN` if `n_trend_breaks == 0`. Matches R's `bf$jump$x` - **not** `bf$Time`, which (in R itself) is actually just the raw observation index of the break, redundant with `Vt.bp`/`trend_breakpoint_idx`, despite its name. |
| `n_iter` | Number of trend/season re-estimation iterations run before convergence (or `max_iter`). |
| `n_valid` | Number of valid (non-NaN) observations used. |
| `valid` | `1.0` if the series had enough observations to fit at all (needs more than `2 * frequency` total observations and at least `min_valid` non-NaN ones). |
| `trend_breakpoint_idx_1..{max_breaks_trend}` | 0-based indices into the original series where each trend break falls, chronological order, `NaN` past `n_trend_breaks`. |
| `season_breakpoint_idx_1..{max_breaks_season}` | Same, for season breaks. |

## 6. Validation

Verified directly against real R output (R 4.4.2, `bfast` 1.7.2, `strucchangeRcpp` 1.5.4.1.0.1, installed from CRAN specifically to validate this port), matching `h=0.15`, `season="harmonic"`, `breaks=NULL`, `level=0.05`, `decomp="stl"`, `type="OLS-MOSUM"` on both sides, across 4 synthetic scenarios (`frequency=23`, 300 observations): an injected trend-only break, a season-only (seasonal-amplitude) break, no break, and both a trend and a season break together.

| Scenario | R `Vt.bp` (trend) | cdts `trend_breakpoint_idx` | R `Wt.bp` (season) | cdts `season_breakpoint_idx` | R `Magnitude` | cdts `magnitude` | R iterations | cdts `n_iter` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| trend break only | 149 | **149** | none | **none** | 0.3998 | **0.3998** | 2 | **2** |
| season break only | none | **none** | none | **none** | 0 | **0** | 1 | **1** |
| no break | none | **none** | none | **none** | 0 | **0** | 1 | **1** |
| both | 149 | **149** | none | **none** | 0.4036 | **0.4036** | 2 | **2** |

Every field matches exactly (breakpoint position to the observation, magnitude to 4 decimal places, iteration count identical). `time` (fractional-year) was cross-checked against R's `bf$jump$x` directly (`2006.478...` on both sides for the trend-break scenario - see the note in [Understanding the Output](#5-understanding-the-output) about why `bf$Time` itself is the wrong field to compare against).

The "season break only" scenario is worth explaining, since it's *why* the preliminary `sctest.efp` pre-check (see [Scope](#3-scope)) had to be ported rather than skipped: a pure seasonal-**amplitude** change (the injected scenario here, and re-confirmed up to a 20x amplitude jump and with a quarter-cycle phase shift instead) turns out to be close to undetectable by the OLS-MOSUM test in *either* R or this port - it's a sum-of-residuals test, so a zero-mean oscillation that's simply scaled up or down averages out to ~0 over the test's window regardless of amplitude, giving the test very low power against this specific kind of change. This isn't a bug in either implementation: with the pre-check bypassed (`level=1.0`) on both sides, R finds `Wt.bp=150` (1-indexed) and cdts finds `season_breakpoint_idx=149` (0-indexed) - the same observation - confirming the underlying BIC segmentation search itself agrees between the two; it's specifically R's (and now this port's) shared statistical gate that suppresses it by default, for this kind of signal. `tests/test_bfast.py` encodes both halves of this (detects it with the pre-check disabled, suppresses it at the default level) so the behavior stays pinned down.

One residual, minor discrepancy: with the pre-check bypassed (`level=1.0`, not the default), R additionally reports a trend break (`Vt.bp=144`) for the season-break scenario that cdts does not. This only shows up in that non-default diagnostic mode, not at `level=0.05` (real usage), and wasn't chased further.

Output is bit-identical between `n_jobs=1` and `n_jobs=-1` (verified in `tests/test_bfast.py`).

## 7. Performance

Each iteration costs roughly two [BFAST Lite](bfast_lite.md)-sized dynamic programs (one over the `~ trend` design, one over `~ harmonic`) plus two OLS-MOSUM pre-checks, and there's one STL decomposition up front - so a single pixel costs on the order of `2 * n_iter` times a comparable bfastlite fit, where `n_iter` is usually small in practice (2-4 iterations to convergence for a single, clear break) but can reach `max_iter` (default 10) for noisy or ambiguous series.

Benchmarked directly against the same real R installation used for validation above (single series, `frequency=23`, 300 observations, mean of 5 reps each, same machine):

| Scenario | R `bfast()` | cdts (single-thread call) | Speedup |
| :--- | :--- | :--- | :--- |
| trend break (2 iterations) | 105.6 ms | 25.2 ms | **~4.2x** |
| both breaks (2 iterations) | 106.5 ms | 25.5 ms | **~4.2x** |
| season break / no break (1 iteration, pre-check stops immediately) | ~4.3 ms | ~0.1 ms | **~40x** |

The 1-iteration cases show a much larger gap because R's fixed per-call overhead (`data.frame`/`model.frame` construction, S3 dispatch, one `stl()` Fortran call) dominates when there's little else to do; the 2-iteration cases are a fairer read on the actual compute advantage, similar in spirit to [BFAST Lite](bfast_lite.md#6-performance)'s own discussion of this effect.

On top of the single-call number, `n_jobs=-1` parallelizes across pixels via OpenMP (2,000 pixels, 300 observations each, half with an injected trend break, on a 20-logical-core machine): `n_jobs=1` takes ~25.2s (12.6 ms/pixel) vs. `n_jobs=-1`'s ~4.0-4.5s (2.0-2.2 ms/pixel) - a **~5.6-6.4x** speedup, short of the full ~19x (one core reserved) because bfast's iterative STL+dual-DP workload per pixel is heavier and less uniform than bfastlite's single DP, per the library-wide threading rule (see the [README](https://github.com/sacridini/cdts#architecture--threading-safety)).

**Batch-call optimization.** Because `bfast()` calls its segmented-regression search (the O(n²) RSS-triangle + Bai-Perron DP) up to `2 * max_iter` times per pixel - versus [BFAST Lite](bfast_lite.md)'s single call - its heap-allocation traffic is proportionally much higher too. Two changes address this, both purely internal (no output change - verified bit-identical against the pre-optimization output on the 2,000-pixel benchmark above):

1. The `~ trend` and `~ harmonic` design matrices depend only on `n_time`/`start_time`/`frequency`/`order`, which are shared by every pixel in a batch call, so they're now built once per `fit_bfast_batch()` call instead of once per pixel (matching how bfast_lite.cpp/bfast_monitor.cpp already build theirs).
2. The RSS-triangle and dynamic-program tables inside the segmented-regression search are now held in a small per-OpenMP-thread scratch struct and reused across that thread's whole share of pixels and iterations (analogous to bfast_monitor.cpp's `BFMScratch`), instead of being freshly heap-allocated on every one of those `2 * max_iter` calls.

The single-thread (`n_jobs=1`) time barely moved (12.6 ms/pixel before and after), but the parallel (`n_jobs=-1`) time dropped from 3.56 ms/pixel to ~2.0-2.2 ms/pixel - i.e. the fix mattered far more under parallelism than in isolation. That asymmetry itself is informative: it points at cross-thread heap-allocator contention (many OpenMP threads repeatedly allocating/freeing small vectors concurrently) rather than raw per-pixel compute as the thing that improved, which is exactly what reusing thread-local buffers targets.

## 8. References

- Verbesselt, J., Hyndman, R., Newnham, G., & Culvenor, D. (2010). Detecting trend and seasonal changes in satellite image time series. **Remote Sensing of Environment**, 114(1), 106-115. [https://doi.org/10.1016/j.rse.2009.08.014](https://doi.org/10.1016/j.rse.2009.08.014)
- Verbesselt, J., Zeileis, A., & Herold, M. (2012). Near real-time disturbance detection using satellite image time series. **Remote Sensing of Environment**, 123, 98-108. [https://doi.org/10.1016/j.rse.2011.09.024](https://doi.org/10.1016/j.rse.2011.09.024)
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. **Journal of Official Statistics**, 6(1), 3-73.
- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. **Journal of Applied Econometrics**, 18(1), 1-22. [https://doi.org/10.1002/jae.659](https://doi.org/10.1002/jae.659)
- Chu, C.-S. J., Hornik, K., & Kuan, C.-M. (1995). MOSUM tests for parameter constancy. **Biometrika**, 82(3), 603-617. (the OLS-MOSUM retrospective test the preliminary `sctest.efp` pre-check uses, distinct from Chu, Stinchcombe & White 1996's monitoring boundary already cited in the [BFAST Monitor](bfast_monitor.md) tutorial)
- Zeileis, A., Leisch, F., Hornik, K., & Kleiber, C. (2002). strucchange: An R package for testing for structural change in linear regression models. **Journal of Statistical Software**, 7(2), 1-38. [https://doi.org/10.18637/jss.v007.i02](https://doi.org/10.18637/jss.v007.i02)
- R packages [`bfast`](https://github.com/bfast2/bfast) and [`strucchangeRcpp`](https://github.com/bfast2/strucchangeRcpp) (the reference implementation this port follows and was validated against), and R's own `stats::stl()`.
