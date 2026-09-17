# BFAST Monitor (Near-Real-Time Change Detection)

## 1. Introduction

`bfastmonitor` answers a different question than [LandTrendr](landtrendr.md), [CCDC](ccdc.md), or [Mann-Kendall](mann_kendall.md): instead of retrospectively segmenting an entire time series, it asks "**is a disturbance happening right now, in the most recent observations?**". It fits a trend + harmonic regression model on a stable "history" period, then monitors every new observation for the first point at which the residuals' fluctuation process crosses a statistical significance boundary — the same near-real-time monitoring logic used by operational deforestation-alert systems (RADD, GLAD).

`cdts` provides a pixel-wise, C++/OpenMP-accelerated port of `bfastmonitor()`'s default `type="OLS-MOSUM"` monitoring process, from the R package [`bfast`](https://github.com/bfast2/bfast) (Verbesselt *et al.*) and its [`strucchangeRcpp`](https://github.com/bfast2/strucchangeRcpp) dependency (the OLS-MOSUM boundary-crossing test of Chu, Stinchcombe & White, 1996), with the same Dask distribution strategy as [Mann-Kendall](mann_kendall.md): OpenMP parallelizes across pixels within a chunk, Dask distributes chunks across cores/workers.

### Scope of this port

Only `bfastmonitor()`'s default monitoring process (`type="OLS-MOSUM"`) and `history="all"` (the entire pre-monitoring period used as the stable history) are implemented. R's default `history="ROC"` (which auto-trims unstable older history via a reversed-CUSUM test) and `history="BP"` are **not** ported. See the [BFAST](bfast.md) and [BFAST Lite](bfast_lite.md) tutorials for the classic iterative `bfast()` and single-pass `bfastlite()`, respectively.

!!! warning "`history=\"all\"` has a higher false-positive rate than the nominal `alpha` in practice"
    Cross-validated directly against R: on 50 pure-noise (no injected break) synthetic series at `alpha=0.05`, R's own `bfastmonitor(..., history="all")` flagged a "break" on **~44%** of them — not the ~5% the significance level suggests — and `cdts` matched this (~42%) exactly. This is a property of `bfastmonitor` itself with an untrimmed history, not a port bug (it's presumably why R defaults to `history="ROC"` instead). Prefer a genuinely stable, disturbance-free history window when calling this, and treat single detections with the same caution you'd apply to any `alpha=0.05` test run many times (i.e. correct for multiple comparisons across your pixels, or raise `alpha`/tune `h`/`period`).

---

## 2. Background: What the Model Fits

For each pixel, the series is split at `monitor_start_time` into:

- **History period** (`time < monitor_start_time`): a linear model `response ~ trend + harmon` is fit by OLS — `trend` is a simple 1..n index, `harmon` is `order` pairs of `cos(2πkt)`/`sin(2πkt)` seasonal terms (default `order=3`). This exactly matches `bfastpp()`'s design matrix.
- **Monitoring period** (`time >= monitor_start_time`): residuals against the *history-fit* coefficients are computed for every new observation, and a rolling **OLS-MOSUM** (moving sum) process is tracked against a boundary that grows over time (Chu-Stinchcombe-White, 1996). The first crossing is the detected breakpoint.

Time here is **synthetic and regular** — `start_time + i/frequency` — matching R's `ts`/`time()` semantics exactly (not real per-observation dates). `start_time` should be an integer (e.g. `2000.0`) so the harmonic terms align with calendar seasons.

### Parameters that must match a discrete grid

`h` (MOSUM window, as a fraction of history length) and `period` (how many "history lengths" ahead the monitoring boundary's guarantee covers) index a pre-simulated critical-value table (ported directly from `strucchangeRcpp`), so they must be:

- `h` ∈ `{0.25, 0.5, 1.0}`
- `period` ∈ `{2, 4, 6, 8, 10}`

Any other value raises an error immediately (matching `strucchangeRcpp`'s own behavior) rather than silently extrapolating.

---

## 3. Using bfastmonitor in CDTS

### Step 3.1: Via the xarray accessor (recommended)

```python
import numpy as np
import cdts

# annual_ndvi: (time, y, x) DataArray, 16-day composites (frequency=23/year),
# starting January of year 2010.
result = annual_ndvi.cdts.run_bfast_monitor(
    start_time=2010.0,
    monitor_start_time=2022.0,  # start monitoring from 2022 onward
    frequency=23,
    h=0.25,
    period=10,
    alpha=0.05,
)

result = result.compute()
disturbed = result.sel(metric="has_break") == 1.0
break_time = result.sel(metric="breakpoint")       # fractional-year time of the first detected break
break_idx = result.sel(metric="breakpoint_idx")    # index into the input time series
magnitude = result.sel(metric="magnitude")         # median residual over the monitoring period
```

### Step 3.2: Direct Dask array entry point

```python
from cdts.bfast import run_bfast_monitor_dask

# arr: dask.array.Array, shape (time, y, x)
out = run_bfast_monitor_dask(
    arr, start_time=2010.0, monitor_start_time=2022.0, frequency=23,
).compute()
```

---

## 4. Understanding the Output

The result has a `metric` dimension with 7 values (`cdts.bfast.BFM_METRIC_NAMES`):

| Metric | Meaning |
| :--- | :--- |
| `breakpoint` | Fractional-year time of the first detected break, or `NaN` if none. |
| `breakpoint_idx` | 0-based index into the input time series of the first detected break, or `NaN`. |
| `magnitude` | Median residual over the monitoring period (from `monitor_start_time` onward) against the history-fit model — the size/direction of the shift. |
| `sigma` | Residual standard error of the history-period fit. |
| `n_history` | Number of valid (non-NaN) observations used to fit the history model. |
| `has_break` | `1.0` if a break was detected, `0.0` otherwise. |
| `valid` | `1.0` if the history period had enough observations to fit at all (`0.0` means too few history observations relative to the number of harmonic+trend regressors, or a degenerate fit — everything else is `NaN` for that pixel). |

---

## 5. Validation

Verified directly against R's `bfastmonitor()` (package `bfast`, using its `strucchangeRcpp` dependency) across 6 scenarios — a mid-series break, no break, an early break right at the start of monitoring, NaN gaps in the history, monthly (`frequency=12`) data, and a non-default `h`/`period` combination. All 6 matched to floating-point tolerance (`breakpoint_idx`, `magnitude`, `sigma`, and `n_history` identical; `breakpoint`/`magnitude` differences on the order of 1e-11 to 1e-13, consistent with the R↔C++ file round trip rather than any algorithmic divergence).

---

## 6. References

- Verbesselt, J., Hyndman, R., Zeileis, A., & Culvenor, D. (2010). Phenological change detection while accounting for abrupt and gradual trends in satellite image time series. **Remote Sensing of Environment**, 114(12), 2970–2980. [https://doi.org/10.1016/j.rse.2010.08.003](https://doi.org/10.1016/j.rse.2010.08.003)
- Verbesselt, J., Zeileis, A., & Herold, M. (2012). Near real-time disturbance detection using satellite image time series. **Remote Sensing of Environment**, 123, 98–108. [https://doi.org/10.1016/j.rse.2012.02.022](https://doi.org/10.1016/j.rse.2012.02.022)
- Chu, C.-S. J., Stinchcombe, M., & White, H. (1996). Monitoring structural change. **Econometrica**, 64(5), 1045–1065. [https://doi.org/10.2307/2171955](https://doi.org/10.2307/2171955)
- Zeileis, A., Leisch, F., Kleiber, C., & Hornik, K. (2005). Monitoring structural change in dynamic econometric models. **Journal of Applied Econometrics**, 20(1), 99–121. [https://doi.org/10.1002/jae.776](https://doi.org/10.1002/jae.776)
- R package [`bfast`](https://github.com/bfast2/bfast) and [`strucchangeRcpp`](https://github.com/bfast2/strucchangeRcpp) (the reference implementations this port was validated against).
