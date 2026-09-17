import numpy as np
import dask.array as da
from typing import Optional

from cdts._core.bfastmonitor import fit_bfast_monitor_batch
from cdts._core.bfastlite import fit_bfast_lite_batch

# breakpoint, breakpoint_idx, magnitude, sigma, n_history, has_break, valid
N_BFM_METRICS = 7
BFM_METRIC_NAMES = ["breakpoint", "breakpoint_idx", "magnitude", "sigma", "n_history", "has_break", "valid"]


def run_bfast_monitor_dask(
    arr: da.Array,
    start_time: float,
    monitor_start_time: float,
    frequency: int,
    order: int = 3,
    h: float = 0.25,
    period: int = 10,
    alpha: float = 0.05,
    min_valid: int = 10,
    n_jobs: int = -1,
) -> da.Array:
    """
    Pixel-wise bfastmonitor (near-real-time structural change monitoring)
    across a Dask array's time axis, ported from the R package `bfast`
    (Verbesselt et al., 2012, doi:10.1016/j.rse.2012.02.022) and its
    `strucchangeRcpp` dependency's OLS-MOSUM monitoring process (Chu,
    Stinchcombe & White, 1996), to a C++/OpenMP backend for per-pixel
    throughput, with the same Dask map_blocks strategy as
    run_mann_kendall_dask.

    Fits a trend + harmonic regression model on a "history" period, then
    monitors the subsequent period for the first point at which the
    OLS-MOSUM fluctuation process crosses a significance boundary - i.e.
    "is a disturbance currently happening, and when did it start". This is
    the near-real-time complement to LandTrendr/CCDC's retrospective,
    whole-series segmentation.

    Input array shape: (time, y, x), one equally-spaced observation per
    `1/frequency` (e.g. frequency=23 for 16-day composites) - matching R's
    `ts`/`time()` semantics: time is synthetic and regular
    (`start_time + i/frequency`), not derived from real per-observation
    dates. Pixels with fewer than `min_valid` non-NaN history observations
    are all-NaN in the output (except `valid`, which is 0).

    IMPORTANT - scope: only the default `type="OLS-MOSUM"` monitoring
    process is implemented, and only `history="all"` (the entire
    pre-`monitor_start_time` period is used as the stable history, with no
    ROC/BP auto-trimming of unstable older history, unlike R's default
    `history="ROC"`). See the bfastmonitor tutorial for the full
    method background and how this compares to the classic iterative
    `bfast()`/`bfastlite()` (not yet ported - both additionally need STL
    decomposition and/or the Bai-Perron optimal-breakpoint dynamic program).

    start_time: the series' start time, in the same fractional-year-like
    units as `monitor_start_time` (e.g. 2000.0 for a series starting
    January of year 2000). Must be an integer number of years for the
    harmonic terms to align with calendar seasons (matches R's `ts` `start`
    convention).

    monitor_start_time: the time at which monitoring begins (the boundary
    between the "history" and "monitoring" periods), in the same units as
    `start_time` (e.g. 2015.5 to start monitoring mid-2015).

    h: MOSUM window size, as a fraction of the history length. Must be one
    of 0.25, 0.5, 1.0 (the OLS-MOSUM critical-value table's grid - matches
    strucchangeRcpp's own restriction).

    period: monitoring period parameter (`end` in R's bfastmonitor). Must be
    one of 2, 4, 6, 8, 10 (the critical-value table's grid).

    order: harmonic order for the seasonal regressors (default 3, capped at
    `frequency`).
    """
    if h not in (0.25, 0.5, 1.0):
        raise ValueError(f"h must be one of 0.25, 0.5, 1.0; got {h!r}")
    if period not in (2, 4, 6, 8, 10):
        raise ValueError(f"period must be one of 2, 4, 6, 8, 10; got {period!r}")

    def _block(block):
        if block.size == 0:
            return np.full((N_BFM_METRICS, block.shape[1], block.shape[2]), np.nan, dtype=np.float32)

        time_steps, rows, cols = block.shape
        pixels = rows * cols

        values_2d = np.ascontiguousarray(block.reshape(time_steps, pixels).T)

        out = fit_bfast_monitor_batch(
            values_array=values_2d,
            start_time=start_time,
            monitor_start_time=monitor_start_time,
            frequency=frequency,
            order=order,
            h=h,
            period=period,
            alpha=alpha,
            min_valid=min_valid,
            n_jobs=n_jobs,
        )  # (7, pixels)

        return out.reshape(N_BFM_METRICS, rows, cols).astype(np.float32)

    return da.map_blocks(
        _block,
        arr,
        dtype=np.float32,
        drop_axis=[0],
        new_axis=[0],
        chunks=(N_BFM_METRICS, arr.chunks[1], arr.chunks[2]),
    )


def bfl_metric_names(max_breaks_output: int) -> "list[str]":
    """Row names for run_bfast_lite_dask's output, given max_breaks_output."""
    return ["n_breaks", "rss", "lwz", "n_valid", "valid"] + [
        f"breakpoint_idx_{i + 1}" for i in range(max_breaks_output)
    ]


def run_bfast_lite_dask(
    arr: da.Array,
    start_time: float,
    frequency: int,
    order: int = 3,
    h: float = 0.15,
    max_breaks_output: int = 5,
    min_valid: int = 20,
    n_jobs: int = -1,
) -> da.Array:
    """
    Pixel-wise bfastlite (single-pass multiple-breakpoint detection) across
    a Dask array's time axis, ported from the R package `bfast`'s
    `bfastlite()` and its `strucchangeRcpp` dependency's `breakpoints()` -
    the Bai & Perron (2003) optimal multiple-breakpoint dynamic program, via
    Brown-Durbin-Evans recursive residuals for an O(n^2) (rather than O(n^3))
    segment-RSS table - to a C++/OpenMP backend, with the same Dask
    map_blocks strategy as run_bfast_monitor_dask.

    Unlike bfastmonitor (single break/no-break, near-real-time), bfastlite
    retrospectively segments the *whole* series into an optimal number of
    pieces (chosen by minimizing the LWZ - Liu, Wu & Zidek, 1997 -
    model-selection criterion, matching bfastlite's own default
    `breaks="LWZ"`), fitting a trend + harmonic model (same design matrix as
    run_bfast_monitor_dask) within each piece. No STL decomposition is
    needed (unlike the classic iterative `bfast()`, not yet ported).

    Input array shape: (time, y, x), one equally-spaced observation per
    `1/frequency` - same synthetic/regular time convention as
    run_bfast_monitor_dask (matches R's `ts`/`time()` semantics).

    Because the number of breaks varies per pixel, the output caps how many
    breakpoints are reported via `max_breaks_output` (extra slots are
    NaN-padded; pixels needing more are simply not fully described - raise
    `max_breaks_output` if that matters for your data). Output row names:
    `cdts.bfast.bfl_metric_names(max_breaks_output)`.

    start_time: the series' start time (e.g. 2000.0), same convention as
    run_bfast_monitor_dask.

    h: minimum segment size as a fraction of the series length (default
    0.15, matching bfastlite's own default). Unlike run_bfast_monitor_dask's
    `h`, this is a free fraction - no critical-value-table grid restriction.

    max_breaks_output: maximum number of breakpoints to report per pixel
    (also caps the search depth actually attempted, alongside the
    theoretical `ceil(n/h_obs) - 2` bound).

    min_valid: pixels with fewer non-NaN observations than this are
    returned as invalid (`valid=0`, all other metrics NaN).
    """
    def _block(block):
        n_metrics = 5 + max_breaks_output
        if block.size == 0:
            return np.full((n_metrics, block.shape[1], block.shape[2]), np.nan, dtype=np.float32)

        time_steps, rows, cols = block.shape
        pixels = rows * cols

        values_2d = np.ascontiguousarray(block.reshape(time_steps, pixels).T)

        out = fit_bfast_lite_batch(
            values_array=values_2d,
            start_time=start_time,
            frequency=frequency,
            order=order,
            h=h,
            max_breaks_output=max_breaks_output,
            min_valid=min_valid,
            n_jobs=n_jobs,
        )  # (5+max_breaks_output, pixels)

        return out.reshape(n_metrics, rows, cols).astype(np.float32)

    n_metrics = 5 + max_breaks_output
    return da.map_blocks(
        _block,
        arr,
        dtype=np.float32,
        drop_axis=[0],
        new_axis=[0],
        chunks=(n_metrics, arr.chunks[1], arr.chunks[2]),
    )
