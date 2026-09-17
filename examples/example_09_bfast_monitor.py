"""
Example 09: BFAST Monitor End-to-End (Near-Real-Time Disturbance Monitoring)

Loads a synthetic 16-day composite datacube (no network needed) with a
stable history period and, for half the pixels, an abrupt disturbance
injected during the monitoring period. Runs `bfastmonitor` via the `.cdts`
accessor and saves the breakpoint/magnitude/sigma maps with
`cdts.save_raster`.
"""
import os
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
import cdts
from cdts.bfast import BFM_METRIC_NAMES

FREQUENCY = 23  # 16-day composites/year
START_TIME = 2015.0
MONITOR_START_TIME = 2019.0  # 4 years of history, then start monitoring


def build_synthetic_cube(n_years=6, rows=20, cols=20, seed=1):
    """
    Synthetic NDVI-like series: intercept + trend + one harmonic seasonal
    cycle, stable throughout history. Half the pixels (x >= cols/2) get an
    abrupt drop injected partway through the monitoring period (e.g. a
    clear-cut), the other half stays stable end to end.
    """
    rng = np.random.RandomState(seed)
    n_time = n_years * FREQUENCY
    t = np.arange(n_time)
    time_years = START_TIME + t / FREQUENCY

    disturbance_row = np.searchsorted(time_years, MONITOR_START_TIME + 0.5)  # ~mid-monitoring

    data = np.zeros((n_time, rows, cols), dtype=np.float64)
    for y in range(rows):
        for x in range(cols):
            base = 0.55 + rng.normal(0, 0.02)
            season = 0.12 * np.cos(2 * np.pi * t / FREQUENCY) + 0.06 * np.sin(2 * np.pi * t / FREQUENCY)
            series = base + season
            if x >= cols // 2:
                series = series.copy()
                series[disturbance_row:] -= 0.35  # abrupt disturbance (e.g. deforestation)
            data[:, y, x] = series + rng.normal(0, 0.015, n_time)

    return data, time_years, disturbance_row


def main():
    print("CDTS Example 09: BFAST Monitor (Near-Real-Time Monitoring)")

    rows, cols = 20, 20
    print(f"\n[1/3] Generating synthetic 16-day composite cube ({rows}x{cols} px, {FREQUENCY} obs/year)...")
    data, time_years, disturbance_row = build_synthetic_cube(rows=rows, cols=cols)
    print(f"    Disturbance injected at fractional year {time_years[disturbance_row]:.3f} "
          f"(observation index {disturbance_row}) for x >= {cols // 2}.")

    cube = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": time_years, "y": np.arange(rows), "x": np.arange(cols)},
    )

    print("\n[2/3] Running bfastmonitor via .cdts accessor...")
    result = cube.cdts.run_bfast_monitor(
        start_time=START_TIME,
        monitor_start_time=MONITOR_START_TIME,
        frequency=FREQUENCY,
        h=0.25,
        period=10,
        alpha=0.05,
        n_jobs=-1,
    ).compute()

    has_break = result.sel(metric="has_break").values
    print(f"    Breaks detected: {int(np.sum(has_break == 1))} pixels "
          f"(expected ~{rows * cols // 2}, the disturbed half).")
    breakpoint_idx = result.sel(metric="breakpoint_idx").values
    print(f"    Median detected breakpoint index on the disturbed half: "
          f"{np.nanmedian(breakpoint_idx[:, cols // 2:]):.1f} (injected at {disturbance_row}).")

    print("\n[3/3] Saving results with cdts.save_raster()...")
    out_tif = os.path.join("data", "bfast_monitor_breaks.tif")
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    cdts.save_raster(result.values.astype("float32"), out_tif, crs="EPSG:32721", transform=transform, nodata=np.nan)

    print(f"\nDone! {len(BFM_METRIC_NAMES)}-band raster ({', '.join(BFM_METRIC_NAMES)}) saved to {out_tif}")


if __name__ == "__main__":
    main()
