"""
Example 10: BFAST Lite End-to-End (Single-Pass Multiple-Breakpoint Detection)

Loads a synthetic 16-day composite datacube (no network needed) where every
pixel has two injected structural breaks (e.g. a disturbance followed by
recovery/regrowth), runs the single-pass `bfastlite` segmentation via the
`.cdts` accessor, and saves the breakpoint maps with `cdts.save_raster`.
"""
import os
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
import cdts
from cdts.bfast import bfl_metric_names

FREQUENCY = 23
START_TIME = 2010.0
MAX_BREAKS = 5


def build_synthetic_cube(n_years=8, rows=15, cols=15, seed=2):
    """
    Trend + harmonic series with two injected breaks per pixel: a drop
    (disturbance) around 1/3 of the series, and a partial recovery around
    2/3 - a classic disturbance-then-regrowth trajectory.
    """
    rng = np.random.RandomState(seed)
    n_time = n_years * FREQUENCY
    t = np.arange(n_time)
    break1 = n_time // 3
    break2 = (2 * n_time) // 3

    data = np.zeros((n_time, rows, cols), dtype=np.float64)
    for y in range(rows):
        for x in range(cols):
            base = 0.6 + rng.normal(0, 0.02)
            season = 0.1 * np.cos(2 * np.pi * t / FREQUENCY) + 0.05 * np.sin(2 * np.pi * t / FREQUENCY)
            series = base + season
            series[break1:] -= 0.3   # disturbance
            series[break2:] += 0.15  # partial recovery
            data[:, y, x] = series + rng.normal(0, 0.02, n_time)

    return data, break1, break2


def main():
    print("CDTS Example 10: BFAST Lite (Single-Pass Multiple-Breakpoint Detection)")

    rows, cols = 15, 15
    print(f"\n[1/3] Generating synthetic cube ({rows}x{cols} px) with 2 injected breaks per pixel...")
    data, break1, break2 = build_synthetic_cube(rows=rows, cols=cols)
    print(f"    Injected breaks at observation indices {break1} (disturbance) and {break2} (recovery).")

    cube = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"y": np.arange(rows), "x": np.arange(cols)},
    )

    print("\n[2/3] Running bfastlite via .cdts accessor...")
    result = cube.cdts.run_bfast_lite(
        start_time=START_TIME, frequency=FREQUENCY, h=0.15, max_breaks_output=MAX_BREAKS, n_jobs=-1,
    ).compute()

    n_breaks = result.sel(metric="n_breaks").values
    print(f"    Mean number of breaks detected per pixel: {np.nanmean(n_breaks):.2f} (injected: 2).")
    bp1 = result.sel(metric="breakpoint_idx_1").values
    bp2 = result.sel(metric="breakpoint_idx_2").values
    print(f"    Median 1st breakpoint: {np.nanmedian(bp1):.1f} (injected {break1}); "
          f"median 2nd breakpoint: {np.nanmedian(bp2):.1f} (injected {break2}).")

    print("\n[3/3] Saving results with cdts.save_raster()...")
    out_tif = os.path.join("data", "bfast_lite_breaks.tif")
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    cdts.save_raster(result.values.astype("float32"), out_tif, crs="EPSG:32721", transform=transform, nodata=np.nan)

    names = bfl_metric_names(MAX_BREAKS)
    print(f"\nDone! {len(names)}-band raster ({', '.join(names)}) saved to {out_tif}")


if __name__ == "__main__":
    main()
