"""
Example 08: Mann-Kendall / Theil-Sen Trend Detection End-to-End

Loads a synthetic annual NDVI datacube (no network needed), runs the
pixel-wise Mann-Kendall trend test + Theil-Sen slope estimator via the
`.cdts` xarray accessor, and saves the resulting trend/slope/p-value maps
as a multi-band GeoTIFF with `cdts.save_raster`.
"""
import os
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
import cdts
from cdts.trend import MK_METRIC_NAMES


def build_synthetic_ndvi_cube(n_years=20, rows=40, cols=40, seed=0):
    """
    Fake annual NDVI composites (time, y, x): the left half of the image
    greens up over time (a real trend), the right half only has noise
    around a constant mean (no trend) - a simple ground truth to check the
    trend test against.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_years)
    data = np.zeros((n_years, rows, cols), dtype=np.float64)

    greening_slope = 0.01  # NDVI units per year
    for y in range(rows):
        for x in range(cols):
            base = 0.4 + rng.normal(0, 0.02)
            if x < cols // 2:
                series = base + greening_slope * t
            else:
                series = np.full(n_years, base)
            data[:, y, x] = series + rng.normal(0, 0.015, n_years)

    return data


def main():
    print("CDTS Example 08: Mann-Kendall / Theil-Sen Trend Detection")

    n_years, rows, cols = 20, 40, 40
    start_year = 2004
    print(f"\n[1/3] Generating synthetic annual NDVI cube ({n_years} years, {rows}x{cols} px)...")
    data = build_synthetic_ndvi_cube(n_years=n_years, rows=rows, cols=cols)

    cube = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={
            "time": np.arange(start_year, start_year + n_years),
            "y": np.arange(rows),
            "x": np.arange(cols),
        },
    )

    print("\n[2/3] Running Mann-Kendall (hamed_rao, autocorrelation-corrected) via .cdts accessor...")
    # One NDVI composite per year, so slope comes out directly in NDVI/year.
    result = cube.cdts.run_mann_kendall(method="hamed_rao", alpha=0.05, n_jobs=-1).compute()

    trend = result.sel(metric="trend").values
    slope = result.sel(metric="slope").values
    n_greening = int(np.sum(trend == 1))
    n_no_trend = int(np.sum(trend == 0))
    print(f"    Detected 'increasing' trend in {n_greening} pixels (expected ~{rows * cols // 2}).")
    print(f"    Detected 'no trend' in {n_no_trend} pixels.")
    print(f"    Mean slope on the greening half: {np.nanmean(slope[:, :cols // 2]):.4f} NDVI/year "
          f"(injected: 0.01).")

    print("\n[3/3] Saving results with cdts.save_raster()...")
    out_tif = os.path.join("data", "mann_kendall_trend.tif")
    # Fake a 30m-resolution geotransform (e.g. UTM-like), since this data has
    # no real-world footprint - see MK_METRIC_NAMES for the band order.
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    cdts.save_raster(result.values.astype("float32"), out_tif, crs="EPSG:32721", transform=transform, nodata=np.nan)

    print(f"\nDone! {len(MK_METRIC_NAMES)}-band raster ({', '.join(MK_METRIC_NAMES)}) saved to {out_tif}")


if __name__ == "__main__":
    main()
