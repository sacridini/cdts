"""
Example 13: Temporal Regularization End-to-End (Irregular ARD -> Regular Composites)

Loads a synthetic multi-band datacube with irregular acquisition dates (as
real satellite archives have, due to revisit gaps and cloud cover - no
network needed), regularizes it to fixed 16-day composites using both the
`median` and `medoid` strategies via `cdts.regularize_time_series`, and
saves the resulting composites with `cdts.save_raster`.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_origin
import cdts


def build_irregular_cube(rows=20, cols=20, seed=5):
    """
    Simulates ~1 year of Sentinel-2-like acquisitions with an irregular
    revisit interval (3-7 days, mimicking swath overlap + dropped cloudy
    scenes) and two bands (NIR, Red).
    """
    rng = np.random.RandomState(seed)
    dates = [pd.Timestamp("2022-01-01")]
    while dates[-1] < pd.Timestamp("2022-12-25"):
        dates.append(dates[-1] + pd.Timedelta(days=int(rng.randint(3, 8))))
    dates = pd.DatetimeIndex(dates)
    n_time = len(dates)

    doy = dates.dayofyear.values
    season = 0.3 * np.sin((doy / 365.0) * 2 * np.pi)
    nir = (0.5 + season)[:, None, None] + rng.normal(0, 0.03, (n_time, rows, cols))
    red = (0.15 - 0.3 * season)[:, None, None] + rng.normal(0, 0.02, (n_time, rows, cols))
    # A handful of dates are corrupted with cloud noise (large outliers) -
    # this is exactly what the `medoid` strategy (an *actual* observation,
    # picked for being closest to the group's median) is more robust to
    # than a per-band `median` (which can average in bad pixels band-by-band).
    n_outliers = max(1, n_time // 10)
    outlier_idx = rng.choice(n_time, size=n_outliers, replace=False)
    nir[outlier_idx] += 0.6
    red[outlier_idx] += 0.6

    data = np.stack([nir, red], axis=1)  # (time, band, y, x)
    cube = xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={"time": dates, "band": ["nir", "red"], "y": np.arange(rows), "x": np.arange(cols)},
    )
    return cube, n_outliers


def main():
    print("CDTS Example 13: Temporal Regularization (Irregular ARD -> Regular Composites)")

    rows, cols = 20, 20
    print(f"\n[1/3] Generating synthetic irregular-date cube ({rows}x{cols} px, 2 bands)...")
    cube, n_outliers = build_irregular_cube(rows=rows, cols=cols)
    print(f"    {cube.sizes['time']} irregular acquisitions between "
          f"{str(cube.time.values[0])[:10]} and {str(cube.time.values[-1])[:10]}, "
          f"{n_outliers} cloud-corrupted dates injected.")

    print("\n[2/3] Regularizing to 16-day composites (median and medoid)...")
    median_cube = cdts.regularize_time_series(cube, freq="16D", method="median").compute()
    medoid_cube = cdts.regularize_time_series(cube, freq="16D", method="medoid").compute()
    print(f"    Regularized from {cube.sizes['time']} irregular dates down to "
          f"{median_cube.sizes['time']} regular 16-day composites.")

    print("\n[3/3] Saving results with cdts.save_raster()...")
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)

    out_median = os.path.join("data", "regularized_median.tif")
    cdts.save_raster(median_cube.values.astype("float32"), out_median, crs="EPSG:32721", transform=transform, nodata=np.nan)
    print(f"    Median composites ({median_cube.sizes['time']} dates x {median_cube.sizes['band']} bands) -> {out_median}")

    out_medoid = os.path.join("data", "regularized_medoid.tif")
    cdts.save_raster(medoid_cube.values.astype("float32"), out_medoid, crs="EPSG:32721", transform=transform, nodata=np.nan)
    print(f"    Medoid composites ({medoid_cube.sizes['time']} dates x {medoid_cube.sizes['band']} bands) -> {out_medoid}")

    print("\nDone!")


if __name__ == "__main__":
    main()
