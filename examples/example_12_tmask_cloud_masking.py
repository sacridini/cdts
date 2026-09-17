"""
Example 12: Tmask End-to-End (Time-Series-Based Cloud/Shadow Masking)

Loads a synthetic Green/SWIR1 datacube (no network needed) with randomly
injected cloud (high Green) and cloud-shadow (low SWIR) contamination, runs
the Tmask robust time-series masking algorithm (Zhu & Woodcock, 2014), and
saves the resulting per-observation clear/cloudy mask stack with
`cdts.save_raster`.
"""
import os
import numpy as np
from rasterio.transform import from_origin
import cdts


def build_synthetic_stack(n_obs=140, rows=8, cols=8, seed=4):
    """
    Reflectance is scaled 0-10000 (standard Landsat/Sentinel convention).
    Baseline Green/SWIR1 follow a smooth seasonal cycle; ~15% of dates get a
    cloud (Green spike) and another ~10% get a cloud-shadow (SWIR dip),
    each affecting a random contiguous patch of the image (like a real cloud
    footprint) rather than every pixel at once.
    """
    rng = np.random.RandomState(seed)
    dates_julian = np.array([(i * 8) % 365 + 1 + (i * 8 // 365) * 365 for i in range(n_obs)])
    w = 2.0 * np.pi / 365.25

    green = 900 + 200 * np.cos(w * dates_julian)[:, None, None] + rng.normal(0, 40, (n_obs, rows, cols))
    swir1 = 1800 + 300 * np.sin(w * dates_julian)[:, None, None] + rng.normal(0, 60, (n_obs, rows, cols))

    is_cloud = np.zeros(n_obs, dtype=bool)
    is_shadow = np.zeros(n_obs, dtype=bool)
    for i in range(n_obs):
        r = rng.rand()
        if r < 0.15:
            is_cloud[i] = True
            y0, x0 = rng.randint(0, rows // 2), rng.randint(0, cols // 2)
            green[i, y0:y0 + rows // 2, x0:x0 + cols // 2] += 3000  # bright cloud
        elif r < 0.25:
            is_shadow[i] = True
            y0, x0 = rng.randint(0, rows // 2), rng.randint(0, cols // 2)
            swir1[i, y0:y0 + rows // 2, x0:x0 + cols // 2] -= 1200  # dark shadow

    green = np.clip(green, 50, None)
    swir1 = np.clip(swir1, 50, None)
    return dates_julian, green, swir1, is_cloud, is_shadow


def main():
    print("CDTS Example 12: Tmask (Time-Series-Based Cloud/Shadow Masking)")

    rows, cols = 8, 8
    print(f"\n[1/3] Generating synthetic Green/SWIR1 stack ({rows}x{cols} px)...")
    dates_julian, green, swir1, is_cloud, is_shadow = build_synthetic_stack(rows=rows, cols=cols)
    print(f"    {green.shape[0]} observations, {int(is_cloud.sum())} cloud-contaminated dates, "
          f"{int(is_shadow.sum())} shadow-contaminated dates injected.")

    print("\n[2/3] Running Tmask (per-pixel robust harmonic regression)...")
    clear_mask = cdts.apply_tmask_stack(dates_julian, green, swir1, scale_factor=10000.0)

    clear_fraction = clear_mask.mean(axis=0)
    print(f"    Mean fraction of observations flagged clear per pixel: {clear_fraction.mean():.2%} "
          f"(injected contamination rate: {(is_cloud | is_shadow).mean():.2%}).")

    print("\n[3/3] Saving results with cdts.save_raster()...")
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)

    out_mask = os.path.join("data", "tmask_clear_mask_stack.tif")
    cdts.save_raster(clear_mask.astype("uint8"), out_mask, crs="EPSG:32721", transform=transform, nodata=255)
    print(f"    Full clear/cloudy mask stack ({clear_mask.shape[0]} bands, 1=clear/0=cloudy) -> {out_mask}")

    out_fraction = os.path.join("data", "tmask_clear_fraction.tif")
    cdts.save_raster(clear_fraction.astype("float32"), out_fraction, crs="EPSG:32721", transform=transform, nodata=np.nan)
    print(f"    Per-pixel clear-observation fraction -> {out_fraction}")

    print("\nDone!")


if __name__ == "__main__":
    main()
