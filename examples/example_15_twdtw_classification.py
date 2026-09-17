"""
Example 15: TWDTW End-to-End (Time-Weighted Dynamic Time Warping Classification)

Loads a synthetic single-season NDVI datacube (no network needed) where the
left half of the image follows a "soybean"-like early-peak phenological
curve and the right half a "corn"-like late-peak curve (both with noise and
a random phase jitter, as real fields would have), classifies every pixel
against two reference patterns with `cdts.classify_twdtw`, and saves the
classification + distance maps with `cdts.save_raster`.
"""
import os
import numpy as np
from rasterio.transform import from_origin
import cdts
from cdts.twdtw import classify_twdtw


def double_logistic(doy, sos, eos, peak_amplitude=0.7, base=0.15, steepness=0.1):
    """A simple double-logistic vegetation growth curve peaking between sos/eos."""
    mid = (sos + eos) / 2.0
    green_up = 1.0 / (1.0 + np.exp(-steepness * (doy - sos)))
    green_down = 1.0 / (1.0 + np.exp(steepness * (doy - eos)))
    return base + peak_amplitude * green_up * green_down / (green_up * green_down).max() * (doy < mid + 200)


def build_patterns():
    pattern_dates = np.arange(1, 366, 16)
    soybean = double_logistic(pattern_dates, sos=40, eos=130)
    corn = double_logistic(pattern_dates, sos=110, eos=220)
    return pattern_dates, soybean, corn


def build_synthetic_cube(rows=20, cols=20, seed=8):
    rng = np.random.RandomState(seed)
    dates_array = np.arange(1, 366, 16).astype(np.int32)
    n_time = len(dates_array)
    _, soybean_pattern, corn_pattern = build_patterns()

    values = np.zeros((rows, cols, n_time), dtype=np.float64)
    truth = np.zeros((rows, cols), dtype=np.int32)  # 0 = soybean, 1 = corn
    for y in range(rows):
        for x in range(cols):
            if x < cols // 2:
                base_curve = soybean_pattern
                truth[y, x] = 0
            else:
                base_curve = corn_pattern
                truth[y, x] = 1
            # random phase jitter (a few days) + noise, like real field-to-field variability
            jitter = rng.randint(-8, 9)
            shifted = np.interp(dates_array + jitter, dates_array, base_curve, left=base_curve[0], right=base_curve[-1])
            values[y, x, :] = shifted + rng.normal(0, 0.03, n_time)

    return values, dates_array, truth


def main():
    print("CDTS Example 15: TWDTW (Time-Weighted Dynamic Time Warping) Classification")

    rows, cols = 20, 20
    print(f"\n[1/3] Generating synthetic single-season NDVI cube ({rows}x{cols} px)...")
    values, dates_array, truth = build_synthetic_cube(rows=rows, cols=cols)
    pattern_dates, soybean_pattern, corn_pattern = build_patterns()
    patterns = {
        "soybean": (soybean_pattern, pattern_dates),
        "corn": (corn_pattern, pattern_dates),
    }
    print(f"    {values.shape[2]} composites/season, patterns: {list(patterns.keys())}.")

    print("\n[2/3] Running classify_twdtw() (C++/OpenMP batch TWDTW against both patterns)...")
    classification_map, distance_map, class_names = classify_twdtw(
        values, dates_array, patterns, alpha=0.1, beta=0.05, gamma=50.0, max_time_warp=60, n_jobs=-1,
    )

    accuracy = (classification_map == truth).mean()
    print(f"    Classes (index order): {class_names}")
    print(f"    Agreement with the injected ground truth: {accuracy:.1%}")
    print(f"    Mean TWDTW distance to the winning class: {distance_map.mean():.3f}")

    print("\n[3/3] Saving results with cdts.save_raster()...")
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)

    out_class = os.path.join("data", "twdtw_classification.tif")
    cdts.save_raster(classification_map.astype("uint8"), out_class, crs="EPSG:32721", transform=transform, nodata=255)
    print(f"    Classification map (0={class_names[0]}, 1={class_names[1]}) -> {out_class}")

    out_dist = os.path.join("data", "twdtw_distance.tif")
    cdts.save_raster(distance_map.astype("float32"), out_dist, crs="EPSG:32721", transform=transform, nodata=np.nan)
    print(f"    Winning-class distance map -> {out_dist}")

    print("\nDone!")


if __name__ == "__main__":
    main()
