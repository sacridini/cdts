"""
Example 19: LandTrendr Accuracy Validation Dashboard End-to-End

Loads a synthetic multi-year, multi-band datacube and a matching synthetic
LandTrendr "year of disturbance" result (no network needed), and generates a
standalone, serverless HTML validation dashboard with
`cdts.generate_landtrendr_accuracy_dashboard` - a true-color image chip +
temporal trajectory plot per validation point, letting an analyst manually
confirm/correct each predicted disturbance year in a browser, with no
server required. This is CDTS's one "output is HTML, not a raster" tool, so
there's no `cdts.save_raster` call here - the dashboard function writes its
own self-contained `.html` file directly.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import cdts


def build_synthetic_cube(n_years=8, rows=60, cols=60, seed=14):
    """
    A tiny fake Sentinel-2-like cube: bands B04/B03/B02 (true color) and NBR
    (a common disturbance index). A rectangular patch is clear-cut halfway
    through the series (NBR drops, RGB brightens - a classic disturbance
    signature) so the dashboard has something real to show.
    """
    rng = np.random.RandomState(seed)
    lon = np.linspace(-55.05, -55.00, cols)
    lat = np.linspace(-11.00, -10.95, rows)[::-1]
    years = pd.date_range("2016-07-01", periods=n_years, freq="YS-JUL")

    disturbance_year_idx = n_years // 2
    disturbed = np.zeros((rows, cols), dtype=bool)
    disturbed[20:40, 15:35] = True

    bands = ["B04", "B03", "B02", "NBR"]
    data = np.zeros((n_years, len(bands), rows, cols), dtype=np.float32)
    for t in range(n_years):
        healthy_rgb = np.array([0.06, 0.09, 0.05]) + rng.normal(0, 0.01, 3)
        cut_rgb = np.array([0.25, 0.23, 0.18]) + rng.normal(0, 0.01, 3)
        for b in range(3):
            data[t, b] = healthy_rgb[b]
            if t >= disturbance_year_idx:
                data[t, b][disturbed] = cut_rgb[b]
        nbr = np.full((rows, cols), 0.55, dtype=np.float32)
        if t >= disturbance_year_idx:
            nbr[disturbed] = 0.05
        data[t, 3] = nbr + rng.normal(0, 0.02, (rows, cols)).astype(np.float32)

    cube = xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={"time": years, "band": bands, "y": lat, "x": lon},
    )
    return cube, years[disturbance_year_idx].year, disturbed


def build_fake_lt_results(cube, disturbance_year, disturbed_mask):
    """A fake `extract_events`-style result: just a 'yod' band, correct
    inside the disturbed patch and 0 (no disturbance) elsewhere."""
    yod = np.where(disturbed_mask, disturbance_year, 0).astype(np.float32)
    return xr.DataArray(
        yod[None, :, :],
        dims=["band", "y", "x"],
        coords={"band": ["yod"], "y": cube.y, "x": cube.x},
    )


def main():
    print("CDTS Example 19: LandTrendr Accuracy Validation Dashboard")

    print("\n[1/3] Generating a synthetic multi-year cube with an injected clear-cut...")
    cube, disturbance_year, disturbed_mask = build_synthetic_cube()
    print(f"    {cube.sizes['time']} annual composites, disturbance injected in {disturbance_year} "
          f"over a {disturbed_mask.sum()}-pixel patch.")

    lt_results = build_fake_lt_results(cube, disturbance_year, disturbed_mask)

    # 3 validation points: 2 inside the disturbed patch, 1 clearly outside it.
    points = [
        (float(cube.x[20]), float(cube.y[25])),  # inside the disturbance patch
        (float(cube.x[25]), float(cube.y[30])),  # inside the disturbance patch
        (float(cube.x[50]), float(cube.y[5])),   # outside - stable forest
    ]
    print(f"\n[2/3] Preparing {len(points)} validation points (2 inside the disturbance, 1 outside)...")

    print("\n[3/3] Generating the standalone HTML dashboard with cdts.generate_landtrendr_accuracy_dashboard()...")
    out_html = os.path.join("data", "lt_accuracy_dashboard.html")
    cdts.generate_landtrendr_accuracy_dashboard(
        cube=cube, points=points, lt_results=lt_results, output_html=out_html, window_size=15,
    )

    print(f"\nDone! Open {out_html} in a browser to inspect/validate each point.")


if __name__ == "__main__":
    main()
