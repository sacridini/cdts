# LandTrendr in CDTS: End-to-End Tutorial

## 1. Introduction to LandTrendr

**LandTrendr** (Landsat-based detection of Trends in Disturbance and Recovery) is a highly influential trajectory-based algorithm designed to extract both abrupt changes (like deforestation or fire) and gradual changes (like forest degradation, disease, or recovery) from annual satellite imagery.

### Background and References
Originally developed by Robert Kennedy et al., LandTrendr works by reducing complex, noisy, annual time series data into a sequence of simplified straight-line segments. It minimizes the residual error between the actual satellite observations and the simplified straight-line model, essentially "filtering out" inter-annual noise (like slight phenological differences or minor atmospheric effects) to reveal the true underlying landscape dynamics.

- **Original Paper**: [Kennedy, R.E., Yang, Z. and Cohen, W.B., 2010. Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr—Temporal segmentation algorithms. Remote Sensing of Environment, 114(12), pp.2897-2910.](https://doi.org/10.1016/j.rse.2010.07.008)
- **Google Earth Engine Implementation**: The algorithm is also natively available in GEE. Learn more at the [eMapR Lab GitHub](https://github.com/eMapR/LT-GEE).

By integrating LandTrendr into **CDTS**, you gain the ability to run this powerful algorithm locally, on HPC clusters, or natively on massive GeoTIFF stacks without being constrained by cloud-platform quotas.

---

## 2. End-to-End Workflow

This tutorial will guide you through an end-to-end process: loading a real multi-band GeoTIFF, running the C++ optimized LandTrendr algorithm in Python, extracting change metrics, and visualizing the results.

### Step 2.1: Loading the Data

LandTrendr requires an annual time series of a single spectral index. The Normalized Burn Ratio (NBR) is the most commonly used index for forest disturbance because it is highly sensitive to canopy removal and moisture loss.

In LandTrendr conventions, indices are often inverted (e.g., `-NBR` or `10000 - NBR`) so that a **disturbance** is represented by a **positive increase** in value.

```python
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# 1. Define the years corresponding to our data stack
# Let's assume we have a 30-year Landsat stack from 1990 to 2019
start_year = 1990
end_year = 2019
years = np.arange(start_year, end_year + 1)

# 2. Load the GeoTIFF using rasterio
input_path = "data/annual_nbr_stack_1990_2019.tif"
with rasterio.open(input_path) as src:
    # Read the entire 3D array (Bands, Rows, Cols)
    raster_stack = src.read()
    profile = src.profile

print(f"Loaded stack with shape: {raster_stack.shape}")
# Example Output: Loaded stack with shape: (30, 2000, 2000)
```

### Step 2.2: Despiking (Optional but Recommended)

Satellite data often contains residual noise (e.g., missed cloud shadows) that appear as sharp, one-year dips in the time series. If not removed, LandTrendr might fit false segments to these spikes. The `cdts.smooth` module provides a despiking function to handle this.

```python
from cdts.smooth import desawtooth

# Despike the time series to remove 1-year anomalous drops
# This is a critical pre-processing step for LandTrendr
smoothed_stack = desawtooth(raster_stack)
```

### Step 2.3: Running the Algorithm

We use the `run_landtrendr_array` function to apply the segmentation logic across all pixels in parallel.

```python
from cdts.raster import run_landtrendr_array

print("Running LandTrendr segmentation...")
vertices_stack = run_landtrendr_array(
    years=years,
    raster_stack=smoothed_stack,
    max_segments=6,        # Allow up to 6 distinct trend segments
    pval_threshold=0.05,   # Statistical significance threshold for segments
    n_jobs=-1              # Distribute work across all CPU cores
)
print("Segmentation complete!")
```

**Understanding the Output (`vertices_stack`)**:
The output is a 3D numpy array. If `max_segments=6`, the maximum number of vertices is 7. The output will have `14` bands (2 * 7).
- **Bands 0 to 6**: The *Years* of the identified vertices.
- **Bands 7 to 13**: The *Fitted Values* corresponding to those years.

### Step 2.4: Extracting Disturbance Events

Now that we have the simplified trajectories for every pixel, we want to extract the greatest disturbance event.

```python
from cdts.metrics import extract_events

# Extract the greatest loss event (disturbance)
events = extract_events(
    vertices_stack=vertices_stack,
    event_type="loss",           # "loss" corresponds to a drop in the original NBR
    sort_by="greatest",          # Select the segment with the largest absolute change
    min_magnitude=150.0,         # Minimum change magnitude to be considered an event
    min_duration=1               # Minimum duration (1 = abrupt change, >1 = gradual)
)

# The result is a dictionary of 2D maps
yod_map = events["yod"]           # Year of Detection
magnitude_map = events["magnitude"] # Change Magnitude
duration_map = events["duration"]   # Change Duration
```

### Step 2.5: Exporting Results

You can export these 2D metrics back to GeoTIFFs using `rasterio` so they can be viewed in QGIS or ArcGIS.

```python
# Update the rasterio profile for a single-band 2D output
out_profile = profile.copy()
out_profile.update(count=1, dtype='uint16', nodata=0)

# Save Year of Detection
with rasterio.open("results/lt_yod.tif", "w", **out_profile) as dst:
    dst.write(yod_map, 1)

# Save Magnitude (updating profile to float32)
out_profile.update(dtype='float32')
with rasterio.open("results/lt_magnitude.tif", "w", **out_profile) as dst:
    dst.write(magnitude_map, 1)
```

## 3. Visualizing a Single Pixel Trajectory

To truly understand LandTrendr, it helps to plot the original data alongside the fitted vertices for a single pixel.

```python
import matplotlib.pyplot as plt

# Pick a pixel that experienced a disturbance
row, col = 500, 500

# 1. Get original time series
original_ts = smoothed_stack[:, row, col]

# 2. Get the fitted vertices for this pixel
pixel_vertices = vertices_stack[:, row, col]

# The first half of the array are the years, the second half are the values
max_v = vertices_stack.shape[0] // 2
vertex_years = pixel_vertices[:max_v]
vertex_values = pixel_vertices[max_v:]

# Filter out empty vertices (where year == 0)
valid_idx = vertex_years > 0
vertex_years = vertex_years[valid_idx]
vertex_values = vertex_values[valid_idx]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(years, original_ts, marker='o', label='Original Data (Smoothed)', color='gray', linestyle='--')
plt.plot(vertex_years, vertex_values, marker='s', label='LandTrendr Fitted Trajectory', color='red', linewidth=2)

plt.title("LandTrendr Pixel Trajectory")
plt.xlabel("Year")
plt.ylabel("Spectral Index Value")
plt.legend()
plt.grid(True)
plt.show()
```

## 4. Best Practices and Tips

1. **Index Selection**: While NBR is the standard for forest disturbance, Tasseled Cap Wetness (TCW) or Tasseled Cap Angle (TCA) are extremely effective. NDVI is generally less sensitive to structural forest changes but good for agricultural monitoring.
2. **Out-of-Core Processing**: If your input GeoTIFF is larger than your available RAM, use `cdts.raster.run_landtrendr_image` instead of `run_landtrendr_array`. The image-based function automatically chunks the raster and processes it in blocks, keeping memory usage strictly bounded.
3. **Overfitting**: A `max_segments` value of 6 is empirically proven to be optimal for a 30-year time series. Increasing it to 8 or 10 on a 30-year stack will lead to the algorithm overfitting noise, resulting in false positive disturbances.
