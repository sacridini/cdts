# LandTrendr in CDTS: End-to-End Tutorial

LandTrendr (Landsat-based detection of Trends in Disturbance and Recovery) is a trajectory-based algorithm designed to identify both abrupt and gradual changes in time series of pixel values. This tutorial will guide you through preparing your data, executing the algorithm, extracting meaningful metrics, and interpreting the results using the Continuous Disturbance Time Series (`cdts`) package.

## 1. Preparing the Input Data

To run LandTrendr across multiple pixels, the `cdts` package requires your input time series data to be formatted as standard NumPy arrays.

You will need:
- A 1D array of **years** (or time steps).
- A 3D array of **spectral values** (or indices, like NBR, NDVI) with the shape `(time_steps, rows, cols)`.

```python
import numpy as np

# Example: 10 years of data (2000-2009)
years = np.arange(2000, 2010)

# Simulate a 3D raster stack (10 time steps, 100 rows, 100 columns)
# For LandTrendr, vegetation indices are often inverted so disturbance = increase in value, 
# or kept as is if tracking loss.
rows, cols = 100, 100
raster_stack = np.random.uniform(0.5, 0.9, size=(10, rows, cols))

# Inject a disturbance event at year 2003 (index 3) for a specific pixel
raster_stack[3:, 50, 50] = raster_stack[3:, 50, 50] - 0.4
```

> **Note:** Completely empty pixels or pixels containing only NoData (`np.nan` or `0`) will be automatically skipped by the internal engine to optimize processing time.

## 2. Running LandTrendr on an Array

To process a 3D numpy array, use the `run_landtrendr_array` function from `cdts.raster`. This function utilizes Python's multiprocessing to efficiently process pixels in parallel.

```python
from cdts.raster import run_landtrendr_array

# Execute the LandTrendr algorithm
vertices_stack = run_landtrendr_array(
    years=years,
    raster_stack=raster_stack,
    max_segments=6,
    pval_threshold=0.05,
    n_jobs=-1  # Use -1 to use all available CPU cores
)
```

### Understanding the Parameters:
- `years` (np.ndarray): The 1D array representing the temporal axis.
- `raster_stack` (np.ndarray): The 3D data array `(time_steps, rows, cols)`.
- `max_segments` (int): The maximum number of line segments to fit to the time series. Default is `6`.
- `pval_threshold` (float): The p-value threshold used for determining if a fitted segment is statistically significant. Default is `0.05`.
- `n_jobs` (int): Number of parallel workers. `-1` uses all CPU cores.

### Interpreting the Output (`vertices_stack`):
The output is a 3D numpy array of shape `(2 * max_vertices, rows, cols)`, where `max_vertices = max_segments + 1`. 
- **The first half** of the first dimension contains the **Years** of the identified vertices.
- **The second half** contains the **Fitted Values** for those vertices.

## 3. Extracting and Interpreting Events

Once you have the fitted vertices, you often want to extract specific events (e.g., the greatest vegetation loss, the fastest recovery). The `cdts.metrics.extract_events` function parses the `vertices_stack` and extracts a 2D map of spatial metrics.

```python
from cdts.metrics import extract_events

events = extract_events(
    vertices_stack=vertices_stack,
    event_type="loss",           # "loss" (value decreases) or "gain" (value increases)
    sort_by="greatest",          # "greatest" (magnitude), "newest" (year), "fastest" (rate), "longest" (duration)
    min_magnitude=0.1,           # Filter out minor changes
    min_duration=1,              # Minimum duration in years
    pre_val_threshold=0.5        # Exclude pixels that were already degraded before the event
)
```

### Event Metrics:
The `extract_events` function returns a dictionary of 2D numpy arrays `(rows, cols)`:
- `yod`: Year of Detection (the starting year of the segment).
- `magnitude`: The absolute change in value over the segment.
- `duration`: The length of the segment in years.
- `pre_val`: The fitted value *before* the event started.
- `post_val`: The fitted value *after* the event ended.
- `rate`: Magnitude divided by duration (change per year).

```python
# Access the Year of Detection (YOD) map
yod_map = events["yod"]

# Check the year the disturbance happened at our injected pixel
print(f"Disturbance year: {yod_map[50, 50]}") 
```

## 4. Best Practices

1. **Despike your data first**: The `cdts` library provides a `desawtooth` function that removes temporal spikes (e.g., from unmasked clouds) which can heavily skew LandTrendr trajectory fitting. Consider applying this before running the algorithm.
2. **Chunking for Large Rasters**: A full GeoTIFF stack might exceed your RAM. Use `cdts.raster.run_landtrendr_image` to process large GeoTIFF files natively in spatial chunks rather than loading the entire array into memory.
3. **Choosing `max_segments`**: A `max_segments` value of 6 is standard. Increasing it can lead to overfitting noise, while decreasing it might cause the algorithm to miss subtle, gradual trends (like slow degradation or long-term recovery).
