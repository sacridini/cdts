# TWDTW (Time-Weighted Dynamic Time Warping)

`cdts` features an incredibly optimized C++ implementation of Time-Weighted Dynamic Time Warping (TWDTW), designed specifically for satellite image time series classification.

## Features

- **Blazing Fast**: Written in pure C++ with OpenMP parallelization and SIMD vectorization.
- **Low Memory Footprint**: Uses a 2-row algorithm to keep the dynamic programming matrix strictly inside the CPU L1 Cache.
- **Multivariate (Multi-Band) Support**: Handles multiple spectral bands simultaneously using Eigen's Euclidean norms.
- **Sakoe-Chiba Band**: Constrains the time-warping window to avoid impossible phenological alignments.
- **Lower Bounding (LB Keogh)**: Prunes non-matching pixels in $O(N)$ time before the $O(N \times M)$ DTW matrix is even allocated.
- **Subsequence Matching**: Can search for short crop patterns inside long continuous time series.

## Usage

### Single Time Series

```python
import numpy as np
from cdts.twdtw import run_twdtw

ts_values = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.1])
ts_dates = np.array([1, 45, 90, 135, 180, 225])

pat_values = np.array([0.1, 0.8, 0.1])
pat_dates = np.array([10, 100, 200])

# Compute Distance
distance = run_twdtw(
    ts_values, ts_dates, 
    pat_values, pat_dates,
    alpha=0.1, beta=0.05, gamma=50.0,
    max_time_warp=365
)

# Extract Warping Path
distance, path = run_twdtw(
    ts_values, ts_dates, 
    pat_values, pat_dates,
    return_path=True
)
print("Alignment path:", path)
```

### High-Performance Image Classification

For satellite cubes, use `classify_twdtw`. It automatically utilizes Lower Bounding and Early Abandonment to classify millions of pixels in seconds.

```python
from cdts.twdtw import classify_twdtw

# cube: 3D [Y, X, Time] or 4D [Y, X, Time, Bands]
# dates: 1D array of acquisition days
patterns = {
    "Forest": (forest_sig, forest_dates),
    "Soy": (soy_sig, soy_dates)
}

classes_map, dist_map, class_names = classify_twdtw(cube, dates, patterns, n_jobs=-1)
```

### Multivariate TWDTW

Just pass a 2D array `(Time, Bands)` for a single series, or a 4D array `(Y, X, Time, Bands)` for a batch. The C++ engine automatically switches to Eigen SIMD Euclidean distance for spatial mapping!
