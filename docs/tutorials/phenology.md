# Phenology Extraction

The `cdts` package includes a high-performance C++ backend for **Phenology Extraction**, designed to compute vegetation metrics (such as Start of Season, End of Season, Length of Season, and Peak of Season) from dense satellite time series.

This module is heavily inspired by state-of-the-art tools (like R's `phenofit`) but is built entirely in C++ with OpenMP and Eigen, ensuring that gigabytes of rasters can be processed efficiently in Python.

## Core Features

- **Double-Logistic Curve Fitting**: Native Levenberg-Marquardt optimizer for fitting various double-logistic and asymmetric Gaussian curves.
- **Supported Equations**:
  - `BECK` (Beck et al., 2006)
  - `ELMORE` (Elmore et al., 2012)
  - `GU` (Gu et al., 2009)
  - `KLOSTERMAN` (Klosterman et al., 2014)
  - `ZHANG` (Zhang et al., 2003)
  - `AG` (Asymmetric Gaussian)
  - `DL` (Standard Double Logistic)
- **Extraction Methods**:
  - `THRESHOLD` (Amplitude ratio thresholding)
  - `DERIVATIVE` (Points of maximum curvature/derivative)
- **Time Series Smoothing**: 
  - Whittaker Smoother (Sparse matrices)
  - HANTS (Harmonic Analysis of Time Series)
- **Robustness**: Automatically bypasses non-vegetated areas, missing data, and invalid flat curves.

## Basic Usage

To extract phenology from an `xarray` dataset or an in-memory numpy array, you can use the `cdts.phenology` module. The core execution runs entirely in C++, parallelized across CPU cores.

```python
import numpy as np
import pandas as pd
import cdts._core as core

# Imagine a vegetation index (e.g., EVI) time series of shape (n_pixels, time_steps)
values_array = np.random.rand(1000, 500) # 1000 pixels, 500 time steps
dates = pd.date_range(start='2001-01-01', periods=500, freq='16D')

# Convert dates to continuous DOY (Day of Year) relative to a base year
dates_doy = np.array([d.timetuple().tm_yday + (d.year - 2001) * 365 for d in dates], dtype=np.float64)

# Fit curves and extract metrics
sos, eos, los, pop = core.phenology.fit_phenology_batch(
    values_array=values_array,
    dates_array=dates_doy,
    curve_type=1,           # 0=WHITTAKER, 1=BECK, 2=ELMORE, etc.
    extraction_method=2,    # 1=THRESHOLD, 2=DERIVATIVE
    max_seasons=30,         # Maximum expected seasons per pixel across the whole series
    whittaker_lambda=2.0,   # Smoothing parameter
    apply_whittaker=True,   # Apply fast Whittaker smoother first
    apply_hants=False,      # Option to apply Harmonic Analysis
    min_season_length=3,    # Filter out seasons shorter than 3 days
    min_amplitude=0.01,     # Filter out seasons with very low amplitude
    min_pixel_amplitude=0.01, # Skip pixels with no variation
    n_jobs=-1               # Use all CPU cores
)

print("Start of Season (DOY):", sos)
print("End of Season (DOY):", eos)
print("Length of Season (Days):", los)
print("Peak of Season (DOY):", pop)
```

### Understanding the Outputs

The C++ module returns four NumPy arrays corresponding to the metrics for each requested season up to `max_seasons`.
The output shape is `(n_pixels, max_seasons)`.

- **SOS (Start of Season)**: The start day of the vegetative cycle.
- **EOS (End of Season)**: The end day of the vegetative cycle.
- **LOS (Length of Season)**: The total duration of the cycle (`EOS - SOS`).
- **POP (Peak of Season)**: The day where the vegetation index reaches its maximum within the fitted curve.

For non-vegetated pixels, or areas missing data, the algorithm skips the fitting and returns `NaN`. When exporting to a TIF file via `rioxarray`, remember to explicitly set the `nodata` tag so that GIS software properly hides missing pixels:

```python
import xarray as xr
import rioxarray

# Assuming band_sos is reshaped to (height, width)
ds_out = xr.DataArray(
    band_sos,
    coords={'y': ds.y, 'x': ds.x},
    dims=['y', 'x']
)

# IMPORTANT: Set nodata flag for proper GIS rendering
ds_out.rio.write_nodata(np.nan, inplace=True)
ds_out.rio.write_crs(ds.rio.crs, inplace=True)
ds_out.rio.to_raster('SOS_output.tif')
```

## Performance & Parallelization

The core loops of the phenology module avoid the Python Global Interpreter Lock (GIL) and are written using C++ and OpenMP. To fully leverage your hardware:
- Pass `n_jobs=-1` (or specify the exact thread count).
- Memory allocations inside the `phenology` loops are minimized by doing variable mapping via Eigen (`Eigen::Map`).
- If you use macOS, you need `libomp` to enable threading (e.g. `brew install libomp`). On Windows/Linux, OpenMP is usually enabled by default via MSVC or GCC.
