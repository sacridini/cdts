# Continuous Change Detection and Classification (CCDC)

This tutorial provides a comprehensive guide to running the Continuous Change Detection and Classification (CCDC) algorithm using the `cdts` Python package. It covers data preparation, execution options, parameter tuning, and how to interpret the results.

## 1. Introduction

CCDC is a robust algorithm designed to monitor land cover change using dense satellite time series data. It operates by estimating harmonic models to fit the seasonal and inter-annual trends of clear observations. When a new observation significantly deviates from the established model over consecutive dates, a "structural break" or change event is recorded.

## 2. Preparing the Input Time Series Data

CCDC expects a dense, chronologically ordered time series of spectral bands and an associated Quality Assessment (QA) mask. 

- **Stacking Bands**: The data must be structured as a multidimensional array. When using GeoTIFFs, the bands should be interleaved by date (e.g., `date1_band1`, `date1_band2`, ..., `date2_band1`, etc.).
- **Dates**: You must provide a 1-dimensional NumPy array of dates, typically expressed as Julian days or ordinal dates (e.g., `datetime.toordinal()`).
- **Cloud and Shadow Masking**: A QA mask is critical. Clouds, shadows, and snow can cause false anomalies. CCDC expects a QA array where clear pixels are distinguished from clouds/shadows (e.g., `0` for clear). If your data is already pre-masked and cloud-free, you can pass an array of zeros.

## 3. Running the Tool

The `cdts` package offers multiple entry points for CCDC depending on your workflow.

### Option A: Using Xarray and Dask (Recommended for scale)

If you are working with large datasets, the built-in Xarray accessor allows you to run CCDC efficiently out-of-core and in parallel using Dask.

```python
import xarray as xr
import numpy as np
import cdts # Registers the xarray accessor automatically

# Assume `da` is an xarray DataArray of shape (bands, time, y, x)
# Assume `qa_stack` is an array of shape (time, y, x) with 0 indicating clear pixels

# Extract ordinal dates from the xarray time dimension
dates = da.time.dt.dayofyear.values # or use ordinal dates

# Run CCDC
ccdc_results = da.cdts.run_ccdc(
    dates=dates,
    qa_stack=qa_stack,
    max_segments=6,
    return_coefs=True,
    conseq_anom=3
)
```

### Option B: Processing GeoTIFF Images Directly

If your data is stored locally as a single massive GeoTIFF stack, you can process it directly.

```python
import numpy as np
from cdts.raster import run_ccdc_image

# Array of ordinal dates corresponding to each timestamp in the stack
dates_array = np.array([737425, 737441, 737457]) 

run_ccdc_image(
    input_path="time_series_stack.tif",
    output_dir="./ccdc_output",
    dates=dates_array,
    num_bands=6,
    qa_band_idx=-1,  # Set to the index of the QA band if interleaved. -1 means no QA.
    max_segments=6,
    n_jobs=-1,       # Use all available CPU cores
    return_coefs=True,
    conseq_anom=3
)
```

## 4. Detailed Parameter Explanation

Tuning CCDC parameters is crucial for adapting the algorithm to specific ecosystems or sensor characteristics.

- **`min_obs` (default: 12)**: The minimum number of valid, clear observations required to initialize a harmonic model. Setting this too low may result in unstable models, while setting it too high delays the initialization.
- **`conseq_anom` (default: 3)**: The number of consecutive anomalous observations required to officially flag a structural break. 
- **`chi2_prob_threshold` (default: 0.99)**: The probability threshold for the chi-square distribution test. It determines the sensitivity of anomaly detection. A lower value makes the model more sensitive to change (potentially increasing noise), while a higher value requires a stronger signal to register an anomaly.
- **`max_segments` (default: 6)**: The maximum number of distinct change segments to retain per pixel.
- **`return_coefs` (default: True)**: If `True`, the output includes the full harmonic coefficients (intercept, slopes, harmonic terms) and RMSE for every band. If `False`, it only returns the dates of the structural breaks to save memory.

## 5. Exporting and Interpreting Results

When `return_coefs=True` is used, the algorithm outputs a multi-dimensional array containing the change segments and the harmonic models. The structure shape is `(max_segments, params_per_segment, rows, cols)`.

The number of parameters per segment is defined as `3 + (num_bands * 7)`. The indices are mapped as follows:

- **Index 0**: `t_start` (Start date of the stable segment)
- **Index 1**: `t_end` (End date of the stable segment)
- **Index 2**: `t_break` (Date of the detected break/change, if any; 0 if no break)
- **For each band (starting at Index 3)**:
  - `rmse` (Root Mean Square Error of the fit)
  - 6 Harmonic Coefficients: Intercept, Slope, $cos(\omega t)$, $sin(\omega t)$, $cos(2\omega t)$, $sin(2\omega t)$ (where $\omega = 2\pi / 365.25$).

### Generating Synthetic Images
One powerful feature of CCDC is the ability to reconstruct cloud-free images for any date using the harmonic coefficients.

```python
from cdts.ccdc import predict_synthetic_image

# Assuming ccdc_coefs_stack is the output from run_ccdc_array or loaded from disk
target_date = 737500 # A specific Julian or ordinal date

synthetic_img = predict_synthetic_image(
    ccdc_coefs_stack=ccdc_coefs_stack, 
    target_julian_day=target_date, 
    num_bands=6
)
```

## 6. Best Practices

- **High-Quality QA Masks**: CCDC is extremely sensitive to missed clouds and cloud shadows, which will be falsely identified as land cover changes. Ensure your QA masks are rigorous.
- **Data Density**: CCDC thrives on dense time series data. Harmonized Landsat and Sentinel-2 (HLS) data or multi-sensor virtual constellations work best.
- **Tuning `conseq_anom`**: Increasing the consecutive anomalies parameter (e.g., to 4 or 5) will reduce false positives caused by ephemeral changes (like snow or brief flooding) but will delay the detection of permanent land cover changes. Decreasing it (e.g., 2) will detect changes faster but might introduce noise.
- **Minimum Observations**: Ensure `min_obs` is large enough to capture at least one full annual cycle (e.g., 12 to 15 observations) before allowing a model break.
