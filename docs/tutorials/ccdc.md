# Continuous Change Detection and Classification (CCDC)

## 1. Introduction to CCDC

**CCDC** (Continuous Change Detection and Classification) is a paradigm-shifting algorithm designed to monitor land cover change using highly dense satellite time series data (such as all available clear Landsat observations). 

Instead of looking at data on an annual basis (like LandTrendr), CCDC models the natural seasonal phenology of the landscape using harmonic (Fourier) regression. When a sequence of new observations deviates significantly from this established harmonic model, CCDC registers a "structural break"—a change in land cover.

### Background and References
Developed by Zhe Zhu and Curtis Woodcock, CCDC is particularly powerful because it can detect changes at any time of the year and immediately provide harmonic coefficients that describe the new land cover state, which are excellent features for Random Forest classification.

- **Original CCDC Paper**: [Zhu, Z. and Woodcock, C.E., 2014. Continuous change detection and classification of land cover using all available Landsat data. Remote sensing of Environment, 144, pp.152-171.](https://doi.org/10.1016/j.rse.2014.01.011)
- **COLD Algorithm (Evolution of CCDC)**: [Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A.H., Cohen, W.B., Qiu, S. and Zhou, C., 2020. Continuous monitoring of land disturbance based on Landsat time series. Remote Sensing of Environment, 238, p.111116.](https://doi.org/10.1016/j.rse.2019.03.009)

The `cdts` Python package implements the core harmonic modeling and break detection mathematically identical to the original C/C++ and MATLAB implementations, but wraps it in a modern, scalable architecture using `dask` and `xarray`.

---

## 2. End-to-End Workflow

In this tutorial, we will load a dense multi-band, multi-date raster stack, mask out clouds, run the CCDC algorithm, and extract the dates of change.

### Step 2.1: Preparing the Input Data

CCDC expects a dense, chronologically ordered time series of spectral bands and an associated Quality Assessment (QA) mask. 

- **Stacking Bands**: When using GeoTIFFs, the data must be interleaved by date. For example, if you are tracking 6 bands (Blue, Green, Red, NIR, SWIR1, SWIR2) and 1 QA band, your GeoTIFF must have 7 bands for Date 1, 7 bands for Date 2, and so on.
- **Dates**: You must provide a 1-dimensional list or array of dates, typically expressed as Julian days or ordinal dates (e.g., `datetime.toordinal()`).

```python
import numpy as np
import rasterio
from datetime import datetime

# Example dates for 50 clear observations
dates_str = ["2020-01-15", "2020-02-01", "2020-02-17", "..."]
# Convert to ordinal dates (number of days since Jan 1, 1 AD)
dates = np.array([datetime.strptime(d, "%Y-%m-%d").toordinal() for d in dates_str])

# Load the raster stack (Shape: Bands, Rows, Cols)
# If we have 50 dates and 7 bands per date (6 spectral + 1 QA), total bands = 350
input_path = "data/dense_landsat_stack.tif"
with rasterio.open(input_path) as src:
    data_stack = src.read()
    profile = src.profile

print(f"Data stack shape: {data_stack.shape}")
```

### Step 2.2: Extracting the QA Mask

CCDC expects a dedicated 3D mask array where `0` indicates a clear, valid observation, and `1` indicates a cloud, shadow, or snow pixel to be ignored.

```python
num_bands_per_date = 7
num_dates = len(dates)
rows, cols = data_stack.shape[1], data_stack.shape[2]

# Initialize arrays
spectral_stack = np.zeros((num_dates * 6, rows, cols), dtype=np.int16)
qa_stack = np.zeros((num_dates, rows, cols), dtype=np.uint8)

# Separate the spectral bands from the QA band
for i in range(num_dates):
    # The first 6 bands are spectral
    start_idx = i * num_bands_per_date
    spectral_stack[i*6 : (i+1)*6, :, :] = data_stack[start_idx : start_idx+6, :, :]
    
    # The 7th band is the QA mask
    qa_band = data_stack[start_idx + 6, :, :]
    
    # Convert QA values to a binary mask (0 = clear, 1 = cloud/shadow)
    # This depends on your specific QA band decoding logic. 
    # Example for a simple cloud mask where values > 0 are clouds:
    qa_stack[i, :, :] = (qa_band > 0).astype(np.uint8)
```

### Step 2.3: Running the Tool in Python

We use the `cdts.ccdc.run_ccdc` function (or `run_ccdc_image` for direct file processing) to execute the algorithm.

```python
from cdts.ccdc import run_ccdc

print("Running CCDC...")
ccdc_results = run_ccdc(
    dates=dates,
    spectral_stack=spectral_stack,
    qa_stack=qa_stack,
    num_bands=6,
    max_segments=6,
    return_coefs=True,    # Set to True to get the harmonic models back
    conseq_anom=3         # Number of consecutive anomalies to trigger a break
)
print("CCDC complete!")
```

## 3. Detailed Parameter Explanation

Tuning CCDC parameters is crucial for adapting the algorithm to specific ecosystems or sensor characteristics.

- **`min_obs` (default: 12)**: The minimum number of valid, clear observations required to initialize a harmonic model. Setting this too low may result in unstable models.
- **`conseq_anom` (default: 3)**: The number of consecutive anomalous observations required to officially flag a structural break. The **COLD** algorithm variant simply changes this to `6`.
- **`chi2_prob_threshold` (default: 0.99)**: The probability threshold for the chi-square distribution test. It determines the sensitivity of anomaly detection. A lower value makes the model more sensitive to change (potentially increasing noise).

## 4. Exporting and Interpreting Results

When `return_coefs=True` is used, the output is a multi-dimensional array of shape `(max_segments, params_per_segment, rows, cols)`.

The number of parameters per segment is `3 + (num_bands * 7)`. The indices are:
- **Index 0**: `t_start` (Start date of the stable segment)
- **Index 1**: `t_end` (End date of the stable segment)
- **Index 2**: `t_break` (Date of the detected break/change, if any; 0 if no break)
- **For each band (starting at Index 3)**:
  - `rmse` (Root Mean Square Error of the fit)
  - 6 Harmonic Coefficients: Intercept, Slope, $cos(\omega t)$, $sin(\omega t)$, $cos(2\omega t)$, $sin(2\omega t)$ (where $\omega = 2\pi / 365.25$).

### Extracting the Date of the First Change

```python
# The first segment is index 0. The date of the break is at parameter index 2.
first_break_dates = ccdc_results[0, 2, :, :]

# Filter out pixels that had no change (t_break == 0)
changed_pixels = first_break_dates > 0

print(f"Number of changed pixels: {np.sum(changed_pixels)}")

# Export the break dates to a GeoTIFF using CDTS built-in save_raster
from cdts.io import save_raster

save_raster(
    array=first_break_dates,
    output_path="results/ccdc_first_break.tif",
    crs=profile['crs'],
    transform=profile['transform'],
    nodata=0
)
```

## 5. Generating Synthetic Images (Advanced)

One powerful feature of CCDC is the ability to reconstruct cloud-free images for *any* date using the harmonic coefficients. This is highly useful for filling gaps in time series.

```python
from cdts.ccdc import predict_synthetic_image

# Predict what the landscape looked like on a specific date
target_date = datetime(2021, 7, 15).toordinal()

synthetic_img = predict_synthetic_image(
    ccdc_coefs_stack=ccdc_results, 
    target_julian_day=target_date, 
    num_bands=6
)

print(f"Synthetic image shape: {synthetic_img.shape}")
# Result: (6, Rows, Cols) - A perfectly clear 6-band image for July 15, 2021!
```

## 6. Best Practices

- **High-Quality QA Masks**: CCDC is extremely sensitive to missed clouds and cloud shadows, which will be falsely identified as land cover changes. Ensure your QA masks are rigorous (consider using the `Fmask` or `Tmask` algorithms).
- **Data Density**: CCDC thrives on dense time series data. Harmonized Landsat and Sentinel-2 (HLS) data or multi-sensor virtual constellations work best.
- **Minimum Observations**: Ensure `min_obs` is large enough to capture at least one full annual cycle (e.g., 12 to 15 observations) before allowing a model break.
