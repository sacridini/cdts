# Continuous Change Detection and Classification (CCDC)

## 1. Introduction to CCDC

**CCDC** (Continuous Change Detection and Classification) is a paradigm-shifting algorithm designed to monitor land cover change using highly dense satellite time series data (such as all available clear Landsat observations). 

Instead of looking at data on an annual basis (like LandTrendr), CCDC models the natural seasonal phenology of the landscape using harmonic (Fourier) regression. When a sequence of new observations deviates significantly from this established harmonic model, CCDC registers a "structural break"—a change in land cover.

### Background

Developed by Zhe Zhu and Curtis Woodcock (see [References](#7-references)), CCDC is particularly powerful because it can detect changes at any time of the year and immediately provide harmonic coefficients that describe the new land cover state, which are excellent features for Random Forest classification. The **COLD** variant (also referenced below) simply increases the number of consecutive anomalies required to flag a break, trading sensitivity for robustness.

The `cdts` Python package is a C++ port of the original MATLAB implementation (`TrendSeasonalFit_v12_30Line.m` from GERSL/CCDC, including its Tmask cloud screening and its lasso fits through the bundled Fortran GLMnet, reproduced in single precision like the original), validated model-by-model against that code: identical model dates, categories and observation counts, with coefficients agreeing to ~1e-9. It wraps it in a modern, scalable architecture using `dask` and `xarray`.

---

## 2. End-to-End Workflow

In this tutorial, we will load a dense multi-band, multi-date raster stack, mask out clouds, run the CCDC algorithm, and extract the dates of change.

### Step 2.1: Preparing the Input Data

CCDC expects a dense, chronologically ordered time series of spectral bands and an associated Quality Assessment (QA) mask. 

- **Stacking Bands**: When using GeoTIFFs, the data must be interleaved by date. For example, if you are tracking 6 bands (Blue, Green, Red, NIR, SWIR1, SWIR2) and 1 QA band, your GeoTIFF must have 7 bands for Date 1, 7 bands for Date 2, and so on.
- **Dates**: a 1-dimensional list or array of Python ordinal days (`datetime.toordinal()`).
- **Scale**: like the original, CCDC expects **surface reflectance x 10000** (valid range 0-10000). Its lasso penalty (lambda = 20), range test and cloud screen are defined on that scale, so 0-1 reflectance must be multiplied by 10000 first (Landsat Collection-2 Level-2 DNs: `0.275 * DN - 2000`).

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

CCDC expects a 3D array of **Fmask codes**, exactly like the original: `0` clear land, `1` water (both usable), `2` cloud shadow, `3` snow, `4` cloud, `255` no observation. Snow is handled specially (a mostly-snow pixel gets a dedicated snow model), so keep it as `3` rather than folding it into the cloud code.

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
    
    # Convert to Fmask codes. Example for a Landsat Collection-2 QA_PIXEL band:
    fmask = np.zeros_like(qa_band, dtype=np.uint8)            # clear land
    fmask[(qa_band & (1 << 7)) != 0] = 1                        # water
    fmask[(qa_band & (1 << 4)) != 0] = 2                        # cloud shadow
    fmask[(qa_band & (1 << 5)) != 0] = 3                        # snow
    fmask[(qa_band & (1 << 3)) != 0] = 4                        # cloud
    fmask[(qa_band & 1) != 0] = 255                             # fill / no data
    qa_stack[i, :, :] = fmask
```

### Step 2.3: Running the Tool in Python

We use `cdts.raster.run_ccdc_array` for a stack (OpenMP over pixels), `cdts.ccdc.run_ccdc` for a single pixel, or `run_ccdc_image` for direct file processing.

```python
from cdts.raster import run_ccdc_array

# (bands, time, rows, cols)
raster_stack = spectral_stack.reshape(num_dates, 6, rows, cols).transpose(1, 0, 2, 3)

print("Running CCDC...")
ccdc_results = run_ccdc_array(
    dates=dates,
    raster_stack=raster_stack.astype(float),
    qa_stack=qa_stack,
    max_segments=6,
    return_coefs=True,    # Set to True to get the harmonic models back
    conseq_anom=6         # Number of consecutive anomalies to trigger a break
)
print("CCDC complete!")
```

For a single pixel, `run_ccdc(dates, values, qa)` returns one dict per model with
`t_start`, `t_end`, `t_break`, `coefs`, `rmse`, `magnitude`, `change_prob`,
`category` and `num_obs`, with the original's meanings.

## 3. Detailed Parameter Explanation

Tuning CCDC parameters is crucial for adapting the algorithm to specific ecosystems or sensor characteristics.

The defaults are the original's (`CCDC_Parameters.txt` defaults: 0.99, 6, 8).

- **`conseq_anom` (default: 6)**: consecutive anomalous observations required to flag a change (the original's `conse`).
- **`chi2_prob_threshold` (default: 0.99)**: change probability; the change threshold is `chi2inv(p, len(detection_bands))`. A lower value makes the model more sensitive to change.
- **`tmax_cg_prob_threshold` (default: 0.999999)**: observations above `chi2inv(p, ...)` are treated as outliers (e.g. unflagged clouds) and dropped.
- **`num_c` (default: 8)**: maximum number of harmonic coefficients (4, 6 or 8); models grow from 4 to 6 to 8 coefficients as observations accumulate.
- **`detection_bands` (default: `[1, 2, 3, 4, 5]`, Green..SWIR2)**, **`tmask_bands` (default: `[1, 4]`, Green and SWIR1)**, **`thermal_band`** (index of a brightness-temperature band in deg C x 100, if present), **`valid_range`**: the original's band roles, adjustable for other band layouts.
- **`min_obs`**: kept for API compatibility; the original's 12-observation minimum (3 x 4 coefficients) is fixed.

## 4. Exporting and Interpreting Results

When `return_coefs=True` is used, the output is a multi-dimensional array of shape `(max_segments, params_per_segment, rows, cols)`.

The number of parameters per segment is `3 + (num_bands * 9)`. The indices are:
- **Index 0**: `t_start` (Start date of the stable segment)
- **Index 1**: `t_end` (End date of the stable segment)
- **Index 2**: `t_break` (Date of the detected break/change, if any; 0 if no break)
- **For each band (starting at Index 3)**:
  - `rmse` (Root Mean Square Error of the fit)
  - 8 Harmonic Coefficients: Intercept, Slope, $cos(\omega t)$, $sin(\omega t)$, $cos(2\omega t)$, $sin(2\omega t)$, $cos(3\omega t)$, $sin(3\omega t)$ (where $\omega = 2\pi / 365.25$). As in the original, $t$ is the MATLAB datenum (Python ordinal day + 366); `cdts.ccdc.predict` and `predict_synthetic_image` take ordinal days and handle the offset.

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
- **Minimum Observations**: a model is only initialised once there are at least 12 clear observations spanning at least one year, as in the original.
- **Spatial chunking for distributed runs**: chunk the input spatially (e.g. `chunks={'time': -1, 'y': 256, 'x': 256}`). The `time` axis must **not** be chunked (`-1`), since CCDC needs a pixel's entire history to fit the harmonic model.
- **Cluster tuning**: CCDC is CPU-intensive — prefer compute-optimized worker nodes (e.g. `c2-standard` on GCP, `c5` on AWS) over high-memory nodes, since spatial chunks can be kept small.
- **Avoid `.compute()` on large outputs**: use `.to_zarr()` to sink data directly from the workers to cloud storage, bypassing your local head node entirely.

---

## 7. Distributed Processing with Dask

CDTS provides full native Dask integration for CCDC, letting you run heavy harmonic regression mapping across many remote machines simultaneously. This section walks through an end-to-end workflow: connecting to a Dask cluster, preparing a QA mask, fitting CCDC across all machines, and generating cloud-free synthetic imagery — all distributed.

### 7.1. End-to-End Example

The following script connects to a remote cluster, lazily loads a massive STAC catalog, prepares a QA mask for cloud filtering, fits the CCDC harmonics across all machines, and saves the output directly to a Cloud Storage bucket in parallel using Zarr.

```python
import xarray as xr
from dask.distributed import Client
import cdts

# 1. Connect to the multi-node Dask Cluster
scheduler_address = 'tcp://192.168.0.100:8786'
client = Client(scheduler_address)
print(f"Connected to Cluster! Dashboard: {client.dashboard_link}")

# 2. Lazily load a massive spatial extent from Planetary Computer (Landsat C2 L2)
cube = cdts.build_time_series(
    source='planetary_computer',
    collection='landsat-c2-l2',
    bbox=[-63.0, -11.0, -62.0, -10.0],  # Massive area
    start_date='2015-01-01',
    end_date='2023-12-31',
    bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'qa_pixel'],
    resolution=30
)

# 3. Extract and configure the Cloud / QA Mask
# CCDC needs to know which pixels to ignore during regression.
qa_mask = cdts.extract_water_mask(cube.sel(band='qa_pixel')) # or equivalent cloud mask parser
spectral_cube = cube.drop_sel(band=['qa_pixel'])

# 4. Extract Julian Dates
# CCDC fits harmonics based on the continuous Julian dates of the time-series.
dates_julian = cube.time.dt.dayofyear.values + (cube.time.dt.year.values * 365)

# 5. Define CCDC execution using the xarray accessor
# xarray will distribute the block-wise C++ regression across the workers.
ccdc_results = spectral_cube.cdts.run_ccdc(
    dates=dates_julian,
    qa_stack=qa_mask,           # Automatically skips clouded pixels
    max_segments=6,             # Maximum number of breaks a pixel can have
    return_coefs=True,          # Return the harmonic coefficients (Intercept, Slopes, Sines, Cosines)
    n_jobs=1                    # Set to 1 per worker (Dask already handles the parallelism)
)

# 6. Execute and Save in Parallel (Zarr)
# .to_zarr() is highly recommended for distributed writes to cloud buckets (S3/GCS)
# This is the moment computation actually triggers across all machines!
ccdc_results.to_zarr('s3://my-bucket/ccdc_coefficients.zarr', mode='w')

print("Distributed CCDC processing complete!")
```

### 7.2. Using the Results (Synthetic Images) Across the Cluster

Once you have your coefficients saved, you can load them lazily and generate synthetic (completely cloud-free) imagery for **any arbitrary day of the year**, leveraging Dask again for parallel reconstruction — complementing the single-machine example in [Section 5](#5-generating-synthetic-images-advanced).

```python
# Re-load the distributed results
coef_stack = xr.open_zarr('s3://my-bucket/ccdc_coefficients.zarr')

# We can use xr.apply_ufunc to apply the synthetic prediction across the cluster
def predict_wrapper(coefs):
    from cdts.ccdc import predict_synthetic_image
    return predict_synthetic_image(
        ccdc_coefs_stack=coefs,
        target_julian_day=(2024 * 365) + 200, # Example: Day 200 of 2024
        num_bands=6
    )

synthetic_cube = xr.apply_ufunc(
    predict_wrapper,
    coef_stack,
    input_core_dims=[['band']],
    output_core_dims=[['band']],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float]
)

# Save the synthetic cloud-free mosaic back to storage!
synthetic_cube.to_zarr('s3://my-bucket/ccdc_synthetic_day200.zarr', mode='w')
```

---

## 8. References

- Zhu, Z., & Woodcock, C. E. (2014). Continuous change detection and classification of land cover using all available Landsat data. **Remote Sensing of Environment**, 144, 152–171. [https://doi.org/10.1016/j.rse.2014.01.011](https://doi.org/10.1016/j.rse.2014.01.011)
- Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., & Zhou, C. (2020). Continuous monitoring of land disturbance based on Landsat time series (COLD). **Remote Sensing of Environment**, 238, 111116. [https://doi.org/10.1016/j.rse.2019.03.009](https://doi.org/10.1016/j.rse.2019.03.009)
