# API Reference

This page provides the comprehensive documentation for the primary functions exposed by the `cdts` library. These are the core building blocks you will use when writing custom Python scripts for Change Detection.

---

## `cdts.raster.run_landtrendr_image`

Executes the LandTrendr algorithm directly on a large multi-band GeoTIFF stored on disk. It handles reading the image in spatial chunks to prevent memory overload, processes the chunks in parallel across CPU cores, and writes the output directly back to disk.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `input_path` | `str` | **Required** | Path to the input multi-band GeoTIFF. |
| `output_dir` | `str` | **Required** | Directory where output files will be saved. |
| `start_year` | `int` | `2000` | Calendar year corresponding to the first band. |
| `max_segments` | `int` | `6` | Maximum number of line segments allowed per pixel. |
| `chunk_size` | `int` | `512` | Pixel size of the chunks to read and process at once. |
| `n_jobs` | `int` | `-1` | CPU cores to use. `-1` means all available cores. |
| `save_vertices`| `bool`| `False` | Whether to save the raw fitted vertices stack to disk. |
| `event_type` | `str` | `'loss'` | Type of event to extract (`'loss'` or `'gain'`). |

**Usage Example**

```python
from cdts.raster import run_landtrendr_image

# Runs LandTrendr on a massive GeoTIFF out-of-core and saves the event metrics directly
run_landtrendr_image(
    input_path="data/landsat_nbr_stack_1990_2020.tif",
    output_dir="results/landtrendr_outputs",
    start_year=1990,
    max_segments=6,
    chunk_size=1024,
    n_jobs=-1,
    event_type='loss'
)
```

---

## `cdts.raster.run_ccdc_image`

Executes the Continuous Change Detection and Classification (CCDC) algorithm directly on a dense multi-band, multi-date GeoTIFF stack stored on disk. Like its LandTrendr counterpart, it handles memory safely via out-of-core chunking.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `input_path` | `str` | **Required** | Path to the stacked GeoTIFF. |
| `output_dir` | `str` | **Required** | Directory to save the harmonic coefficients and break dates. |
| `dates` | `list`| **Required** | A list of ordinal dates matching the timestamps of the stack. |
| `num_bands` | `int` | `6` | Number of spectral bands per date in the stack. |
| `qa_band_idx` | `int` | `-1` | Zero-based index of the QA band. `-1` disables QA masking. |
| `max_segments` | `int` | `6` | Maximum number of distinct change segments to retain. |
| `conseq_anom` | `int` | `3` | Number of consecutive anomalies required to trigger a break. |
| `n_jobs` | `int` | `-1` | CPU cores to use for processing. |

**Usage Example**

```python
import numpy as np
from datetime import datetime
from cdts.raster import run_ccdc_image

# Generate a list of ordinal dates for the stack
dates_str = ["2020-01-15", "2020-02-01", "2020-02-17"]
dates_ordinal = [datetime.strptime(d, "%Y-%m-%d").toordinal() for d in dates_str]

# Process the GeoTIFF and save outputs
run_ccdc_image(
    input_path="data/dense_stack.tif",
    output_dir="results/ccdc_outputs",
    dates=dates_ordinal,
    num_bands=6,
    qa_band_idx=5, # The 6th band is the cloud mask
    max_segments=6,
    conseq_anom=3,
    n_jobs=-1
)
```

---

## `cdts.io.save_raster`

A highly robust, all-in-one utility to save NumPy arrays (2D, 3D, or 4D) and Xarray DataArrays to GeoTIFF format. It automatically handles `rasterio` profile generation, CRS/Transform extraction, deflate compression, and internal tiling.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `array` | `np.ndarray` | **Required**| The NumPy array or Xarray to save. |
| `output_path` | `str` | **Required**| Output filepath for the GeoTIFF. |
| `reference_cube`| `Any` | `None` | A rasterio dataset or xarray from which to inherit the CRS and transform. |
| `crs` | `str` | `'EPSG:4326'` | The Coordinate Reference System string. |
| `transform` | `Affine`| `None` | A `rasterio.Affine` transform object. |
| `nodata` | `float`| `None` | NoData value for the output raster. |

**Usage Example**

```python
import rasterio
from cdts.io import save_raster

# Read a source file to get its profile (or pass an xarray directly)
with rasterio.open("data/source.tif") as src:
    profile = src.profile

# Suppose we processed the data and got a 2D result array
result_array = (src.read(1) * 2).astype('float32')

# Save effortlessly without manually building a rasterio profile dictionary
save_raster(
    array=result_array,
    output_path="results/processed_data.tif",
    crs=profile['crs'],
    transform=profile['transform'],
    nodata=-9999.0
)
```

---

## `cdts.smooth.desawtooth`

Applies a temporal smoothing algorithm (despiking) to a 3D raster stack `(Time, Rows, Cols)` to remove ephemeral 1-year spikes, which are typically caused by unmasked clouds, shadows, or smoke. This is a highly recommended pre-processing step before running LandTrendr.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `raster_stack` | `np.ndarray`| **Required**| A 3D numpy array of the time series. |
| `threshold` | `float` | `0.1` | The value delta required to flag a spike. Varies by index scale. |
| `window_size` | `int` | `3` | The size of the rolling window used to detect anomalies. |

**Usage Example**

```python
import rasterio
from cdts.smooth import desawtooth
from cdts.io import save_raster

# Load the raw 3D array
with rasterio.open("data/raw_nbr_stack.tif") as src:
    raw_stack = src.read()
    profile = src.profile

# Despike the time series
smoothed_stack = desawtooth(raw_stack)

# Save the cleaned stack back to disk
save_raster(
    array=smoothed_stack,
    output_path="results/smoothed_nbr_stack.tif",
    reference_cube=src
)
```

---

## `cdts.tmask.apply_tmask_stack`

Applies the Time-series Cloud Masking (Tmask) algorithm to a 3D temporal stack to dynamically map missed clouds and shadows using robust harmonic regression (Huber).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `dates` | `np.ndarray`| **Required**| A 1D array of Julian dates matching the time dimension. |
| `green_stack` | `np.ndarray`| **Required**| A 3D numpy array of the Green spectral band. |
| `swir_stack` | `np.ndarray`| **Required**| A 3D numpy array of the SWIR spectral band (usually SWIR1). |
| `scale_factor` | `float` | `10000.0` | Multiplier to convert integer inputs to 0.0-1.0 surface reflectance. |

**Usage Example**

```python
import numpy as np
import rasterio
from cdts.tmask import apply_tmask_stack
from cdts.io import save_raster

dates = np.array([737425, 737441, 737457]) # Ordinal dates

with rasterio.open("data/green.tif") as src:
    green = src.read()
with rasterio.open("data/swir.tif") as src:
    swir = src.read()

# Generate the boolean cloud mask (True = Clear, False = Cloud/Shadow)
qa_mask = apply_tmask_stack(dates, green, swir, scale_factor=10000.0)

# Invert for CCDC (0 = Clear, 1 = Cloud)
ccdc_mask = (~qa_mask).astype('uint8')

# Save the generated mask
save_raster(ccdc_mask, "results/tmask_generated_qa.tif", crs=src.crs, transform=src.transform)
```


---

## `cdts.raster.run_landtrendr_array`

Executes the LandTrendr algorithm in memory on a 3D NumPy array stack `(Time, Rows, Cols)`. It utilizes Python's `multiprocessing` to distribute pixel trajectories across all CPU cores for rapid execution.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `years` | `np.ndarray` | **Required** | A 1D array of years matching the time dimension. |
| `raster_stack` | `np.ndarray`| **Required** | The 3D input numpy array. |
| `max_segments` | `int` | `6` | Maximum number of line segments allowed per pixel. |
| `pval_threshold` | `float`| `0.05` | P-value threshold for fitting statistical segments. |
| `n_jobs` | `int` | `-1` | CPU cores to use. `-1` means all available cores. |

**Usage Example**

```python
import numpy as np
from cdts.raster import run_landtrendr_array

# Synthetic data
years = np.arange(2000, 2020)
raster_stack = np.random.uniform(0.1, 0.8, size=(20, 100, 100))

# Run LandTrendr in parallel
vertices_stack = run_landtrendr_array(years, raster_stack, max_segments=6, n_jobs=-1)
```

---

## `cdts.landtrendr.run_landtrendr`

The lowest-level API entry point for LandTrendr. Operates on a single 1-Dimensional time series. It wraps the raw C++ core logic directly via `pybind11`. Ideal for testing, visualization, or custom integration.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `years` | `np.ndarray` | **Required** | A 1D array of years. |
| `values` | `np.ndarray` | **Required** | A 1D array of pixel values. |
| `max_segments` | `int` | `6` | Maximum segments to fit. |
| `pval_threshold` | `float`| `0.05` | Significance threshold. |

**Usage Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from cdts.landtrendr import run_landtrendr

years = np.arange(2000, 2010)
pixel_values = np.array([100, 95, 110, 800, 750, 780, 700, 650, 600, 500])

vertices = run_landtrendr(years, pixel_values, max_segments=4)
print(f"Fitted vertices: {vertices}")
```

---

## `cdts.raster.run_ccdc_array`

Applies the CCDC algorithm across a multi-dimensional array `(Bands, Time, Rows, Cols)` using parallel processing.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `dates` | `np.ndarray` | **Required** | Array of ordinal dates. |
| `spectral_stack` | `np.ndarray`| **Required** | A 4D numpy array or stacked 3D array of values. |
| `qa_stack` | `np.ndarray`| **Required** | A 3D numpy array indicating clear (0) or masked (1) pixels. |
| `num_bands` | `int` | `6` | Number of spectral bands per date. |
| `max_segments` | `int` | `6` | Maximum change segments per pixel. |
| `n_jobs` | `int` | `-1` | Number of workers. |
| `conseq_anom` | `int` | `3` | Consecutive anomalies required for a break. |

**Usage Example**

```python
from cdts.raster import run_ccdc_array

# Assume pre-loaded dates, spectral stack, and qa mask
coefs = run_ccdc_array(
    dates=ordinal_dates, 
    spectral_stack=spectral_data, 
    qa_stack=cloud_mask, 
    max_segments=6, 
    n_jobs=4
)
```

---

## `cdts.metrics.extract_events`

Parses the raw vertices output generated by LandTrendr and computes intuitive 2D spatial maps representing specific change events (e.g., Year of Detection, Magnitude, Duration).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `vertices_stack` | `np.ndarray`| **Required** | The 3D array output from `run_landtrendr_array`. |
| `event_type` | `str` | `'loss'` | Filter for `'loss'` or `'gain'` events. |
| `sort_by` | `str` | `'greatest'` | Strategy to pick the primary event. (`greatest`, `newest`, `fastest`, `longest`). |
| `min_magnitude`| `float` | `0.0` | Discard events with magnitude below this threshold. |
| `min_duration` | `int` | `1` | Discard events shorter than this many years. |
| `pre_val_threshold`| `float` | `0.0` | Discard events starting below this initial value. |

**Usage Example**

```python
from cdts.metrics import extract_events
from cdts.io import save_raster

# vertices_stack comes from run_landtrendr_array
events = extract_events(
    vertices_stack,
    event_type="loss",
    sort_by="greatest",
    min_magnitude=100.0
)

# Output is a dictionary of 2D arrays
yod_map = events["yod"]
mag_map = events["magnitude"]
```

---

## `cdts.ccdc.predict_synthetic_image`

Reconstructs a perfectly cloud-free Synthetic Image for any arbitrary date using the harmonic (Fourier) coefficients generated by CCDC.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `ccdc_coefs_stack`| `np.ndarray`| **Required** | The array of coefficients output by CCDC. |
| `target_julian_day`| `int` | **Required** | The ordinal date for which to predict the image. |
| `num_bands` | `int` | `6` | Number of spectral bands in the model. |

**Usage Example**

```python
from datetime import datetime
from cdts.ccdc import predict_synthetic_image

# Predict an image for July 1st, 2021
target_date = datetime(2021, 7, 1).toordinal()

synthetic_img = predict_synthetic_image(
    ccdc_coefs_stack=ccdc_results, 
    target_julian_day=target_date, 
    num_bands=6
)
```

---

## `cdts.ai.UTAE`

U-Net with Temporal Attention Encoder. A deep learning architecture specialized for multi-temporal, multi-spectral satellite imagery segmentation. Ideal for processing variable-length time-series with missing data (clouds).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `input_dim` | `int` | **Required** | Number of input spectral bands. |
| `encoder_widths` | `list` | `[64, 64, 64, 128]` | Channel dimensions for each layer of the encoder. |
| `decoder_widths` | `list` | `[32, 32, 64, 128]` | Channel dimensions for the decoder. |
| `out_conv` | `list` | `[32, 2]` | Dimensions of the final output layers (last item is number of classes). |
| `agg_mode` | `str` | `'att_group'`| Strategy for temporal aggregation. |

**Usage Example**

```python
import torch
from cdts.ai import UTAE

# Initialize a UTAE for 10-class segmentation with 4 spectral bands
model = UTAE(
    input_dim=4,
    out_conv=[32, 10]
)

# Dummy Data: (Batch, Time, Bands, H, W)
X = torch.randn(2, 12, 4, 128, 128)
predictions = model(X) # Output shape: (2, 10, 128, 128)
```
