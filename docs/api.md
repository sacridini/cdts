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
