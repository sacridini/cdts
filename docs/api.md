# API Reference

This page provides the comprehensive documentation for the primary functions exposed by the `cdts` library. These are the core building blocks you will use when writing custom Python scripts for Change Detection.

## Data Acquisition & Pre-processing

### `cdts.cube.build_time_series`

Dynamically builds a lazy, Dask-backed `xarray.DataArray` (DataCube) directly from cloud-native STAC catalogs (like AWS Earth Search, Microsoft Planetary Computer, or Brazil Data Cube). It automatically handles API pagination, reprojection, and spatial alignment without downloading the raw files first.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `source` | `str` | `'earth_search'`| STAC catalog alias or direct URL. |
| `collection` | `str` | `'sentinel-2-l2a'`| The dataset collection ID. |
| `bbox` | `list`| `None` | Bounding box `[minx, miny, maxx, maxy]` in EPSG:4326. |
| `vector_path`| `str` | `None` | Path to a vector file (Shapefile/GeoJSON) to derive `bbox`. |
| `start_date` | `str` | `'2020-01-01'`| Start date in `YYYY-MM-DD`. |
| `end_date` | `str` | `'2020-12-31'`| End date in `YYYY-MM-DD`. |
| `cloud_cover_max`| `int` | `30` | Maximum cloud cover percentage metadata filter. |
| `bands` | `list`| `None` | Specific bands to load (e.g., `["red", "nir"]`). |
| `resolution` | `int` | `None` | Spatial resolution (meters) for automatic reprojection. |
| `epsg` | `int` | `4326` | Output projection EPSG code. |
| `validate_items` | `bool` | `False` | Pre-tests each STAC URL to drop corrupted files before building the stack. |
| `access_token` | `str` | `None` | API token for restricted catalogs (e.g., Brazil Data Cube, or private AWS/API Gateway catalogs requiring URL tokens). |

**Usage Example**

```python
from cdts.cube import build_time_series

# Build a cloud-native xarray DataCube for an ROI
cube = build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    bbox=[-48.0, -16.0, -47.9, -15.9],
    start_date="2021-01-01",
    end_date="2021-12-31",
    cloud_cover_max=20,
    bands=["blue", "green", "red", "nir"],
    resolution=10,
    epsg=32722
)

print(cube) # Dask-backed xarray DataArray
```

### `cdts.gee.download_gee_timeseries`

Downloads analysis-ready time series data directly from Google Earth Engine (GEE). It handles Landsat sensor harmonization (Landsat 5/7/8/9), cloud masking (using QA_PIXEL), and annual compositing (Medoid) on Google's servers before downloading. It supports both direct local downloads via multithreaded tiling and asynchronous batch exports to Google Drive.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `roi` | `list` or `ee.Geometry`| **Required**| Bounding box `[min_lon, min_lat, max_lon, max_lat]` or an `ee.Geometry`. |
| `start_date` | `str` | **Required**| Start date in `YYYY-MM-DD`. |
| `end_date` | `str` | **Required**| End date in `YYYY-MM-DD`. |
| `out_dir` | `str` | **Required**| Directory to save the output `.tif` files. |
| `method` | `str` | `'direct'` | Download method. Use `'direct'` for immediate tiled local download, or `'drive'` for batch export to Google Drive. |
| `composite_type`| `str` | `'annual'` | The type of temporal composition to apply. Options include `'annual'` (LandTrendr-style Medoid composites) and `'dense'` (all valid observations for CCDC). |
| `bands` | `list` | `None` | Specific bands or indices to export. Supports standard bands (e.g., `'SR_B4'`) and on-the-fly indices (`'NDVI'`, `'NBR'`, `'EVI'`, `'NDWI'`, `'kNDVI'`). Defaults to all 6 spectral bands. |
| `project` | `str` | `None` | Google Cloud Project ID for GEE authentication. Highly recommended to prevent access errors. |

**Usage Example**

```python
from cdts.gee import download_gee_timeseries

# 1. Direct local tiled download for a small/medium region
download_gee_timeseries(
    roi=[-47.95, -15.85, -47.85, -15.75], 
    start_date='2010-01-01',
    end_date='2020-12-31', 
    out_dir='./gee_data',
    method='direct',
    composite_type='annual',
    project='my-gcp-project-id'
)

# 2. Export a massive region to Google Drive
download_gee_timeseries(
    roi=[-53.11, -25.31, -44.15, -19.78], 
    start_date='1985-01-01',
    end_date='2022-12-31', 
    out_dir='./data',
    method='drive',
    project='my-gcp-project-id'
)
```

### `cdts.io.save_raster`

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



### `cdts.io.load_raster`

Loads a GeoTIFF image into a NumPy array and retrieves its spatial profile. 
It also provides a built-in safety checker (`raster_check`) to quickly validate if your raster conforms to the strict format and value requirements of specific CDTS algorithms (like LandTrendr or CCDC) before you start heavy processing.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `file_path` | `str` | **Required** | Path to the raster file (`.tif`). |
| `raster_check` | `str` | `None` | The algorithm to validate against (`'landtrendr'`, `'ccdc'`, or `'cold'`). |

**Usage Example**

```python
from cdts.io import load_raster

# 1. Standard loading
array, profile = load_raster("data/annual_nbr_stack.tif")

# 2. Loading with Data Validation
# This will raise warnings if the data is unscaled (floats) or lacks the required time depth
lt_array, lt_profile = load_raster(
    "data/annual_nbr_stack.tif", 
    raster_check="landtrendr"
)
```

### `cdts.tmask.apply_tmask_stack`

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

### `cdts.smooth.desawtooth`

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

## Core Algorithms (Change Detection)

### `cdts.raster.run_landtrendr_image`

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

### `cdts.raster.run_landtrendr_array`

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

### `cdts.landtrendr.run_landtrendr`

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

### `cdts.raster.run_ccdc_image`

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

### `cdts.raster.run_ccdc_array`

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

## Metrics & Post-Processing

### `cdts.metrics.extract_events`

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

### `cdts.ccdc.predict_synthetic_image`

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

## AI & Deep Learning

### `cdts.ai.STACCubeDataset`

A specialized PyTorch `Dataset` that seamlessly bridges `xarray.DataArray` (or DataCubes) with deep learning workflows. It automatically slices massive satellite image stacks into smaller spatial patches (chips) suitable for neural network training and handles temporal padding.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `cube` | `xr.DataArray`| **Required** | The input DataCube. |
| `labels` | `xr.DataArray`| `None` | The target mask/labels (for training). |
| `patch_size` | `int` | `128` | Spatial size of the generated chips (e.g., 128x128). |
| `stride` | `int` | `128` | Stride for extracting patches. |
| `max_seq_len` | `int` | `None` | Maximum number of timesteps (pads with zeros if shorter). |

**Usage Example**

```python
from torch.utils.data import DataLoader
from cdts.ai import STACCubeDataset

# cube is a pre-loaded xarray
dataset = STACCubeDataset(
    cube=cube, 
    labels=ground_truth_mask, 
    patch_size=128, 
    stride=64, 
    max_seq_len=24
)

# Ready for PyTorch training loops
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### `cdts.ai.UTAE`

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

### `cdts.ai.TempCNN`

A 1D Temporal Convolutional Neural Network designed specifically for classifying satellite time-series at the pixel level. It uses causal/dilated convolutions to capture seasonal phenology without requiring recurrent layers (like LSTMs).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `input_dim` | `int` | **Required** | Number of spectral bands. |
| `num_classes`| `int` | **Required** | Number of output classification categories. |
| `sequence_len`|`int` | **Required** | Number of timesteps in the sequence. |
| `hidden_dims`| `int` | `64` | Number of filters in the convolutional layers. |
| `kernel_size`| `int` | `5` | Size of the 1D temporal convolution kernel. |
| `dropout` | `float`| `0.5` | Dropout probability for regularization. |

**Usage Example**

```python
import torch
from cdts.ai import TempCNN

model = TempCNN(input_dim=6, num_classes=5, sequence_len=36)

# Pixel-level time-series tensor (Batch, Channels, Time)
X = torch.randn(32, 6, 36) 
logits = model(X) # Shape: (32, 5)
```

### `cdts.ai.SiameseChangeDetector`

A PyTorch module for bi-temporal Change Detection. It uses a Siamese CNN architecture (two identical subnetworks sharing weights) to extract features from an image "Time 1" and "Time 2", followed by a contrastive distance metric to highlight areas of change.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `input_dim` | `int` | **Required** | Number of input spectral bands. |
| `backbone` | `str` | `'resnet18'` | The CNN backbone (`resnet18`, `resnet34`, or `unet`). |
| `pretrained` | `bool` | `True` | Load pre-trained ImageNet weights (adapts first layer). |
| `distance_metric`| `str`| `'euclidean'`| Metric used to compare features (`euclidean` or `cosine`). |

**Usage Example**

```python
import torch
from cdts.ai import SiameseChangeDetector

model = SiameseChangeDetector(input_dim=4, backbone='resnet18')

# Two temporal snapshots (Batch, Channels, H, W)
img_t1 = torch.randn(8, 4, 256, 256)
img_t2 = torch.randn(8, 4, 256, 256)

# Returns a spatial change probability map
change_map = model(img_t1, img_t2) # Shape: (8, 1, 256, 256)
```

### `cdts.ai.GeoFoundationViT`

A wrapper for Geospatial Foundation Models (like Prithvi or SatMAE) based on Vision Transformers (ViT). This class allows you to load pre-trained massive models and fine-tune them or use them for zero-shot feature extraction on your own rasters.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `model_name` | `str` | `'prithvi-100m'`| The name/ID of the foundation model to load. |
| `checkpoint_path`| `str`| `None` | Local path to `.pth` weights (if not downloading automatically). |
| `freeze_encoder`| `bool`| `False` | Freeze the transformer backbone for transfer learning. |
| `task` | `str` | `'segmentation'`| Fine-tuning head (`segmentation` or `classification`). |

**Usage Example**

```python
from cdts.ai import GeoFoundationViT

# Load a foundation model and freeze the encoder for transfer learning
model = GeoFoundationViT(
    model_name="prithvi-100m", 
    freeze_encoder=True, 
    task="segmentation"
)
```

