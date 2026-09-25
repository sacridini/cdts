# API Reference

This page provides the comprehensive documentation for the primary functions exposed by the `cdts` library. These are the core building blocks you will use when writing custom Python scripts for Change Detection.

## Data Acquisition & Pre-processing

### `cdts.cube.build_time_series`

Dynamically builds a lazy, Dask-backed `xarray.DataArray` (DataCube) directly from cloud-native STAC catalogs (like AWS Earth Search, Microsoft Planetary Computer, or Brazil Data Cube). It automatically handles API pagination, reprojection, and spatial alignment without downloading the raw files first.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `source` | `str` | `'earth_search'`| STAC catalog alias (`"earth_search"`, `"planetary_computer"`, `"brazil_data_cube"`) or any direct STAC API URL. |
| `collection` | `str` | `'sentinel-2-l2a'`| The dataset collection ID. Examples: `"sentinel-2-l2a"`, `"landsat-c2-l2"`, `"CBERS4A_WFI_L4_SR_1"`. Note: Available collections depend strictly on the chosen `source`. |
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

**Other sensors (MODIS, Sentinel-1 SAR):** `build_time_series` is generic STAC — not hardcoded to Sentinel-2/Landsat — so any collection hosted by `source` works, e.g. MODIS (`modis-13Q1-061`) or Sentinel-1 (`sentinel-1-rtc`) via `source="planetary_computer"`. `apply_cloud_mask` only decodes Sentinel-2's `scl` and Landsat's `qa_pixel`, so leave it `False` for these and QA/decode separately (`cdts.qc` for MODIS; Sentinel-1 has no clouds to mask). See the [STAC tutorial](tutorials/stac-downloads.md#3-other-sensors-modis--sentinel-1-sar) for full examples.

### `cdts.gee.download_gee_timeseries`

Downloads analysis-ready time series data directly from Google Earth Engine (GEE). It handles Landsat sensor harmonization (Landsat 5/7/8/9), cloud masking (using QA_PIXEL), and annual compositing (Medoid) on Google's servers before downloading. It supports both direct local downloads via multithreaded tiling and asynchronous batch exports to Google Drive.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `roi` | `str`, `list`, file path, `GeoDataFrame` or `ee.Geometry`| **Required**| Area to download. Accepts a Landsat WRS-2 tile id (`'217/076'`), a Sentinel-2 MGRS tile id (`'23KPQ'`), a bounding box `[min_lon, min_lat, max_lon, max_lat]`, a path to a local vector file (Shapefile, GeoPackage, GeoJSON, KML) or raster file (GeoTIFF), a `GeoDataFrame`/shapely geometry, or an `ee.Geometry`. Local inputs are read offline and reprojected to lon/lat; their bounding box is downloaded. |
| `start_date` | `str` | **Required**| Start date in `YYYY-MM-DD`. |
| `end_date` | `str` | **Required**| End date in `YYYY-MM-DD`. |
| `out_dir` | `str` | **Required**| Directory to save the output `.tif` files. |
| `method` | `str` | `'auto'` | Download method. `'auto'` (default) uses a concurrent tiled direct download and falls back to a Google Drive batch export for very large images or ones that hit Earth Engine's interactive compute limits. `'direct'` and `'drive'` force one route. |
| `composite_type`| `str` | `'annual'` | The type of temporal composition to apply. Options include `'annual'` (LandTrendr-style Medoid composites) and `'dense'` (all valid observations for CCDC). |
| `bands` | `list` | `None` | Specific bands or indices to export. Supports standard bands (e.g., `'SR_B4'`) and on-the-fly indices (`'NDVI'`, `'NBR'`, `'EVI'`, `'NDWI'`, `'kNDVI'`). Defaults to all 6 spectral bands. |
| `project` | `str` | `None` | Google Cloud Project ID for GEE authentication. Highly recommended to prevent access errors. |

**Usage Example**

```python
from cdts.gee import download_gee_timeseries

# 1. Automatic: direct tiled download, Drive export only when needed
download_gee_timeseries(
    roi=[-47.95, -15.85, -47.85, -15.75], 
    start_date='2010-01-01',
    end_date='2020-12-31', 
    out_dir='./gee_data',
    method='auto',
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

### `cdts.gee.downloader.download_gee_image`

Downloads a single `ee.Image` to a local GeoTIFF, picking the fastest route that will work. This is what `download_gee_timeseries` calls once per composite. Use it directly when you build your own image or need the tuning options.

`method='direct'` fixes one pixel grid for the whole ROI and fetches it as tiles with concurrent `ee.data.computePixels` calls. Each tile is written straight into its window of the output file, with no temporary tiles and no mosaicking. Tiles are sized on the fly from the concurrency Earth Engine allows, and concurrency adapts to `HTTP 429` responses. `method='drive'` runs a batch export through Google Drive. `method='auto'` uses `'direct'`, falling back to `'drive'` for images over `max_direct_mb` or on Earth Engine interactive compute limits.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `image` | `ee.Image` | **Required** | The image to download. |
| `roi` | `ee.Geometry` | **Required** | Region whose bounding box is downloaded. Use `cdts.gee.roi.resolve_roi` to build it from a WRS-2 id, bbox, vector/raster file or GeoDataFrame. |
| `out_filename` | `str` | **Required** | Output GeoTIFF path. Never left partially written. |
| `method` | `str` | `'auto'` | `'auto'`, `'direct'` or `'drive'`. |
| `scale` | `float` | `30` | Pixel size in meters (converted to degrees at the equator for a geographic CRS, as Earth Engine does). |
| `crs` | `str` | `'EPSG:4326'` | Output CRS, e.g. `'EPSG:32723'`. |
| `sub_tile_workers` | `int` | `16` | Upper bound on concurrent requests. The actual level starts at 4 and adapts to the account's limit (~40 standard tier, ~2 in Restricted Mode). |
| `max_tile_mb` | `float` | `None` (32) | Cap on the raw size of one tile request (≤ 32). |
| `max_direct_mb` | `float` | `4096` | In `'auto'` mode, larger images (raw size) go to a Drive export. |
| `max_retries` / `base_backoff` | `int` / `float` | `5` / `5.0` | Retry policy for network or server errors. Throttled (`429`) requests get short jittered retries instead. |
| `tile_size` | `float` | `None` | Deprecated and ignored (tiles used to be sized in degrees). |

**Returns** the output path, or `None` if the download failed. Nothing is written at `out_filename` in that case.

**Usage Example**

```python
from cdts.gee.auth import initialize_gee
from cdts.gee.roi import resolve_roi
from cdts.gee.harmonization import get_harmonized_collection
from cdts.gee.composites import create_annual_medoid
from cdts.gee.downloader import download_gee_image

initialize_gee(project="my-gcp-project")
roi = resolve_roi("data/study_area.gpkg")        # or "217/076", "23KPQ", [min_lon, min_lat, max_lon, max_lat]
col = get_harmonized_collection(roi, "1985-01-01", "2025-12-31")

for year in range(1985, 2026):
    img = create_annual_medoid(col, year)
    img = img.normalizedDifference(["SR_B5", "SR_B7"]).rename("NBR").toFloat()
    download_gee_image(img, roi, f"nbr_{year}.tif", crs="EPSG:32723", sub_tile_workers=32)
```

### `cdts.gee.roi.resolve_roi`

Turns any supported area description into the geometry the GEE functions need, so user code never has to build Earth Engine objects. `cdts.gee.roi.roi_bounds` is the fully offline part: it returns the lon/lat bounding box of a local input or a Sentinel-2 tile id. `cdts.gee.roi.s2_tile_utm_bounds('23KPQ')` gives a Sentinel-2 tile's exact UTM box (`('EPSG:32723', (600000, 7390200, 709800, 7500000))`).

Sentinel-1 has no fixed tiling grid (relative orbits describe ground tracks, not areas), so describe Sentinel-1 areas with any of the inputs below.

| Input | Example | Result |
| :--- | :--- | :--- |
| WRS-2 path/row | `'217/076'`, `'217_076'`, `'217076'` | Tile footprint, looked up from a Landsat Collection 2 scene |
| Sentinel-2 MGRS tile | `'23KPQ'`, `'T23KPQ'` | 109.8 km tile computed offline from the id (within ~50 m of real S2 footprints). Tiles crossing the 180° meridian raise `ValueError` |
| Bounding box | `[-43.6, -23.1, -43.1, -22.6]` | That box (lon/lat) |
| Vector file | `'area.shp'`, `'area.gpkg'`, `'area.geojson'`, `'area.kml'` | Extent of all features, reprojected to lon/lat |
| Raster file | `'reference.tif'` | Raster extent, reprojected to lon/lat |
| GeoDataFrame / GeoSeries | `geopandas.read_file(...)` | Extent, reprojected to lon/lat (a missing CRS is assumed lon/lat, with a warning) |
| shapely geometry | `box(...)` | Its bounds (assumed lon/lat) |
| `ee.Geometry` | | Passed through |

Local inputs are reduced to their bounding box. The download covers that box anyway, and a box keeps requests small however detailed the polygon is.

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
| `raster_check` | `str` | `None` | The algorithm to validate against. Options: `'landtrendr'`, `'ccdc'`, or `'cold'`. |

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

### `cdts.tmask.run_tmask_pixel`

Applies the Tmask (Zhu & Woodcock, 2014) cloud/shadow detection algorithm to a single pixel's time series. Fits a robust (Huber) harmonic regression to the Green and SWIR bands and flags observations whose residuals exceed fixed thresholds as cloud (unusually bright Green) or shadow (unusually dark SWIR). This is the per-pixel building block used internally by `apply_tmask_stack`.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `dates_julian` | `np.ndarray` | **Required** | 1D array of Julian/continuous DOY dates for the pixel's observations. |
| `green_band` | `np.ndarray` | **Required** | 1D reflectance array for the Green band. |
| `swir_band` | `np.ndarray` | **Required** | 1D reflectance array for the SWIR band (usually SWIR1, ~1.6μm). |
| `scale_factor` | `float` | `10000.0` | Multiplier applied to convert integer inputs to 0.0-1.0 surface reflectance. |

**Output**: 1D boolean array the same length as the input (`True` = clear, `False` = cloud/shadow). Returns all-`True` if fewer than 5 observations are supplied.

**Usage Example**

```python
import numpy as np
from cdts.tmask import run_tmask_pixel

dates = np.array([1, 17, 33, 49, 65, 81, 97])
green = np.array([900, 920, 4500, 910, 895, 905, 930])  # a cloud spike at index 2
swir = np.array([1200, 1180, 1190, 1210, 1195, 1205, 1188])

clear_mask = run_tmask_pixel(dates, green, swir, scale_factor=10000.0)
print(clear_mask)  # [ True  True False  True  True  True  True]
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

### `cdts.smooth.apply_savgol_filter`

Applies a Savitzky-Golay filter along the time axis of a data cube to remove minor temporal noise and regularize trajectories before AI training or classification.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `cube` | `np.ndarray` | **Required** | The data cube, shape `(Time, Bands, H, W)` or `(Time, H, W)`. |
| `window_length` | `int` | `5` | Length of the filter window (must be odd). |
| `polyorder` | `int` | `2` | Order of the polynomial fit to the samples within each window. |
| `axis` | `int` | `0` | The temporal axis. |

**Usage Example**

```python
from cdts import apply_savgol_filter

# raw_array: (Time, Y, X)
smoothed_array = apply_savgol_filter(raw_array, window_length=5, polyorder=2)
```

### `cdts.smooth.apply_whittaker_filter`

Applies a Whittaker smoother along the time axis. Often preferable to Savitzky-Golay for NDVI/EVI-style indices, since it penalizes roughness directly and handles missing/cloudy observations gracefully when per-observation weights are supplied. *(Not exported at the `cdts` top level — import from `cdts.smooth` directly.)*

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `cube` | `np.ndarray` | **Required** | 3D array `(Time, Y, X)`. |
| `lmbd` | `float` | `10.0` | Smoothing parameter — larger values produce a smoother curve. |
| `axis` | `int` | `0` | The temporal axis. |
| `weights` | `np.ndarray` | `None` | Optional array matching `cube`'s shape, with per-observation weights (`0` for cloud, `1` for clear). |

**Usage Example**

```python
from cdts.smooth import apply_whittaker_filter

smoothed = apply_whittaker_filter(raw_stack, lmbd=10.0, weights=clear_sky_weights)
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


### `cdts.build_local_cube`

Builds a lazy `xarray.DataArray` (DataCube) by parsing a directory of local GeoTIFF files. It extracts the date and band from the filenames using a regular expression.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `data_dir` | `str` | | Path to the directory containing `.tif` files. |
| `regex_pattern` | `str` | | Regular expression containing named groups `(?P<date>...)` and optionally `(?P<band>...)`. |
| `date_format` | `str` | `"%Y%m%d"` | String format to parse the extracted date. |

**Usage Example**

```python
import cdts

# Ingesting ARD files named like "CBERS_20200101_B04.tif"
cube = cdts.build_local_cube(
    data_dir="/data/tiles",
    regex_pattern=r".*_(?P<date>\d{8})_(?P<band>B\d{2})\.tif"
)
```

### `cdts.regularize_time_series`

Regularizes irregular time series to fixed temporal intervals (e.g., 16-day, monthly). Extremely useful for preparing data for machine learning or temporal harmonization.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `cube` | `xr.DataArray` | | The input spatiotemporal cube. |
| `freq` | `str` | `'16D'` | Pandas frequency string (e.g., `'16D'`, `'1M'`). |
| `method` | `str` | `'median'` | Aggregation method. Supported: `'median'`, `'medoid'`. |

**Usage Example**

```python
import cdts

# Regularize to 16-day medoid composites
reg_cube = cdts.regularize_time_series(cube, freq='16D', method='medoid')
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
| `n_jobs` | `int` | `-1` | CPU cores to use. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |
| `save_vertices`| `bool`| `False` | Whether to save the raw fitted vertices stack to disk. |
| `event_type` | `str` | `'loss'` | Type of event to extract. Options: `'loss'` (e.g., deforestation) or `'gain'` (e.g., regrowth). |

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

Executes the LandTrendr algorithm in memory on a 3D NumPy array stack `(Time, Rows, Cols)`. It utilizes a C++ backend with OpenMP to distribute pixel trajectories across CPU cores for extremely rapid batch execution. Also available as `DataArray.cdts.run_landtrendr(years, max_segments=6, pval_threshold=0.05, n_jobs=-1)` for lazy, Dask-backed execution.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `years` | `np.ndarray` | **Required** | A 1D array of years matching the time dimension. |
| `raster_stack` | `np.ndarray`| **Required** | The 3D input numpy array. |
| `max_segments` | `int` | `6` | Maximum number of line segments allowed per pixel. |
| `pval_threshold` | `float`| `0.05` | P-value threshold for fitting statistical segments. |
| `n_jobs` | `int` | `-1` | CPU cores to use. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |

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

### `cdts.landtrendr.apply_vertices`

Applies LandTrendr's structural vertices — the break years fitted on a primary index (FTV, "Fitted to Vertices") — to a secondary spectral band or index. Instead of re-segmenting the secondary band independently, it reuses the primary segmentation's years and linearly interpolates the secondary band's values at those years. Useful for smoothing/denoising a band that wasn't itself used for the disturbance segmentation (e.g. segment on NBR, apply vertices to NDVI or a raw spectral band).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `vertex_years` | `np.ndarray` | **Required** | Years of the vertices fitted on the primary index (from `run_landtrendr`). |
| `other_band_years` | `np.ndarray` | **Required** | Years available for the secondary band. |
| `other_band_values` | `np.ndarray` | **Required** | Secondary band's values matching `other_band_years`. |

**Usage Example**

```python
from cdts.landtrendr import run_landtrendr, apply_vertices

# 1. Segment on the primary index (e.g. NBR)
vertices = run_landtrendr(years, nbr_values, max_segments=4)
vertex_years = [v["year"] for v in vertices]

# 2. Reuse the same break years to fit a secondary index (e.g. NDVI)
ndvi_fitted = apply_vertices(vertex_years, years, ndvi_values)
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
| `n_jobs` | `int` | `-1` | CPU cores to use for processing. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |

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

Applies the CCDC algorithm across a multi-dimensional array `(Bands, Time, Rows, Cols)`. It utilizes a C++ backend with OpenMP to distribute pixels across CPU cores for extremely rapid batch execution. Also available as `DataArray.cdts.run_ccdc(dates, qa_stack=None, max_segments=6, return_coefs=True, conseq_anom=3, n_jobs=-1)` for lazy, Dask-backed execution.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `dates` | `np.ndarray` | **Required** | Array of ordinal dates. |
| `spectral_stack` | `np.ndarray`| **Required** | A 4D numpy array or stacked 3D array of values. |
| `qa_stack` | `np.ndarray`| **Required** | A 3D numpy array indicating clear (0) or masked (1) pixels. |
| `num_bands` | `int` | `6` | Number of spectral bands per date. |
| `max_segments` | `int` | `6` | Maximum change segments per pixel. |
| `n_jobs` | `int` | `-1` | Number of workers. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |
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

### `cdts.classify.train_ccdc_classifier`

Trains a `scikit-learn` `RandomForestClassifier` for land cover classification, using CCDC's harmonic coefficients (and RMSE) as features. A thin, opinionated wrapper — swap in your own `scikit-learn` model if you need a different classifier.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `X_train` | `np.ndarray` | **Required** | Shape `(n_samples, n_features)` — typically CCDC harmonic coefficients + RMSE extracted at training points. |
| `y_train` | `np.ndarray` | **Required** | Shape `(n_samples,)` — land cover class labels. |
| `n_estimators` | `int` | `100` | Number of trees in the forest. |
| `random_state` | `int` | `42` | Random seed for reproducibility. |

**Output**: a fitted `sklearn.ensemble.RandomForestClassifier`.

### `cdts.classify.classify_ccdc_stack`

Applies a trained classifier to a full CCDC coefficient GeoTIFF stack, out-of-core (reads/writes in chunks so it scales to large rasters without loading everything into memory).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `clf` | `RandomForestClassifier` | **Required** | A classifier trained via `train_ccdc_classifier` (or any `scikit-learn`-compatible model with a matching feature layout). |
| `coef_stack_path` | `str` | **Required** | Path to the CCDC coefficient GeoTIFF (e.g. written by `run_ccdc_image`). |
| `output_path` | `str` | **Required** | Path to write the classified land cover GeoTIFF (`uint8`, `nodata=0`). |
| `chunk_size` | `int` | `512` | Pixel size of the spatial chunks read/classified/written at a time. |

**Usage Example**

```python
from cdts.classify import train_ccdc_classifier, classify_ccdc_stack

# X_train: harmonic coefficients extracted at labeled training points
rf_model = train_ccdc_classifier(X_train=training_coefs, y_train=training_labels, n_estimators=100)

classify_ccdc_stack(
    clf=rf_model,
    coef_stack_path="output/ccdc_coefs.tif",
    output_path="output/land_cover_map.tif",
    chunk_size=512
)
```

## Phenology Extraction

### `cdts.phenology.run_phenology_dask`

Pixel-wise phenology curve fitting (Whittaker/HANTS smoothing + Levenberg-Marquardt curve fitting) across a Dask array's time axis, reimplemented in C++/Eigen/OpenMP from the methodology of the R package [`phenofit`](https://github.com/eco-hydro/phenofit) (Kong *et al.*, 2022). Extracts **21 metrics per season** (19 phenological dates/derived metrics + per-season R2 and RMSE goodness-of-fit) in a single pass. See the [Phenology tutorial](tutorials/phenology.md) for the full metric definitions, curve models, and a real-world walkthrough. Also available as `DataArray.cdts.run_phenology(...)`.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `arr` | `dask.array.Array` | **Required** | Input array, shape `(time, y, x)`. |
| `dates` | `np.ndarray` | **Required** | Dates matching the time dimension (day-of-year or continuous day count). |
| `curve_type` | `int` | **Required** | Curve model, from `cdts._core.phenology.CurveType`: `BECK`, `ELMORE`, `GU`, `KLOSTERMAN`, `ZHANG`, `AG` (Asymmetric Gaussian), or `DL` (Double Logistic). |
| `extraction_method` | `int` | `0` | Metric-extraction strategy passed to the C++ core; `0` (default) returns all 19 metrics regardless, computed via their respective methodologies (TRS/DER/Gu/Zhang — see the tutorial). |
| `max_seasons` | `int` | `2` | Maximum growing seasons to extract per pixel (per year, if `return_annual=True`). |
| `whittaker_lambda` | `float` | `10.0` | Smoothness penalty for the Whittaker smoother. |
| `apply_whittaker` | `bool` | `True` | Whether to apply Whittaker smoothing before curve fitting. |
| `apply_hants` | `bool` | `False` | Use HANTS (Fourier-based) smoothing instead of/alongside Whittaker. |
| `hants_frequencies` | `int` | `3` | Number of harmonic frequencies for HANTS. |
| `hants_threshold` | `float` | `0.1` | Outlier rejection threshold for HANTS. |
| `min_season_length` | `int` | `0` | Discard seasons shorter than this many calendar days. |
| `min_amplitude` | `float` | `0.0` | Discard seasons with less than this amplitude (peak minus trough). |
| `min_pixel_amplitude` | `float` | `0.1` | Minimum overall pixel amplitude required to attempt curve fitting at all. |
| `return_annual` | `bool` | `True` | Remap detected seasons into calendar years (`year` dim) instead of sequential season slots (`season` dim). |
| `base_year` | `int` | `2001` | First calendar year, used to decode dates and size the output when `return_annual=True`. |
| `n_jobs` | `int` | `-1` | CPU cores for the OpenMP batch pass. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |
| `weights` | `dask.array.Array` | `None` | Optional `(time, y, x)` per-observation reliability weights in `[0, 1]` (e.g. from `cdts.qc`), down-weighting unreliable observations in smoothing and curve fitting instead of trusting every observation equally. |
| `season_retry` | `bool` | `True` | Retry once with a relaxed trough threshold if a pixel's first pass finds no season at all. |

**Output**: array of shape `(21, max_seasons, y, x)` — see the [metrics list in the tutorial](tutorials/phenology.md#23-metric-extraction-methods) for the row order (`TRS2.sos`, `TRS2.eos`, ..., `LOS`, `POP`, `R2`, `RMSE`).

**Usage Example**

```python
from cdts.phenology import run_phenology_dask
from cdts._core.phenology import CurveType

# cube_16d: dask.array.Array, shape (time, y, x), 16-day composites
pheno_out = run_phenology_dask(
    arr=cube_16d,
    dates=dates_julian,
    curve_type=int(CurveType.BECK),
    max_seasons=2,
    apply_hants=True,
    hants_frequencies=3,
)

pheno_out = pheno_out.compute()
```

### `cdts.qc` — QA/QC Band Decoders

Ports of `phenofit`'s `qcFUN.R` decoders: turn a sensor's raw quality-assurance band into per-observation reliability weights in `[0, 1]`, suitable for the `weights` argument of `run_phenology_dask`/`DataArray.cdts.run_phenology` (or any other weighted smoothing you write yourself). All three share the same `(qa_array, wmin=0.2, wmid=0.5, wmax=1.0)`-style signature and return an array of the same shape as the input.

| Function | QA band decoded | Notes |
| :--- | :--- | :--- |
| `qc_modis_summary(qa, wmin=0.2, wmid=0.5, wmax=1.0)` | MOD13A1/A2/Q1 "SummaryQA" (pixel reliability) | `0`=good→`wmax`, `1`=marginal→`wmid`, `2`/`3`=snow or cloudy→`wmin`, other/fill→`0.0`. |
| `qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)` | MOD09A1/MYD09A1 500m 16-bit "State QA" | Decodes cloud state (bits 0-1), cloud shadow (bit 2), aerosol quantity (bits 6-7), and snow/ice (bit 12). |
| `qc_sentinel2_scl(scl, wmin=0.2, wmid=0.5, wmax=1.0)` | Sentinel-2 L2A Scene Classification Layer | Vegetation/bare soil/water/unclassified/thin cirrus→`wmax`, cloud medium probability→`wmid`, everything else (saturated, shadow, high-probability cloud, snow, no-data)→`wmin`. |

**Usage Example**

```python
from cdts.qc import qc_modis_summary

# qa_cube: (time, y, x) MOD13 SummaryQA band, aligned with cube_16d
weights = qc_modis_summary(qa_cube)  # 0=good, 1=marginal, 2=snow/ice, 3=cloudy -> [1.0, 0.5, 0.2, 0.2]

pheno_results = cube_16d.cdts.run_phenology(
    dates=dates_julian,
    curve_type=int(CurveType.BECK),
    weights=weights,
)
```

## Metrics & Post-Processing

### `cdts.metrics.extract_events`

Parses the raw vertices output generated by LandTrendr and computes intuitive 2D spatial maps representing specific change events (e.g., Year of Detection, Magnitude, Duration).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `vertices_stack` | `np.ndarray`| **Required** | The 3D array output from `run_landtrendr_array`. |
| `event_type` | `str` | `'loss'` | Filter for event types. Options: `'loss'` or `'gain'`. |
| `sort_by` | `str` | `'greatest'` | Strategy to pick the primary event. Options: `'greatest'` (largest magnitude), `'newest'` (most recent), `'fastest'` (shortest duration), `'longest'` (longest duration). |
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


### `cdts.masks.extract_water_mask`

Derives a persistent water mask from a CCDC coefficient stack, using the first segment's Green and SWIR intercepts (water reflects more strongly in Green than SWIR, and has low absolute SWIR reflectance). Useful as a static mask to exclude water bodies before running LandTrendr/CCDC change detection or classification.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `ccdc_coefs_stack` | `np.ndarray` | **Required** | CCDC coefficients, shape `(max_segments, params_per_seg, rows, cols)` (as produced by `run_ccdc_array`/`run_ccdc_image`). |
| `green_band_idx` | `int` | **Required** | 0-based index of the Green band within the coefficient stack's band ordering. |
| `swir_band_idx` | `int` | **Required** | 0-based index of the SWIR1/SWIR2 band. |

**Output**: `uint8` array of shape `(rows, cols)`, where `1` = persistent water.

**Usage Example**

```python
from cdts.masks import extract_water_mask

# coef_stack: (max_segments, params_per_seg, rows, cols), Green is band 1, SWIR1 is band 4
water_mask = extract_water_mask(coef_stack, green_band_idx=1, swir_band_idx=4)
```

### `cdts.spatial.apply_mmu_filter`

Applies a Minimum Mapping Unit (MMU) spatial filter to a disturbance/classification raster **on disk**, removing connected pixel groups smaller than `mmu_pixels` (set to nodata). Reduces "salt and pepper" noise, e.g. after `run_landtrendr_image`/`extract_events`.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `input_path` | `str` | **Required** | Path to the input single-band GeoTIFF (e.g. Year of Detection or magnitude). |
| `output_path` | `str` | **Required** | Path to write the filtered GeoTIFF. |
| `mmu_pixels` | `int` | `11` | Minimum connected-component size (in pixels) to keep; smaller groups are set to nodata. |

**Usage Example**

```python
from cdts.spatial import apply_mmu_filter

# Erase isolated disturbance patches smaller than 11 pixels
apply_mmu_filter(
    input_path="results/yod_map.tif",
    output_path="results/yod_map_mmu.tif",
    mmu_pixels=11,
)
```

### `cdts.spatial.apply_majority_filter`

Applies an in-memory spatial majority (mode) filter to regularize a classification array — every pixel is replaced by the most common value in its neighborhood. Typically run after pixel-based classification (TWDTW, SOM, CCDC) to clean up noisy maps.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `image` | `np.ndarray` | **Required** | 2D classification array. |
| `size` | `int` | `3` | Size of the moving window (e.g. `3` for a 3x3 neighborhood). |

**Usage Example**

```python
from cdts import apply_majority_filter

regularized_map = apply_majority_filter(classified_map, size=3)
```

### `cdts.spatial.apply_bayesian_filter`

Applies Bayesian spatial smoothing to per-class probability maps (e.g. from `classify_twdtw`, a deep-learning model, or any classifier that exposes class probabilities). Unlike `apply_majority_filter`, it weighs by model confidence: each pixel's probability is multiplied by the neighborhood-averaged probability before taking the arg-max. *(Not exported at the `cdts` top level — import from `cdts.spatial` directly.)*

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `probs` | `np.ndarray` | **Required** | 3D array `(Classes, Y, X)` of per-class probabilities/confidence scores. |
| `window_size` | `int` | `3` | Size of the spatial averaging window. |

**Output**: 2D `np.ndarray` (`Y, X`) of the winning class index after smoothing.

**Usage Example**

```python
from cdts.spatial import apply_bayesian_filter

# probs: (Classes, Y, X) from a softmax/probability output
smoothed_classification = apply_bayesian_filter(probs, window_size=3)
```

### `cdts.generate_landtrendr_accuracy_dashboard`

Generates an interactive, serverless HTML dashboard to validate LandTrendr change detection results against raw spatial-temporal data. 
It automatically extracts time-series trajectories and true-color spatial context chips (25x25) for visual interpretation. Features a responsive mobile and desktop layout, live Kappa index calculation, and CSV Export.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `cube` | `xr.DataArray` | | The input spatiotemporal STAC cube containing the original bands. |
| `points` | `str/GDF/list`| | Points to validate. Can be a path to a vector file (`.shp`), a GeoDataFrame, or a list of `(lon, lat)` tuples. |
| `lt_results` | `xr.DataArray` | `None` | (Optional) The output metrics from `cdts.metrics.extract_events`. Used to extract Predicted YOD automatically. |
| `output_html`| `str` | `'lt_accuracy_dashboard.html'` | The path to save the generated HTML file. |
| `window_size`| `int` | `25` | The size of the spatial context window (width and height in pixels). |

**Usage Example**

```python
import cdts

# Generate an interactive HTML Validation tool reading points directly from a Shapefile
cdts.generate_landtrendr_accuracy_dashboard(
    cube=stac_cube,
    points="data/validation_points.shp",
    lt_results=events_ds,
    output_html="validation_rondonia.html"
)
```

## Trend Analysis

### `cdts.trend.run_mann_kendall_dask`

Pixel-wise Mann-Kendall trend test + Theil-Sen slope estimator across a Dask array's time axis, ported from [`pymannkendall`](https://github.com/mmhs013/pymannkendall) to a C++/OpenMP backend for per-pixel throughput. See the [Mann-Kendall tutorial](tutorials/mann_kendall.md) for the full method comparison and a real-world walkthrough. Also available as `DataArray.cdts.run_mann_kendall(...)`.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `arr` | `dask.array.Array` | **Required** | Input array, shape `(time, y, x)`. |
| `method` | `str` | `'hamed_rao'` | `'original'`, `'hamed_rao'` (autocorrelation-corrected, recommended for annual composites), `'yue_wang'` (alternative correction), or `'seasonal'` (pools per-season scores over `period` slots — test a raw sub-annual series directly). |
| `alpha` | `float` | `0.05` | Significance level for the `h`/`trend` decision. |
| `lag` | `int` | `None` | Number of first significant lags for the `hamed_rao`/`yue_wang` autocorrelation correction. `None` uses the full series length. |
| `period` | `int` | `1` | Season-cycle length, only used when `method='seasonal'` (e.g. `23` for MODIS 16-day annual cycles, `12` for monthly data). |
| `min_valid` | `int` | `4` | Pixels with fewer non-NaN observations than this are returned as all-NaN. |
| `n_jobs` | `int` | `-1` | CPU cores for the OpenMP batch pass. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |

**Output**: array of shape `(9, y, x)` — rows `trend, h, p, z, tau, s, var_s, slope, intercept` (see `cdts.trend.MK_METRIC_NAMES`). `slope`/`intercept` are per time step, except for `method='seasonal'` where they are per full `period` cycle — see the [units warning in the tutorial](tutorials/mann_kendall.md#2-background-which-method-should-i-use).

**Usage Example**

```python
from cdts.trend import run_mann_kendall_dask

# arr: dask.array.Array, shape (n_years, rows, cols) - one composite per year
trend_out = run_mann_kendall_dask(arr, method="hamed_rao", alpha=0.05)

trend_out = trend_out.compute()
slope_map = trend_out[7]      # 'slope' row
significant = trend_out[1] == 1.0  # 'h' row
```

## Segmentation (SNIC)

### `cdts.segmentation.run_snic`

SNIC superpixel segmentation (Achanta & Süsstrunk, 2017) of an image or a whole time series cube, implemented in C++/Eigen/OpenMP; from the same seeds it gives the same labels as the authors' reference implementation. Every leading axis (e.g. `time`, `band`) becomes a feature, so segments group pixels with similar trajectories. See the [SNIC tutorial](tutorials/snic.md). Also available as `DataArray.cdts.run_snic(...)`, which returns an `xr.Dataset`.

**Parameters**:
- `data` (`np.ndarray`): `(y, x)`, `(feature, y, x)` or `(time, band, y, x)`. NaN pixels are left unlabelled.
- `spacing` (`float | (float, float)`): seed spacing of the R `snic`/`sits_snic()` grids (default 10).
- `compactness` (`float`): spatial regularity `M` (default 0.5, the sits default).
- `seeds` (`(n, 2)` array, optional): explicit `(row, col)` seeds; overrides the grid.
- `grid` (`str`): `"rectangular"`, `"diamond"`, `"hexagonal"` or `"random"`.
- `padding` (`float | (float, float)`, optional): seed-free margin (default `spacing / 2`).
- `tile_size` (`int | (int, int)`, optional): segment independent tiles in parallel.
- `n_jobs` (`int`): OpenMP threads (`-1` = all but one).

**Output**: `SnicResult` with `labels` `(y, x)`, `means` `(n_seeds, *feature_shape)`, `centroids` `(n_seeds, 2)`, `sizes`, `seeds`.

```python
from cdts import run_snic, snic_to_polygons
res = run_snic(cube, spacing=10, compactness=0.5, tile_size=512)
gdf = snic_to_polygons(res, transform=transform, crs=crs, include_means=True)
```

### `cdts.segmentation.snic_grid`

Seed grids of the R `snic` package (`snic_grid`): `(n, 2)` 0-based `(row, col)`.

### `cdts.segmentation.snic_to_polygons`

Polygonises `SnicResult.labels` into a GeoDataFrame (`supercells`, `x`, `y`, `n_pixels`, optional `f0..fN` means, `geometry`), like `sits_segment()`.

## Change Monitoring (BFAST)

### `cdts.bfast.run_bfast_monitor_dask`

Pixel-wise near-real-time structural change monitoring (`bfastmonitor`), ported from the R package [`bfast`](https://github.com/bfast2/bfast) and its [`strucchangeRcpp`](https://github.com/bfast2/strucchangeRcpp) dependency's OLS-MOSUM monitoring process (Chu, Stinchcombe & White, 1996) to a C++/OpenMP backend, with the same Dask distribution strategy as `run_mann_kendall_dask`. Fits a trend + harmonic model on a stable history period, then flags the first point in the subsequent monitoring period where the residual fluctuation process crosses a significance boundary — "is a disturbance happening right now". See the [BFAST Monitor tutorial](tutorials/bfast_monitor.md) for the full method background, scope (only `type="OLS-MOSUM"` and `history="all"` are implemented — no STL, no Bai-Perron multi-breakpoint search), and a documented false-positive-rate caveat. Also available as `DataArray.cdts.run_bfast_monitor(...)`.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `arr` | `dask.array.Array` | **Required** | Input array, shape `(time, y, x)`, one observation every `1/frequency` (regular, synthetic time — matches R's `ts` semantics, not real per-observation dates). |
| `start_time` | `float` | **Required** | The series' start time (e.g. `2010.0`). Should be an integer so harmonic terms align with calendar seasons. |
| `monitor_start_time` | `float` | **Required** | Time at which monitoring begins — the history/monitoring split point (e.g. `2022.0`). |
| `frequency` | `int` | **Required** | Observations per year (e.g. `23` for 16-day composites, `12` for monthly). |
| `order` | `int` | `3` | Harmonic order for the seasonal regressors (capped at `frequency`). |
| `h` | `float` | `0.25` | MOSUM window size as a fraction of history length. Must be one of `0.25`, `0.5`, `1.0` (the critical-value table's grid). |
| `period` | `int` | `10` | How many "history lengths" ahead the monitoring boundary's guarantee covers. Must be one of `2`, `4`, `6`, `8`, `10`. |
| `alpha` | `float` | `0.05` | Significance level. |
| `min_valid` | `int` | `10` | Pixels with fewer non-NaN history observations than this are returned as invalid (`valid=0`, all other metrics `NaN`). |
| `n_jobs` | `int` | `-1` | CPU cores for the OpenMP batch pass. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |

**Output**: array of shape `(7, y, x)` — rows `breakpoint, breakpoint_idx, magnitude, sigma, n_history, has_break, valid` (see `cdts.bfast.BFM_METRIC_NAMES`).

**Usage Example**

```python
from cdts.bfast import run_bfast_monitor_dask

# arr: dask.array.Array, shape (time, y, x), 16-day composites from 2010
out = run_bfast_monitor_dask(
    arr, start_time=2010.0, monitor_start_time=2022.0, frequency=23,
).compute()

disturbed = out[5] == 1.0        # 'has_break' row
break_idx = out[1]               # 'breakpoint_idx' row
```

### `cdts.bfast.run_bfast_lite_dask`

Pixel-wise single-pass multiple-breakpoint detection (`bfastlite`), ported from the R package [`bfast`](https://github.com/bfast2/bfast) and its [`strucchangeRcpp`](https://github.com/bfast2/strucchangeRcpp) dependency's `breakpoints()` (the Bai & Perron, 2003 optimal multiple-breakpoint dynamic program, via Brown-Durbin-Evans recursive residuals) to a C++/OpenMP backend, with the same Dask distribution strategy as `run_bfast_monitor_dask`. Unlike `run_bfast_monitor_dask` (single break/no-break, near-real-time), this retrospectively segments the *whole* series into an optimal number of pieces (chosen by minimizing the LWZ model-selection criterion, matching bfastlite's own default `breaks="LWZ"`). See the [BFAST Lite tutorial](tutorials/bfast_lite.md) for the full method background, scope (no STL — the classic iterative `bfast()` isn't ported), and validation details. Also available as `DataArray.cdts.run_bfast_lite(...)`.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `arr` | `dask.array.Array` | **Required** | Input array, shape `(time, y, x)`, same synthetic/regular time convention as `run_bfast_monitor_dask`. |
| `start_time` | `float` | **Required** | The series' start time (e.g. `2010.0`). |
| `frequency` | `int` | **Required** | Observations per year. |
| `order` | `int` | `3` | Harmonic order for the seasonal regressors. |
| `h` | `float` | `0.15` | Minimum segment size as a fraction of the series length. Unlike `run_bfast_monitor_dask`'s `h`, this is a free fraction (no critical-value-table grid restriction). |
| `max_breaks_output` | `int` | `5` | Maximum number of breakpoints to report per pixel (also caps the search depth attempted, alongside the theoretical bound). |
| `min_valid` | `int` | `20` | Pixels with fewer non-NaN observations than this are returned as invalid. |
| `n_jobs` | `int` | `-1` | CPU cores for the OpenMP batch pass. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |

**Output**: array of shape `(5 + max_breaks_output, y, x)` — rows `n_breaks, rss, lwz, n_valid, valid, breakpoint_idx_1, ..., breakpoint_idx_{max_breaks_output}` (see `cdts.bfast.bfl_metric_names(max_breaks_output)`). Breakpoint slots past `n_breaks` are `NaN`.

**Usage Example**

```python
from cdts.bfast import run_bfast_lite_dask

# arr: dask.array.Array, shape (time, y, x), 16-day composites from 2010
out = run_bfast_lite_dask(arr, start_time=2010.0, frequency=23).compute()

n_breaks = out[0]
first_break_idx = out[5]  # NaN where n_breaks == 0
```

## Time-Series Classification (TWDTW)

### `cdts.twdtw.run_twdtw`

The lowest-level TWDTW entry point: computes the Time-Weighted Dynamic Time Warping distance between a single time series and a reference pattern. Supports multivariate series (2D `ts_values`, shape `Time x Bands`). Ideal for testing, visualization, or one-off comparisons.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `ts_values` | `np.ndarray` | **Required** | 1D or 2D (`Time x Bands`) pixel time series. |
| `ts_dates` | `np.ndarray` | **Required** | Dates matching `ts_values`. |
| `pattern_values` | `np.ndarray` | **Required** | 1D or 2D reference signature. |
| `pattern_dates` | `np.ndarray` | **Required** | Dates matching `pattern_values`. |
| `alpha` | `float` | `0.1` | Steepness of the logistic time-weight penalty. |
| `beta` | `float` | `0.05` | Midpoint of the time-weight penalty. |
| `gamma` | `float` | `50.0` | Weight scaling constant. |
| `max_time_warp` | `int` | `365` | Maximum allowed temporal shift, in days. |
| `subsequence_matching` | `bool` | `False` | Allow matching a subsequence of `ts_values` instead of requiring full-sequence alignment. |
| `abort_threshold` | `float` | `inf` | Early-abandonment distance threshold. |
| `return_path` | `bool` | `False` | If `True`, also return the optimal warping path. |

**Output**: the TWDTW distance (`float`), or `(distance, path)` if `return_path=True`.

### `cdts.twdtw.run_twdtw_batch`

Runs TWDTW across an entire raster (3D/4D array) against a single reference pattern, using the C++/OpenMP batch engine.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `values_array` | `np.ndarray` | **Required** | `(Y, X, Time)` or `(Y, X, Time, Bands)` array. |
| `dates_array` | `np.ndarray` | **Required** | 1D array of dates matching the time dimension. |
| `pattern_values` | `np.ndarray` | **Required** | 1D or 2D reference signature. |
| `pattern_dates` | `np.ndarray` | **Required** | Dates matching `pattern_values`. |
| `alpha`, `beta`, `gamma`, `max_time_warp`, `subsequence_matching`, `abort_threshold` | | *(same as `run_twdtw`)* | |
| `n_jobs` | `int` | `-1` | CPU cores for the OpenMP batch pass. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |

**Output**: 2D `np.ndarray` (`Y, X`) of TWDTW distances to the pattern.

### `cdts.twdtw.classify_twdtw`

Classifies a full raster cube against multiple reference patterns (one per class), picking the class with the lowest TWDTW distance at each pixel.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `values_array` | `np.ndarray` | **Required** | `(Y, X, Time)` or `(Y, X, Time, Bands)` array. |
| `dates_array` | `np.ndarray` | **Required** | 1D array of dates matching the time dimension. |
| `patterns` | `dict` | **Required** | Maps `class_name -> (pattern_values, pattern_dates)`. |
| `alpha`, `beta`, `gamma`, `max_time_warp`, `subsequence_matching` | | *(same as `run_twdtw`)* | |
| `n_jobs` | `int` | `-1` | CPU cores for the OpenMP batch pass. `-1` reserves one core (`max(1, cpu_count - 1)`), so the host stays responsive. |

**Output**: `(classification_map, distance_map, class_names)` — a 2D `int` array of the winning class index, a 2D `float` array of its TWDTW distance, and the list of class names (index-aligned with `classification_map`).

**Usage Example**

```python
from cdts.twdtw import classify_twdtw
import numpy as np

dates = np.arange(1, 366, 16)  # DOY for a 16-day composite
forest_sig = np.random.rand(23, 4)   # (Time, Bands)
soy_sig = np.random.rand(23, 4)

patterns = {"Forest": (forest_sig, dates), "Agriculture": (soy_sig, dates)}

classes_map, dist_map, class_names = classify_twdtw(
    values_array=cube_16d.values,
    dates_array=dates,
    patterns=patterns,
    alpha=0.1,
    beta=0.05,
    max_time_warp=60,
    n_jobs=-1,
)

# Mask out pixels that matched poorly with every known signature
final_classification = np.where(dist_map < 15.0, classes_map, -1)
```

## Unsupervised Clustering (SOM)

### `cdts.ai.SOM`

A Self-Organizing Map accelerated by C++/OpenMP, for unsupervised clustering and dimensionality reduction of time-series/spectral features (e.g. discovering trajectory clusters without labeled training data, or filtering noisy training samples before supervised classification).

**Parity with `minisom`:** `cdts.ai.SOM` is an operation-by-operation port of Python [`minisom`](https://github.com/JustGlowing/minisom) 2.3.x. With the same `random_seed` and arguments it draws the same initial weights and sample order and applies the same neighborhood, decay and update arithmetic, so the trained codebook is **bit-for-bit identical** to `MiniSom.train` (`algorithm='online'`) or `MiniSom.train_batch_offline` (`algorithm='batch'`) — and 30-190x faster. `num_iters` therefore has `minisom`'s meaning: single-sample updates for `'online'`, full passes over the data for `'batch'`. The batch trainer is deterministic for any `n_jobs`.

**Constructor Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `x` | `int` | **Required** | Number of neurons along the grid's first dimension. |
| `y` | `int` | **Required** | Number of neurons along the grid's second dimension. |
| `input_len` | `int` | **Required** | Number of input features per sample. |
| `sigma` | `float` | `1.0` | Initial spread of the neighborhood function. |
| `learning_rate` | `float` | `0.5` | Initial learning rate. |
| `decay_function` | `str` | `'asymptotic_decay'` | Learning-rate decay: `'asymptotic_decay'`, `'inverse_decay_to_zero'` or `'linear_decay_to_zero'`. |
| `neighborhood_function` | `str` | `'gaussian'` | `'gaussian'`, `'mexican_hat'`, `'bubble'` or `'triangle'`. |
| `topology` | `str` | `'rectangular'` | `'rectangular'` or `'hexagonal'`. |
| `random_seed` | `int` | `42` | Seed of the `numpy.random.RandomState` used for initialization and sample shuffling (same draws as `minisom`). |
| `sigma_decay_function` | `str` | `'asymptotic_decay'` | Sigma decay: `'asymptotic_decay'`, `'inverse_decay_to_one'` or `'linear_decay_to_one'`. |

**Methods**

| Method | Description |
| :--- | :--- |
| `random_weights_init(data)` / `pca_weights_init(data)` | Initializes the weights from random samples / the first two principal components. |
| `train(data, num_iters, n_jobs=-1, algorithm='online', random_order=False, use_epochs=False)` | Trains on `data`, shape `(Samples, Features)`, continuing from the current weights. `algorithm='online'` reproduces `MiniSom.train` (`num_iters` single-sample updates, or epochs with `use_epochs=True`); `algorithm='batch'` reproduces `MiniSom.train_batch_offline` (`num_iters` passes over the whole dataset, parallelized with `n_jobs`). |
| `predict(data, n_jobs=-1)` | Returns the Best Matching Unit (BMU) flat index `i * y + j` for each sample in `data`. |
| `winner(x)` / `quantization(data)` / `quantization_error(data)` | BMU coordinates of one sample / BMU codebook vector of each sample / mean sample-to-BMU distance. |
| `get_weights()` | Codebook of shape `(x, y, input_len)`. |
| `filter_noisy_samples(data, labels, n_jobs=-1)` | Returns a boolean mask flagging samples whose label disagrees with their neuron's majority label — useful for cleaning noisy training sets before supervised classification. |

**Usage Example**

```python
from cdts.ai import SOM

# Flatten cube to (Pixels, Features)
X_train = cube_16d.values.reshape(-1, cube_16d.shape[2] * cube_16d.shape[3])

# Train a 10x10 SOM grid (Batch SOM, 20 passes over the data, OpenMP-parallel)
som = SOM(x=10, y=10, input_len=X_train.shape[1], sigma=1.5)
som.random_weights_init(X_train)
som.train(X_train, num_iters=20, algorithm="batch", n_jobs=-1)

# Predict Best Matching Units (BMUs) for new data
bmus = som.predict(X_train, n_jobs=-1)
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

U-Net with Temporal Attention Encoder, for multi-temporal, multi-spectral satellite imagery *segmentation*. Ported layer-for-layer from the official reference implementation ([VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps), MIT License) of Garnot & Landrieu (2021), "Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks", [doi:10.1109/ICCV48922.2021.00483](https://doi.org/10.1109/ICCV48922.2021.00483). A multi-scale U-Net encodes each frame independently (weights shared across time), an L-TAE fuses the bottleneck across time into a feature map plus per-head attention maps, and those same attention maps (resampled per scale) weight the temporal aggregation of every decoder skip connection.

**Verified against the official implementation:** building both with identical weights (`load_state_dict`, keys match by name — no translation table needed) and the same input reproduces its output **bit-for-bit exactly** (`max abs diff = 0.0`), including the padded-sequence code path (`pad_value`/`pad_mask`, for variable-length/irregularly-sampled series).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `input_dim` | `int` | **Required** | Number of input spectral bands. |
| `encoder_widths` | `list[int]` | `[64, 64, 64, 128]` | Channel widths of the encoder stages, top (highest resolution) to bottom. Also sets the number of downsampling steps (`len - 1`). |
| `decoder_widths` | `list[int]` | `[32, 32, 64, 128]` | Same, for the decoder. Must be the same length as `encoder_widths`, and its last element must equal `encoder_widths[-1]`. |
| `out_conv` | `list[int]` | `[32, 20]` | Channel widths of the final output conv stack; the last value is the number of output classes. |
| `str_conv_k`, `str_conv_s`, `str_conv_p` | `int` | `4`, `2`, `1` | Kernel size, stride, and padding of the strided up/down convolutions. |
| `agg_mode` | `str` | `'att_group'` | Skip-connection temporal aggregation: `'att_group'` (attention-weighted, grouped by head — default), `'att_mean'` (attention-weighted, averaged across heads), or `'mean'` (plain temporal average excluding padded dates). |
| `encoder_norm` | `str` | `'group'` | Normalization in the encoder: `'group'` (GroupNorm), `'batch'`, or `'instance'`. |
| `n_head` | `int` | `16` | Attention heads in the bottleneck L-TAE. |
| `d_model` | `int` | `256` | L-TAE's internal projection width (must be divisible by `n_head`). |
| `d_k` | `int` | `4` | L-TAE's key/query dimension per head. |
| `encoder` | `bool` | `False` | If `True`, return `(features, feature_maps)` instead of class scores. |
| `return_maps` | `bool` | `False` | If `True`, also return the list of per-scale decoder feature maps. |
| `pad_value` | `float` | `0` | Value used to mark padded (missing) timesteps — frames entirely equal to this are skipped in the encoder and excluded from temporal aggregation. |
| `padding_mode` | `str` | `'reflect'` | Spatial padding mode passed to the conv layers. |

**Usage Example**

```python
import torch
from cdts.ai import UTAE

model = UTAE(input_dim=6, out_conv=[32, 10])
model.eval()

# (Batch, Time, Bands, H, W), plus per-sample/per-timestep acquisition dates (Batch, Time)
X = torch.randn(2, 12, 6, 128, 128)
batch_positions = torch.arange(12, dtype=torch.float32).unsqueeze(0).expand(2, -1)

with torch.no_grad():
    predictions = model(X, batch_positions=batch_positions)  # (2, 10, 128, 128)
```

### `cdts.ai.LTAE`

Lightweight Temporal Attention Encoder (L-TAE) — the reusable temporal-fusion block from Garnot & Landrieu (2020), [doi:10.48550/arXiv.2007.00586](https://arxiv.org/abs/2007.00586). Ported layer-for-layer from `sits`'s `.torch_light_temporal_attention_encoder` (`R/api_torch_psetae.R`), so trained weights are portable between the two: `LayerNorm -> Conv1d(1x1) -> LayerNorm -> sinusoidal positional encoding -> multi-head attention with a single learned "master query" per head (not computed from the input, unlike standard self-attention) -> MLP decoder -> Dropout -> LayerNorm`. Takes a `(batch, seq_len, in_channels)` sequence and returns a `(batch, n_neurons[-1])` fused embedding.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `in_channels` | `int` | `128` | Input feature dimension per timestep. |
| `day_offsets` | `list[float]` | **Required** | The fixed timeline (day counts from the first observation) this instance is built for — like `n_times` in `TempCNN`, an `LTAE` instance is tied to one sequence length for its lifetime (matches `sits`'s `timeline` parameter). |
| `n_heads` | `int` | `16` | Number of attention heads. |
| `n_neurons` | `tuple[int,...]` | `(256, 128)` | Width of the internal 1x1 conv projection (`n_neurons[0]`, the attention `d_model`) followed by the decoder MLP's hidden dims. |
| `dropout_rate` | `float` | `0.2` | Dropout rate before the final `LayerNorm`. |

**Usage Example**

```python
from cdts.ai import LTAE

day_offsets = list(range(0, 36 * 16, 16))  # 36 steps, 16-day composites
ltae = LTAE(in_channels=128, day_offsets=day_offsets)

x = torch.randn(4, 36, 128)  # (batch, seq_len, in_channels)
fused = ltae(x)  # (4, 128)
```

### `cdts.ai.LightTAE`

The full pixel-level time-series classifier built around `LTAE`: `PixelSpatialEncoder (per-pixel MLP) -> LTAE -> MLP decoder to class logits`. Ported layer-for-layer from `sits`'s `sits_lighttae()` (`R/sits_lighttae.R`) — this is the model directly comparable to a trained `sits_lighttae()` output, unlike the bare `LTAE` block above. Verified against `sits`: building both models with identical weights and the same input reproduces `sits`'s output within float32 tolerance (max abs diff ~1.2e-7).

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `n_bands` | `int` | **Required** | Number of spectral bands per pixel. |
| `day_offsets` | `list[float]` | **Required** | Fixed timeline (day counts from the first observation) — see `LTAE` above. |
| `n_labels` | `int` | **Required** | Number of output classes. |
| `layers_spatial_encoder` | `tuple[int,...]` | `(32, 64, 128)` | Widths of the per-pixel MLP spatial encoder. |
| `n_heads` | `int` | `16` | Attention heads, passed through to `LTAE`. |
| `n_neurons` | `tuple[int,...]` | `(256, 128)` | Passed through to `LTAE`. |
| `dropout_rate` | `float` | `0.2` | Passed through to `LTAE`. |
| `dim_input_decoder` | `int` | `128` | Input width of the decoder MLP (must match `n_neurons[-1]`). |
| `dim_layers_decoder` | `tuple[int,...]` | `(64, 32)` | Decoder MLP hidden dims; `n_labels` is appended as the final layer automatically. |

**Usage Example**

```python
from cdts.ai import LightTAE

day_offsets = list(range(0, 36 * 16, 16))
model = LightTAE(n_bands=6, day_offsets=day_offsets, n_labels=5)

x = torch.randn(8, 36, 6)  # (batch, n_times, n_bands)
logits = model(x)  # (8, 5)
```

### `cdts.ai.TempCNN`

A 1D Temporal Convolutional Neural Network for classifying satellite time-series at the pixel level (Pelletier *et al.*, 2019, [doi:10.3390/rs11050523](https://doi.org/10.3390/rs11050523)). Ported layer-for-layer from the R package [`sits`](https://github.com/e-sensing/sits)'s `sits_tempcnn()` (`R/sits_tempcnn.R`, `R/api_torch.R`) so trained weights are portable between the two for cross-validation: 3x `(Conv1d -> BatchNorm1d -> ReLU -> Dropout)`, then a **flatten** over the full time axis (not global-average-pooled — the flatten bakes `n_times` into the dense layer's input size, so a given model instance is tied to one fixed sequence length, matching `sits`'s behavior), then `(Linear -> BatchNorm1d -> ReLU -> Dropout)` and a final `Linear` classifier.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `in_channels` | `int` | **Required** | Number of spectral bands. |
| `n_times` | `int` | **Required** | Number of timesteps in the sequence (fixed per model instance — see above). |
| `num_classes`| `int` | `5` | Number of output classification categories. |
| `hidden_dims`| `tuple[int,int,int]` | `(64, 64, 64)` | Number of filters in each of the 3 convolutional blocks. |
| `kernel_sizes`| `tuple[int,int,int]` | `(3, 3, 3)` | Kernel size of each 1D convolution (matches `sits_tempcnn`'s `cnn_kernels` default). |
| `dropout_rates`| `tuple[float,float,float]` | `(0.2, 0.2, 0.2)` | Dropout rate after each convolutional block (matches `sits_tempcnn`'s `cnn_dropout_rates` default). |
| `dense_layer_nodes` | `int` | `256` | Width of the dense layer between the flattened conv output and the classifier. |
| `dense_layer_dropout_rate` | `float` | `0.5` | Dropout rate on the dense layer. |

**Usage Example**

```python
import torch
from cdts.ai import TempCNN

model = TempCNN(in_channels=6, n_times=36, num_classes=5)

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
| `backbone` | `str` | `'resnet18'` | The CNN backbone architecture. Options: `'resnet18'`, `'resnet34'`, or `'unet'`. |
| `pretrained` | `bool` | `True` | Load pre-trained ImageNet weights (adapts first layer). |
| `distance_metric`| `str`| `'euclidean'`| Metric used to compare features. Options: `'euclidean'` or `'cosine'`. |

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
| `task` | `str` | `'segmentation'`| Fine-tuning head objective. Options: `'segmentation'` or `'classification'`. |

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


## Xarray Accessor Utilities

CDTS registers an Xarray accessor under `.cdts` for lazy, Dask-backed execution. Every algorithm's own section above already documents its `DataArray.cdts.run_...(...)` accessor form alongside the plain array/Dask-array entry point (look for "Also available as..." in each section) — this section covers the one accessor method that has no other home.

### `DataArray.cdts.to_zarr_optimized`

Rechunks a `DataArray` to sensible spatial tile sizes and writes it to a [Zarr](https://zarr.dev/) store with consolidated metadata — useful as a final step after a pixel-wise algorithm (LandTrendr, CCDC, Mann-Kendall, BFAST, ...) to get a cloud-friendly, chunk-aligned output ready for repeated partial reads (e.g. from S3/GCS) instead of a single large in-memory array.

**Parameters**

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `store_path` | `str` | **Required** | Path or URL of the Zarr store to write (local path, or a fsspec-compatible URL like `s3://...`). |
| `chunk_size` | `dict` | `{"y": 512, "x": 512}` | Target chunk sizes per spatial dimension. |

**Usage Example**

```python
# result: a (metric, y, x) DataArray from any cdts.cdts.run_...() accessor
result.cdts.to_zarr_optimized("output/trend_result.zarr", chunk_size={"y": 512, "x": 512})
```
