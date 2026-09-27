# Data & I/O

<p class="lead">Build data cubes from cloud catalogs, Earth Engine or local files; read and write GeoTIFFs; composite to a regular time step; and turn QA bands into observation weights.</p>

## Building cubes

### `build_time_series` { .api }

<!-- sig: cdts.cube.build_time_series -->
```python
cdts.cube.build_time_series(
    source="earth_search", collection="sentinel-2-l2a", bbox=None,
    vector_path=None, tiles=None, start_date="2020-01-01",
    end_date="2020-12-31", cloud_cover_max=30, bands=None,
    apply_cloud_mask=False, resolution=None, epsg=4326,
    validate_items=False, access_token=None,
)
```

Builds a lazy, Dask-backed `xarray.DataArray` from a STAC catalog. It searches the catalog, keeps the items matching the area, dates and cloud-cover limit, and stacks them into an aligned cube shaped `(time, band, y, x)`. Nothing is downloaded until the cube is computed. Also exported as `cdts.build_time_series`. Tutorial: [STAC Data Cubes](../tutorials/stac-downloads.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `str` | `"earth_search"` | Catalog alias (`"earth_search"`, `"planetary_computer"`, `"brazil_data_cube"`) or any STAC API URL. |
| `collection` | `str` or `list` | `"sentinel-2-l2a"` | Collection ID, e.g. `"sentinel-2-l2a"`, `"landsat-c2-l2"`, `"modis-13Q1-061"`. Available collections depend on `source`. |
| `bbox` | `list` | `None` | `[min_lon, min_lat, max_lon, max_lat]` in EPSG:4326. |
| `vector_path` | `str` | `None` | Vector file (Shapefile, GeoJSON) whose bounds define the area. |
| `tiles` | `list[str]` | `None` | Sentinel-2 MGRS tiles (`"22JFQ"`) or Landsat WRS-2 path/rows (`"215065"`). |
| `start_date`, `end_date` | `str` | `"2020-01-01"`, `"2020-12-31"` | Date range, `YYYY-MM-DD`. |
| `cloud_cover_max` | `int` | `30` | Maximum scene cloud cover, in percent (metadata filter). |
| `bands` | `list[str]` | `None` | Assets to load, e.g. `["red", "nir"]`. |
| `apply_cloud_mask` | `bool` | `False` | Also load the QA band (`scl` for Sentinel-2, `qa_pixel` for Landsat) and mask clouds and shadows. |
| `resolution` | `float` | `None` | Output pixel size, in units of `epsg`. |
| `epsg` | `int` | `4326` | Output coordinate reference system. |
| `validate_items` | `bool` | `False` | Test each asset URL first and drop broken ones. |
| `access_token` | `str` | `None` | Token for catalogs that require one (e.g. Brazil Data Cube). |

</div>

**Returns** a lazy `xarray.DataArray` `(time, band, y, x)`.

```python
import cdts

cube = cdts.build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    bbox=[-48.0, -16.0, -47.9, -15.9],
    start_date="2021-01-01",
    end_date="2021-12-31",
    bands=["red", "nir"],
    apply_cloud_mask=True,
    resolution=10,
    epsg=32722,
)
```

### `build_local_cube` { .api }

<!-- sig: cdts.local.build_local_cube -->
```python
cdts.local.build_local_cube(
    data_dir, regex_pattern, date_format="%Y%m%d",
)
```

Builds the same kind of lazy cube from a folder of GeoTIFFs, reading the date and band of each file from its name. Also exported as `cdts.build_local_cube`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_dir` | `str` | required | Folder containing the `.tif` files. |
| `regex_pattern` | `str` | required | Regular expression with a named group `(?P<date>...)` and, optionally, `(?P<band>...)`. |
| `date_format` | `str` | `"%Y%m%d"` | `strptime` format of the captured date. |

</div>

```python
cube = cdts.build_local_cube(
    "/data/tiles",
    regex_pattern=r".*_(?P<date>\d{8})_(?P<band>B\d{2})\.tif",
)
```

### `download_gee_timeseries` { .api }

<!-- sig: cdts.gee.download_gee_timeseries -->
```python
cdts.gee.download_gee_timeseries(
    roi, start_date, end_date, out_dir, method="direct",
    composite_type="annual", bands=None, project=None,
)
```

Builds harmonised Landsat 5/7/8/9 composites on Google Earth Engine and downloads them as GeoTIFFs, either directly (tiled, multi-threaded) or through a Google Drive export for large areas. Tutorial: [Google Earth Engine](../tutorials/gee-downloads.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `roi` | `list` or `ee.Geometry` | required | `[min_lon, min_lat, max_lon, max_lat]` or an Earth Engine geometry. |
| `start_date`, `end_date` | `str` | required | Date range, `YYYY-MM-DD`. |
| `out_dir` | `str` | required | Output folder (for `"drive"`, used to name the exports). |
| `method` | `str` | `"direct"` | `"direct"` to download now, `"drive"` to export to Google Drive. |
| `composite_type` | `str` | `"annual"` | `"annual"` medoid composites (LandTrendr) or `"dense"` (every observation, CCDC). |
| `bands` | `list` | `None` | Bands or indices to export (`"SR_B4"`, `"NDVI"`, `"NBR"`, `"EVI"`, `"NDWI"`, `"kNDVI"`). Default: the six reflective bands. |
| `project` | `str` | `None` | Google Cloud project used to initialise Earth Engine. Recommended. |

</div>

```python
from cdts.gee import download_gee_timeseries

download_gee_timeseries(
    roi=[-47.95, -15.85, -47.85, -15.75],
    start_date="1985-01-01", end_date="2024-12-31",
    out_dir="./gee_data", method="direct",
    composite_type="annual", bands=["NBR"], project="my-gcp-project",
)
```

## Compositing

### `regularize_time_series` { .api }

<!-- sig: cdts.regularize.regularize_time_series -->
```python
cdts.regularize.regularize_time_series(
    cube, freq="16D", method="median",
)
```

Composites an irregular cube to a fixed time step (16-day, monthly, yearly) with the median or the medoid of each window. Stays lazy. Also exported as `cdts.regularize_time_series`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `xr.DataArray` | required | Cube with a `time` dimension. `"medoid"` also needs a `band` dimension. |
| `freq` | `str` | `"16D"` | pandas frequency: `"16D"`, `"1MS"`, `"1YS"`, … |
| `method` | `str` | `"median"` | `"median"` (per band) or `"medoid"` (the real observation closest to the multi-band median). |

</div>

```python
cube_16d = cdts.regularize_time_series(cube, freq="16D", method="medoid")
```

## Reading and writing rasters

### `load_raster` { .api }

<!-- sig: cdts.io.load_raster -->
```python
cdts.io.load_raster(file_path, raster_check=None)
```

Reads a GeoTIFF into a NumPy array `(bands, rows, cols)` and returns it with its rasterio profile. Optionally checks that the data looks right for an algorithm (enough layers, integer-scaled values) and warns if not.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file_path` | `str` | required | Path to the raster. |
| `raster_check` | `str` | `None` | `"landtrendr"`, `"ccdc"` or `"cold"` to validate the input for that algorithm. |

</div>

**Returns** `(array, profile)`.

```python
stack, profile = cdts.io.load_raster("ndvi_1985_2024.tif", raster_check="landtrendr")
```

### `save_raster` { .api }

<!-- sig: cdts.io.save_raster -->
```python
cdts.io.save_raster(
    array, output_path, reference_cube=None, crs="EPSG:4326",
    transform=None, nodata=None,
)
```

Writes a 2-D, 3-D or 4-D NumPy array (or an xarray `DataArray`) as a compressed, tiled GeoTIFF. The georeferencing comes from `reference_cube`, or from `crs` and `transform`. Also exported as `cdts.save_raster`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `array` | `np.ndarray` or `xr.DataArray` | required | Data to write. |
| `output_path` | `str` | required | Output file. |
| `reference_cube` | rasterio dataset or `xr.DataArray` | `None` | Object to copy the CRS and transform from. |
| `crs` | `str` | `"EPSG:4326"` | Coordinate reference system, when no `reference_cube` is given. |
| `transform` | `Affine` | `None` | Geotransform, when no `reference_cube` is given. |
| `nodata` | `float` | `None` | NoData value written to the file. |

</div>

```python
cdts.save_raster(loss["yod"], "year_of_loss.tif",
                 crs=profile["crs"], transform=profile["transform"], nodata=0)
```

### `get_georef` { .api }

<!-- sig: cdts.io.get_georef -->
```python
cdts.io.get_georef(reference_cube)
```

Extracts `{"crs": ..., "transform": ...}` from a rasterio dataset or an xarray object. Handy for `snic_to_polygons` or custom writers. Also exported as `cdts.get_georef`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `reference_cube` | rasterio dataset or `xr.DataArray` | required | Object to read the georeferencing from. |

</div>

## Quality bands to weights

Decoders ported from `phenofit`'s `qcFUN.R`. Each turns a sensor's QA band into per-observation reliability weights in `[0, 1]` for weighted smoothing (`apply_whittaker_filter`) and phenology (`weights=`). All three are exported at the top level.

### `qc_sentinel2_scl` { .api }

<!-- sig: cdts.qc.qc_sentinel2_scl -->
```python
cdts.qc.qc_sentinel2_scl(scl, wmin=0.2, wmid=0.5, wmax=1.0)
```

Sentinel-2 L2A Scene Classification Layer: vegetation, bare soil, water, unclassified and thin cirrus get `wmax`; medium-probability cloud gets `wmid`; everything else (saturated, shadow, high-probability cloud, snow, no data) gets `wmin`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `scl` | array | required | SCL values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>

### `qc_modis_summary` { .api }

<!-- sig: cdts.qc.qc_modis_summary -->
```python
cdts.qc.qc_modis_summary(qa, wmin=0.2, wmid=0.5, wmax=1.0)
```

MOD13 "SummaryQA" / pixel reliability: `0` good → `wmax`, `1` marginal → `wmid`, `2` snow and `3` cloudy → `wmin`, fill → `0`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qa` | array | required | SummaryQA values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>

```python
from cdts.qc import qc_modis_summary

weights = qc_modis_summary(qa_cube)          # same shape as qa_cube
```

### `qc_modis_state` { .api }

<!-- sig: cdts.qc.qc_modis_state -->
```python
cdts.qc.qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
```

MOD09 500 m 16-bit "State QA": decodes cloud state, cloud shadow, aerosol quantity and snow/ice bits.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qa` | array | required | State QA values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>
