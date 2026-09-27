# Data & I/O

<p class="lead">Build data cubes from cloud catalogs, Earth Engine or local files; read and write GeoTIFFs; composite to a regular time step; and turn QA bands into observation weights.</p>

## Building cubes

### `build_time_series` { .api }

<!-- sig: zeit.cube.build_time_series -->
```python
zeit.cube.build_time_series(
    source="earth_search", collection="sentinel-2-l2a", bbox=None,
    vector_path=None, tiles=None, start_date="2020-01-01",
    end_date="2020-12-31", cloud_cover_max=30, bands=None,
    apply_cloud_mask=False, resolution=None, epsg=4326,
    validate_items=False, access_token=None,
)
```

Builds a lazy, Dask-backed `xarray.DataArray` from a STAC catalog. It searches the catalog, keeps the items matching the area, dates and cloud-cover limit, and stacks them into an aligned cube shaped `(time, band, y, x)`. Nothing is downloaded until the cube is computed. Also exported as `zeit.build_time_series`. Tutorial: [STAC Data Cubes](../tutorials/stac-downloads.md).

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
import zeit

cube = zeit.build_time_series(
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

<!-- sig: zeit.local.build_local_cube -->
```python
zeit.local.build_local_cube(
    data_dir, regex_pattern, date_format="%Y%m%d",
)
```

Builds the same kind of lazy cube from a folder of GeoTIFFs, reading the date and band of each file from its name. Also exported as `zeit.build_local_cube`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_dir` | `str` | required | Folder containing the `.tif` files. |
| `regex_pattern` | `str` | required | Regular expression with a named group `(?P<date>...)` and, optionally, `(?P<band>...)`. |
| `date_format` | `str` | `"%Y%m%d"` | `strptime` format of the captured date. |

</div>

```python
cube = zeit.build_local_cube(
    "/data/tiles",
    regex_pattern=r".*_(?P<date>\d{8})_(?P<band>B\d{2})\.tif",
)
```

### `download_gee_timeseries` { .api }

<!-- sig: zeit.gee.download_gee_timeseries -->
```python
zeit.gee.download_gee_timeseries(
    roi, start_date, end_date, out_dir, method="auto",
    composite_type="annual", bands=None, project=None,
)
```

Builds harmonised Landsat 5/7/8/9 composites on Google Earth Engine and downloads one GeoTIFF per composite. By default (`method="auto"`) it uses a concurrent tiled direct download and falls back to a Google Drive export for very large images or ones that hit Earth Engine's interactive compute limits. Tutorial: [Google Earth Engine](../tutorials/gee-downloads.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `roi` | `str`, `list`, file path, GeoDataFrame or `ee.Geometry` | required | A Landsat WRS-2 path/row (`"217/076"`), a Sentinel-2 tile (`"23KPQ"`), a bbox `[min_lon, min_lat, max_lon, max_lat]`, a vector or raster file, a GeoDataFrame / shapely geometry, or an `ee.Geometry`. Local inputs are read offline; their bounding box is downloaded. |
| `start_date`, `end_date` | `str` | required | Date range, `YYYY-MM-DD`. |
| `out_dir` | `str` | required | Output folder (for `"drive"`, used to name the exports). |
| `method` | `str` | `"auto"` | `"auto"` (direct, Drive when needed), or force `"direct"` / `"drive"`. |
| `composite_type` | `str` | `"annual"` | `"annual"` medoid composites (LandTrendr) or `"dense"` (every observation, CCDC). |
| `bands` | `list` | `None` | Bands or indices to export (`"SR_B4"`, `"NDVI"`, `"NBR"`, `"EVI"`, `"NDWI"`, `"kNDVI"`). Default: the six reflective bands. |
| `project` | `str` | `None` | Google Cloud project used to initialise Earth Engine. Recommended. |

</div>

```python
from zeit.gee import download_gee_timeseries

download_gee_timeseries(
    roi="217/076",                      # WRS-2 path/row; or a bbox, .shp, .gpkg, .tif ...
    start_date="1985-01-01", end_date="2025-12-31",
    out_dir="./gee_data", composite_type="annual",
    bands=["NBR"], project="my-gcp-project",
)
```

Masked pixels are written as `-inf` (float outputs). Convert them to `NaN` before running LandTrendr, see the [worked example](../tutorials/gee-downloads.md#worked-example-a-full-landsat-tile-19852025-ready-for-landtrendr).

### `download_gee_image` { .api }

<!-- sig: zeit.gee.downloader.download_gee_image -->
```python
zeit.gee.downloader.download_gee_image(
    image, roi, out_filename, method="auto", scale=30, tile_size=None,
    sub_tile_workers=16, crs="EPSG:4326", max_tile_mb=None,
    max_direct_mb=4096, max_retries=5, base_backoff=5.0,
)
```

Downloads one `ee.Image` to a local GeoTIFF by the fastest route that works. This is what `download_gee_timeseries` calls for each composite; use it directly for images you build yourself or to tune the download. The direct route fixes one pixel grid for the whole ROI and fetches it as tiles with concurrent `ee.data.computePixels` calls, writing each tile into its window of the output file (no temporary tiles, no mosaicking). Concurrency adapts to `HTTP 429` responses.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `ee.Image` | required | The image to download. |
| `roi` | `ee.Geometry` | required | Region whose bounding box is downloaded. Build it with `resolve_roi`. |
| `out_filename` | `str` | required | Output GeoTIFF. Never left partially written. |
| `method` | `str` | `"auto"` | `"auto"`, `"direct"` or `"drive"`. |
| `scale` | `float` | `30` | Pixel size in metres (converted to degrees at the equator for a geographic CRS, as Earth Engine does). |
| `tile_size` | `float` | `None` | Deprecated and ignored. |
| `sub_tile_workers` | `int` | `16` | Upper bound on concurrent requests. Starts at 4 and adapts to the account's limit (about 40 on a standard tier, about 2 in Restricted Mode). |
| `crs` | `str` | `"EPSG:4326"` | Output CRS, e.g. `"EPSG:32723"`. |
| `max_tile_mb` | `float` | `None` | Cap on one tile request, in MB (at most 32, the default). |
| `max_direct_mb` | `float` | `4096` | With `"auto"`, larger images (raw size) go to a Drive export. |
| `max_retries`, `base_backoff` | | `5`, `5.0` | Retry policy for network or server errors. Throttled requests get short jittered retries instead. |

</div>

**Returns** the output path, or `None` if the download failed (nothing is written in that case).

```python
from zeit.gee.auth import initialize_gee
from zeit.gee.roi import resolve_roi
from zeit.gee.harmonization import get_harmonized_collection
from zeit.gee.composites import create_annual_medoid
from zeit.gee.downloader import download_gee_image

initialize_gee(project="my-gcp-project")
roi = resolve_roi("data/study_area.gpkg")       # or "217/076", "23KPQ", a bbox ...
col = get_harmonized_collection(roi, "1985-01-01", "2025-12-31")

for year in range(1985, 2026):
    img = create_annual_medoid(col, year)
    img = img.normalizedDifference(["SR_B5", "SR_B7"]).rename("NBR").toFloat()
    download_gee_image(img, roi, f"nbr_{year}.tif", crs="EPSG:32723", sub_tile_workers=32)
```

### `resolve_roi` { .api }

<!-- sig: zeit.gee.roi.resolve_roi -->
```python
zeit.gee.roi.resolve_roi(roi)
```

Turns any supported area description into the `ee.Geometry` the Earth Engine functions need, so your code never builds Earth Engine objects. Local inputs are reduced to their bounding box.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `roi` | see below | required | Area description. |

</div>

| Input | Example | Result |
| :--- | :--- | :--- |
| WRS-2 path/row | `"217/076"`, `"217_076"`, `"217076"` | Tile footprint, looked up from a Landsat Collection 2 scene |
| Sentinel-2 MGRS tile | `"23KPQ"`, `"T23KPQ"` | 109.8 km tile computed offline (within ~50 m of real footprints). Tiles crossing 180° raise `ValueError` |
| Bounding box | `[-43.6, -23.1, -43.1, -22.6]` | That box (lon/lat) |
| Vector file | `.shp`, `.gpkg`, `.geojson`, `.kml` | Extent of all features, reprojected to lon/lat |
| Raster file | `reference.tif` | Raster extent, reprojected to lon/lat |
| GeoDataFrame / GeoSeries | `geopandas.read_file(...)` | Extent, reprojected (no CRS: lon/lat assumed, with a warning) |
| shapely geometry | `box(...)` | Its bounds (assumed lon/lat) |
| `ee.Geometry` | | Passed through |

Sentinel-1 has no fixed tiling grid, so describe Sentinel-1 areas with any of the other inputs. Two offline helpers live in the same module: `zeit.gee.roi.roi_bounds(roi)` returns the lon/lat bounding box of a local input or a Sentinel-2 tile, and `zeit.gee.roi.s2_tile_utm_bounds("23KPQ")` returns a tile's exact UTM box, e.g. `("EPSG:32723", (600000, 7390200, 709800, 7500000))`.

## Compositing

### `regularize_time_series` { .api }

<!-- sig: zeit.regularize.regularize_time_series -->
```python
zeit.regularize.regularize_time_series(
    cube, freq="16D", method="median",
)
```

Composites an irregular cube to a fixed time step (16-day, monthly, yearly) with the median or the medoid of each window. Stays lazy. Also exported as `zeit.regularize_time_series`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `xr.DataArray` | required | Cube with a `time` dimension. `"medoid"` also needs a `band` dimension. |
| `freq` | `str` | `"16D"` | pandas frequency: `"16D"`, `"1MS"`, `"1YS"`, … |
| `method` | `str` | `"median"` | `"median"` (per band) or `"medoid"` (the real observation closest to the multi-band median). |

</div>

```python
cube_16d = zeit.regularize_time_series(cube, freq="16D", method="medoid")
```

## Reading and writing rasters

### `load_raster` { .api }

<!-- sig: zeit.io.load_raster -->
```python
zeit.io.load_raster(file_path, raster_check=None)
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
stack, profile = zeit.io.load_raster("ndvi_1985_2024.tif", raster_check="landtrendr")
```

### `save_raster` { .api }

<!-- sig: zeit.io.save_raster -->
```python
zeit.io.save_raster(
    array, output_path, reference_cube=None, crs="EPSG:4326",
    transform=None, nodata=None,
)
```

Writes a 2-D, 3-D or 4-D NumPy array (or an xarray `DataArray`) as a compressed, tiled GeoTIFF. The georeferencing comes from `reference_cube`, or from `crs` and `transform`. Also exported as `zeit.save_raster`.

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
zeit.save_raster(loss["yod"], "year_of_loss.tif",
                 crs=profile["crs"], transform=profile["transform"], nodata=0)
```

### `get_georef` { .api }

<!-- sig: zeit.io.get_georef -->
```python
zeit.io.get_georef(reference_cube)
```

Extracts `{"crs": ..., "transform": ...}` from a rasterio dataset or an xarray object. Handy for `snic_to_polygons` or custom writers. Also exported as `zeit.get_georef`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `reference_cube` | rasterio dataset or `xr.DataArray` | required | Object to read the georeferencing from. |

</div>

## Quality bands to weights

Decoders ported from `phenofit`'s `qcFUN.R`. Each turns a sensor's QA band into per-observation reliability weights in `[0, 1]` for weighted smoothing (`apply_whittaker_filter`) and phenology (`weights=`). All three are exported at the top level.

### `qc_sentinel2_scl` { .api }

<!-- sig: zeit.qc.qc_sentinel2_scl -->
```python
zeit.qc.qc_sentinel2_scl(scl, wmin=0.2, wmid=0.5, wmax=1.0)
```

Sentinel-2 L2A Scene Classification Layer: vegetation, bare soil, water, unclassified and thin cirrus get `wmax`; medium-probability cloud gets `wmid`; everything else (saturated, shadow, high-probability cloud, snow, no data) gets `wmin`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `scl` | array | required | SCL values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>

### `qc_modis_summary` { .api }

<!-- sig: zeit.qc.qc_modis_summary -->
```python
zeit.qc.qc_modis_summary(qa, wmin=0.2, wmid=0.5, wmax=1.0)
```

MOD13 "SummaryQA" / pixel reliability: `0` good → `wmax`, `1` marginal → `wmid`, `2` snow and `3` cloudy → `wmin`, fill → `0`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qa` | array | required | SummaryQA values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>

```python
from zeit.qc import qc_modis_summary

weights = qc_modis_summary(qa_cube)          # same shape as qa_cube
```

### `qc_modis_state` { .api }

<!-- sig: zeit.qc.qc_modis_state -->
```python
zeit.qc.qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
```

MOD09 500 m 16-bit "State QA": decodes cloud state, cloud shadow, aerosol quantity and snow/ice bits.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qa` | array | required | State QA values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>
