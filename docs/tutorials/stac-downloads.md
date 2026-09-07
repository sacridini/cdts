# STAC & Analysis Ready Data (ARD) Integration

CDTS provides powerful, `sits`-like capabilities for accessing Analysis Ready Data (ARD) from the cloud or local storage. Built on top of `xarray`, `Dask`, and `stackstac`, our data cube engine supports lazy evaluation, automatic semantic cloud masking, multi-dimensional regularization, and direct tile querying.

## Building a Data Cube from STAC

You can build a lazy time series cube from known STAC catalogs (e.g., Earth Search, Planetary Computer, Brazil Data Cube) using `cdts.build_time_series`.

### 1. Spatial Queries (BBox or MGRS Tiles)

You can query data using a standard Bounding Box, a vector shapefile, or explicitly using **MGRS/WRS-2 Tiles**.

```python
import cdts

# Option A: Bounding Box
cube = cdts.build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    bbox=[-48.5, -22.5, -48.0, -22.0],
    start_date="2022-01-01",
    end_date="2022-12-31",
    bands=["red", "green", "blue", "nir"]
)

# Option B: MGRS Tiles (Sentinel-2)
cube_tiles_s2 = cdts.build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    tiles=["22JFQ", "22JGQ"], # Fetch specific Sentinel-2 MGRS tiles
    start_date="2022-01-01",
    end_date="2022-12-31",
    bands=["red", "green", "blue", "nir"]
)

# Option C: WRS-2 Path/Row Tiles (Landsat)
cube_tiles_l8 = cdts.build_time_series(
    source="earth_search",
    collection="landsat-c2-l2",
    tiles=["215065"], # 6-digit Path/Row string (Path 215, Row 065)
    start_date="2022-01-01",
    end_date="2022-12-31",
    bands=["red", "green", "blue", "nir"]
)
```

### 2. Semantic Cloud Masking

CDTS can automatically identify the satellite platform (Sentinel-2, Landsat) and apply semantic cloud masking natively before returning the cube. Just pass `apply_cloud_mask=True`. 
This will automatically download the respective Quality Assurance (QA) band (like `scl` for Sentinel) and mask out clouds, shadows, and cirrus.

```python
cube_clean = cdts.build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    tiles=["22JFQ"],
    start_date="2022-01-01",
    end_date="2022-12-31",
    bands=["red", "green", "blue", "nir"],
    apply_cloud_mask=True # Clouds are masked out!
)
```

## Temporal Regularization (Medoid & Median)

Raw STAC data usually comes in irregular time steps (e.g., passing every 5, 8, or 12 days). For advanced Machine Learning and TWDTW, you must regularize the cube to fixed temporal steps.

You can use `cdts.regularize_time_series` to composite these observations into regular windows (e.g., 16-day composites) using multi-dimensional `medoid` or `median` strategies. Because it uses `xarray`, this computation remains fully lazy!

```python
from cdts import regularize_time_series

# Create a 16-day Medoid composite
cube_16d = regularize_time_series(cube_clean, freq="16D", method="medoid")

# The output has regular 16-day steps on the 'time' dimension
print(cube_16d.time)
```

## Ingesting Local ARD Cubes

If you have already downloaded ARD TIFF files to your machine, you can ingest them into a lazy CDTS cube by providing a Regular Expression to parse the filenames.

```python
from cdts import build_local_cube

# Assume you have files like: "SENTINEL_20220101_B02.tif"
# The regex must capture (?P<date>...) and (?P<band>...)
cube_local = build_local_cube(
    data_dir="/path/to/my/tiffs",
    regex_pattern=r".*_(?P<date>\d{8})_(?P<band>B\d{2})\.tif",
    date_format="%Y%m%d"
)

# You get a full xarray DataArray ready for TWDTW, SOM, or CCDC!
```

## Calculating Spectral Indices (e.g., NDVI)

If you wish to obtain only a specific spectral index like NDVI, there are two possible scenarios depending on the STAC catalog:

**1. The index is pre-calculated by the provider**
If the catalog (such as Brazil Data Cube) natively provides an `ndvi` asset, you can fetch it directly without downloading the raw optical bands:
```python
cube_ndvi = cdts.build_time_series(
    source="brazil_data_cube",
    collection="CBERS4A_WFI_L4_SR",
    tiles=["022024"],
    bands=["ndvi"] # Direct index download
)
```

**2. The index is NOT pre-calculated (e.g., Earth Search)**
Standard Level-2A collections typically do not store the index to save space. You must explicitly download the `red` and `nir` bands and calculate the index locally. Because CDTS is built on Dask, this mathematical operation is lazy and virtually memory-free until you save it or plot it.
```python
cube_raw = cdts.build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    tiles=["22JFQ"],
    bands=["red", "nir"]
)

# Compute NDVI lazily
cube_ndvi = (cube_raw.sel(band="nir") - cube_raw.sel(band="red")) / (cube_raw.sel(band="nir") + cube_raw.sel(band="red"))
```
