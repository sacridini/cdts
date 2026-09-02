# STAC Downloads & Best Practices

When working with massive Earth Observation time series, downloading data directly from cloud catalogs via STAC (SpatioTemporal Asset Catalog) is the most efficient approach. CDTS provides the `build_time_series` function to connect to these catalogs and lazily build data cubes using `stackstac` and `dask`.

However, downloading gigabytes (or terabytes) of imagery over decades introduces unique infrastructure challenges. This guide covers the most common pitfalls and the "tricks and tips" to ensure stable, fail-proof downloads.

## 1. The Coordinate Reference System (CRS) & Resolution Trap

A very common mistake when downloading data is mixing CRS units with resolution units. 

If you use the default WGS84 coordinates (`epsg=4326`), the unit of measurement is **Degrees**. If you pass `resolution=30` hoping for 30-meter pixels, the STAC engine will attempt to create pixels that are **30 degrees wide** (almost the size of a continent). This will immediately crash the engine or produce a 1x1 pixel image.

**Tip:** Always match the resolution to the EPSG unit:
- **For `epsg=4326` (Degrees):** Use a resolution like `0.00027` (which roughly equals 30 meters).
- **For UTM Projections (Meters):** Set your local UTM `epsg` (e.g., `epsg=31980` for SIRGAS 2000 UTM 20S) and then you can safely use `resolution=30`.

## 2. Demystifying the `RasterioIOError: not recognized as being in a supported file format`

When downloading large cubes (e.g., 40+ GB) using Dask, you might encounter a crash after a few minutes with an error resembling:

```text
CPLE_OpenFailedError: '/vsicurl/https://...QA_PIXEL.TIF' not recognized as being in a supported file format.
```

**What is actually happening?**
This is rarely a file format issue. It is almost always a **Connection, Rate Limit, or Token Expiration** disguised as a format error:
1. When Dask opens hundreds of parallel connections to fetch image chunks, cloud providers (like Microsoft Azure) may flag your IP for making "Too Many Requests" (HTTP 429), or the temporary security token (SAS Token) in the URL may expire.
2. The server blocks the request and returns an XML or HTML error page (`<Error>Access Denied</Error>`).
3. `Rasterio` receives this text page instead of the expected `.tif` binary. It fails to parse it as an image and throws the "unsupported format" error.

**How CDTS solves this natively:**
Since `v0.1.0`, CDTS automatically performs a **Pre-flight Validation** if you pass `validate_items=True`. It runs a fast, multi-threaded test across all URLs to filter out corrupted or inaccessible files *before* attempting the massive download. Furthermore, CDTS injects `errors_as_nodata=(Exception,)` deeply into the engine, ensuring that if a random server timeout occurs mid-download, it silently fills that chunk with `NoData` (NaN) instead of crashing the entire 40GB operation.

## 3. Controlling Dask's Parallelism

Even with error suppression, hammering a public API with a default Dask cluster can lead to IP bans or severe rate limiting. If you are prioritizing stability over raw speed—especially for overnight downloads—you can force Dask to download sequentially.

**Tip:** Wrap your `save_raster` command in a synchronous context. It will take longer, but it will never fail due to parallel throttling:

```python
import dask
from cdts.io import save_raster

with dask.config.set(scheduler='synchronous'):
    save_raster(cube, 'output.tif', reference_cube=cube, nodata=0)
```

## 4. Choosing the Right STAC Catalog

The choice of catalog dictates the infrastructure rules you must follow. CDTS supports major catalogs out of the box (`earth_search`, `planetary_computer`, and `brazil_data_cube`).

### Microsoft Planetary Computer (`planetary_computer`)
- **Pros:** Completely free, no account required, massive global archive.
- **Cons:** Uses SAS tokens that expire. Extremely large, long-running downloads might require complex token-refresh logic or fail mid-way if they take over an hour.

### Amazon AWS Earth Search (`earth_search`)
- **Pros:** Extremely fast, no expiring tokens, highly stable.
- **Cons:** Landsat Collection 2 on AWS is configured as *Requester Pays*. You **must** have an AWS account and local credentials configured (`~/.aws/credentials`), otherwise it will instantly fail with an `AWSInvalidCredentialsError`.

### INPE Brazil Data Cube (`brazil_data_cube`)
- **Pros:** The holy grail for Brazilian territory. Highly stable, publicly accessible URLs without expiring tokens, and no AWS billing requirements.
- **Cons:** Regional coverage (South America).
- **Tip:** For Landsat data via BDC, use `collection='landsat-2'`.

### Example: The bulletproof Brazil Data Cube download

```python
from cdts.cube import build_time_series
from cdts.io import save_raster
import dask
from dask.diagnostics import ProgressBar

# 1. Fetch metadata from BDC (Token-free!)
cube = build_time_series(
    source='brazil_data_cube',
    collection='landsat-2',
    bbox=[-64.0, -13.0, -60.0, -9.0], # Rondonia Example
    start_date='2020-01-01', 
    end_date='2024-12-31',
    cloud_cover_max=5,
    resolution=0.000269, # ~30m in degrees (matching epsg:4326)
    validate_items=True  # Drops physically corrupted URLs beforehand
)

# 2. Download sequentially to avoid any rate limits
print("Downloading 20GB+ sequentially. Grab a coffee!")
with dask.config.set(scheduler='synchronous'):
    with ProgressBar():
        save_raster(cube, 'CCDC_Stack.tif')
```
