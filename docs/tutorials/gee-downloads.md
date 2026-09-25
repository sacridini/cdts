# Downloading Time Series via Google Earth Engine (GEE)

The `cdts` package now offers native support for extracting and downloading time series data using **Google Earth Engine (GEE)**. This allows you to skip downloading individual, uncalibrated images and instead let Google process complex data (such as Landsat sensor harmonization, cloud masking, and annual compositing) directly on their servers before downloading.

## Authentication

Earth Engine requires the user to authenticate their machine with Google Cloud Project (GCP) credentials that have API access enabled. Every time you attempt a download, the `download_gee_timeseries` function handles this initialization. If your credentials expire or do not exist, a browser window will open prompting you to log in.

> **Note:** Make sure the selected Google account has access to GEE. It is recommended to provide your GCP project name using the `project='your-project'` parameter.

## Example 1: Automatic Download (`method='auto'`, default)

For most areas you don't need to choose a method. With the default `method='auto'`, `cdts` plans each image before downloading it and picks the fastest route that will work:

- **Direct tiled download** for images up to `max_direct_mb` (4 GB raw by default). `cdts` fixes a single pixel grid for the whole region, splits it into tiles sized in *bytes* rather than degrees, and fetches them concurrently with `ee.data.computePixels`. Each tile is written straight into its place in the output GeoTIFF, so tiles line up exactly with no seams and there is no mosaicking step. Tiles are split in space first; deep stacks such as a `'dense'` time series with hundreds of bands are also split by band, so they stay under Earth Engine's per-request limits (32 MB, 1024 bands).
- **Google Drive export** (see Example 2) for images above that size, or when a tile hits an Earth Engine *interactive* compute limit (user memory limit, computation timeout), which retrying the same request cannot fix.

Concurrency adapts to your account. Earth Engine limits concurrent interactive requests per account (about 40 on a standard tier, only 2–3 for a project in noncommercial *Restricted Mode*). `cdts` starts with 4 concurrent requests, ramps up while requests succeed, and halves on every `HTTP 429`, so it settles just under whatever limit your account actually has.

```python
from cdts.gee import download_gee_timeseries

# Bounding box [min_lon, min_lat, max_lon, max_lat]
my_roi = [-47.95, -15.85, -47.85, -15.75]

download_gee_timeseries(
    roi=my_roi,
    start_date='2010-01-01',
    end_date='2020-12-31',
    out_dir='./gee_direct_data',
    method='auto',             # default: direct tiled download, Drive export when needed
    composite_type='annual',   # Generates LandTrendr-style Annual Medoid Composites
    project='my-gcp-project'   # Replace with your Google Cloud Project ID
)
```

To download a single `ee.Image` you built yourself, use `download_gee_image` directly. It accepts the same `method` and exposes the tuning knobs:

```python
from cdts.gee.downloader import download_gee_image

download_gee_image(
    image, roi, 'out.tif',
    scale=30,
    crs='EPSG:4326',           # or a projected CRS such as 'EPSG:32723'
    sub_tile_workers=16,       # upper bound on concurrent requests (adapts downward on 429)
    max_tile_mb=None,          # default: about one round of requests, 4-32 MB per tile
)
```

Masked pixels are written with the GeoTIFF nodata value Earth Engine uses for the output type (`-inf` for float, the type minimum for signed integers, `0` for unsigned). You can force either route with `method='direct'` or `method='drive'`.

## Example 2: Exporting to Google Drive (Large Areas)

For state-level or national-scale analyses, downloading data directly over the internet in real-time might fail due to API payload limits or simply take too long.

In these cases, pass `method='drive'`. The `cdts` package will set up everything and dispatch a Task directly to Google's servers. Google will silently process and save the final file in the cloud inside your **Google Drive**, under the `CDTS_Downloads` folder.

```python
from cdts.gee import download_gee_timeseries

# Example: Bounding box of a larger region
state_roi = [-53.11, -25.31, -44.15, -19.78]

download_gee_timeseries(
    roi=state_roi, 
    start_date='1985-01-01',
    end_date='2022-12-31', 
    out_dir='./data',  # Used only to name the files in Drive for this method
    method='drive',    # Initiates asynchronous export
    composite_type='annual',
    project='my-gcp-project'
)

# The terminal will display a message similar to:
# [landsat_medoid_1985] Task sent to Google Drive (Task ID: ABCD123456).
```

## What happens under the hood?

When using `composite_type='annual'` (the current `cdts` default for LandTrendr integration):

1. **Sensor Fusion:** The function fetches Landsat 5, 7, 8, and 9 collections (Surface Reflectance Collection 2).
2. **Harmonization:** Values from Landsat 8 and 9 (OLI) are mathematically converted to their ETM+ equivalents using coefficients from Roy et al. (2016) (see [References](#references)). This ensures a perfect time series, free from sensor biases.
3. **Cloud Masking:** The `QA_PIXEL` quality assurance band is used to filter out dense clouds and shadows across all images.
4. **Medoid Compositing:** Instead of a simple median, we apply the Medoid geometric strategy to find the actual real-world pixel that best represents the season (a standard approach in *eMapR/LandTrendr* workflows).

---

## References

- Roy, D. P., Kovalskyy, V., Zhang, H. K., Vermote, E. F., Yan, L., Kumar, S. S., & Egorov, A. (2016). Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity. **Remote Sensing of Environment**, 185, 57–70. [https://doi.org/10.1016/j.rse.2015.12.024](https://doi.org/10.1016/j.rse.2015.12.024)
