# Downloading Time Series via Google Earth Engine (GEE)

The `cdts` package now offers native support for extracting and downloading time series data using **Google Earth Engine (GEE)**. This allows you to skip downloading individual, uncalibrated images and instead let Google process complex data (such as Landsat sensor harmonization, cloud masking, and annual compositing) directly on their servers before downloading.

## Authentication

Earth Engine requires the user to authenticate their machine with Google Cloud Project (GCP) credentials that have API access enabled. Every time you attempt a download, the `download_gee_timeseries` function handles this initialization. If your credentials expire or do not exist, a browser window will open prompting you to log in.

> **Note:** Make sure the selected Google account has access to GEE. It is recommended to provide your GCP project name using the `project='your-project'` parameter.

## Example 1: Direct Download (Multithread Tiled)

If you have a small to medium-sized area (e.g., a municipality or a specific polygon) and want the `.tif` file immediately on your machine, use the `method='direct'` option.

The `cdts` package will automatically slice your region into smaller grids (tiles), open dozens of concurrent threads to Google's servers, download the pieces, and seamlessly mosaic them together (using `rasterio.merge`).

```python
from cdts.gee import download_gee_timeseries

# Bounding box [min_lon, min_lat, max_lon, max_lat]
my_roi = [-47.95, -15.85, -47.85, -15.75]

download_gee_timeseries(
    roi=my_roi, 
    start_date='2010-01-01',
    end_date='2020-12-31', 
    out_dir='./gee_direct_data',
    method='direct',           # Enables immediate tiled download and local mosaicking
    composite_type='annual',   # Generates LandTrendr-style Annual Medoid Composites
    project='my-gcp-project'   # Replace with your Google Cloud Project ID
)
```

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
2. **Harmonization:** Values from Landsat 8 and 9 (OLI) are mathematically converted to their ETM+ equivalents using coefficients from Roy et al. (2016). This ensures a perfect time series, free from sensor biases.
3. **Cloud Masking:** The `QA_PIXEL` quality assurance band is used to filter out dense clouds and shadows across all images.
4. **Medoid Compositing:** Instead of a simple median, we apply the Medoid geometric strategy to find the actual real-world pixel that best represents the season (a standard approach in *eMapR/LandTrendr* workflows).
