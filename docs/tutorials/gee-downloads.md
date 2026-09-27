# Google Earth Engine

<p class="lead">Let Google's servers do the heavy preparation (Landsat sensor harmonisation, cloud masking and compositing) and download a ready-to-use time series as GeoTIFFs. This is the quickest way to get the annual stacks LandTrendr needs.</p>

<div class="glance" markdown>
<div><span class="k">Does</span><span class="v">Harmonise Landsat 5/7/8/9, mask clouds, composite, download</span></div>
<div><span class="k">Composites</span><span class="v"><code>annual</code> medoid (LandTrendr) or <code>dense</code> (CCDC)</span></div>
<div><span class="k">Output</span><span class="v">GeoTIFFs on disk, or in your Google Drive for large areas</span></div>
<div><span class="k">Needs</span><span class="v">An Earth Engine account and a Google Cloud project</span></div>
</div>

## Authentication

Earth Engine requires the user to authenticate their machine with Google Cloud Project (GCP) credentials that have API access enabled. Every time you attempt a download, the `download_gee_timeseries` function handles this initialization. If your credentials expire or do not exist, a browser window will open prompting you to log in.

!!! note
    Make sure the Google account has access to Earth Engine, and pass your Google Cloud project with `project="your-project"`.

## Small areas: direct download

If you have a small to medium-sized area (e.g., a municipality or a specific polygon) and want the `.tif` file immediately on your machine, use the `method='direct'` option.

The `cdts` package will automatically slice your region into smaller grids (tiles), open dozens of concurrent threads to Google's servers, download the pieces, and seamlessly mosaic them together (using `rasterio.merge`).

!!! warning "Request size limits"
    Google's REST API has a strict 50 MB payload limit per request. While the tiling system mitigates this for standard annual composites, attempting to download a `'dense'` time series (which flattens dozens of images and hundreds of bands into a single stack) for a large region using `method='direct'` will likely trigger a *Payload Too Large* error. For dense time series or state-level areas, always use `method='drive'`.

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

## Large areas: export to Google Drive

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

## What happens on the server

When using `composite_type='annual'` (the current `cdts` default for LandTrendr integration):

1. **Sensor Fusion:** The function fetches Landsat 5, 7, 8, and 9 collections (Surface Reflectance Collection 2).
2. **Harmonization:** Values from Landsat 8 and 9 (OLI) are mathematically converted to their ETM+ equivalents using coefficients from Roy et al. (2016) (see [References](#references)). This ensures a perfect time series, free from sensor biases.
3. **Cloud Masking:** The `QA_PIXEL` quality assurance band is used to filter out dense clouds and shadows across all images.
4. **Medoid Compositing:** Instead of a simple median, we apply the Medoid geometric strategy to find the actual real-world pixel that best represents the season (a standard approach in *eMapR/LandTrendr* workflows).

## References

- Roy, D. P., Kovalskyy, V., Zhang, H. K., Vermote, E. F., Yan, L., Kumar, S. S., & Egorov, A. (2016). Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity. **Remote Sensing of Environment**, 185, 57–70. [https://doi.org/10.1016/j.rse.2015.12.024](https://doi.org/10.1016/j.rse.2015.12.024)
