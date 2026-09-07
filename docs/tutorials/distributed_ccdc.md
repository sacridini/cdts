# Distributed Processing with CCDC and Dask

CDTS provides full native Dask integration for **Continuous Change Detection and Classification (CCDC)**, allowing you to run heavy harmonic regression mapping across thousands of remote machines simultaneously.

This guide demonstrates an End-to-End (E2E) workflow for running CCDC on a massive spatial scale, extracting harmonic coefficients, and generating cloud-free synthetic imagery across a Dask Cluster.

## 1. End-to-End Example

The following script connects to a remote cluster, lazily loads a massive STAC catalog, prepares a QA mask for cloud filtering, fits the CCDC harmonics across all machines, and saves the output directly to a Cloud Storage bucket in parallel using Zarr.

```python
import xarray as xr
from dask.distributed import Client
import cdts

# 1. Connect to the multi-node Dask Cluster
scheduler_address = 'tcp://192.168.0.100:8786'
client = Client(scheduler_address)
print(f"Connected to Cluster! Dashboard: {client.dashboard_link}")

# 2. Lazily load a massive spatial extent from Planetary Computer (Landsat C2 L2)
cube = cdts.build_time_series(
    source='planetary_computer',
    collection='landsat-c2-l2',
    bbox=[-63.0, -11.0, -62.0, -10.0],  # Massive area
    start_date='2015-01-01',
    end_date='2023-12-31',
    bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'qa_pixel'],
    resolution=30
)

# 3. Extract and configure the Cloud / QA Mask
# CCDC needs to know which pixels to ignore during regression.
qa_mask = cdts.extract_water_mask(cube.sel(band='qa_pixel')) # or equivalent cloud mask parser
spectral_cube = cube.drop_sel(band=['qa_pixel'])

# 4. Extract Julian Dates
# CCDC fits harmonics based on the continuous Julian dates of the time-series.
dates_julian = cube.time.dt.dayofyear.values + (cube.time.dt.year.values * 365)

# 5. Define CCDC execution using the xarray accessor
# xarray will distribute the block-wise C++ regression across the workers.
ccdc_results = spectral_cube.cdts.run_ccdc(
    dates=dates_julian,
    qa_stack=qa_mask,           # Automatically skips clouded pixels
    max_segments=6,             # Maximum number of breaks a pixel can have
    return_coefs=True,          # Return the harmonic coefficients (Intercept, Slopes, Sines, Cosines)
    n_jobs=1                    # Set to 1 per worker (Dask already handles the parallelism)
)

# 6. Execute and Save in Parallel (Zarr)
# .to_zarr() is highly recommended for distributed writes to cloud buckets (S3/GCS)
# This is the moment computation actually triggers across all machines!
ccdc_results.to_zarr('s3://my-bucket/ccdc_coefficients.zarr', mode='w')

print("Distributed CCDC processing complete!")
```

## 2. Using the Results (Synthetic Images)

Once you have your coefficients saved, you can load them lazily and generate synthetic (completely cloud-free) imagery for **any arbitrary day of the year**, leveraging Dask again for parallel reconstruction.

```python
# Re-load the distributed results
coef_stack = xr.open_zarr('s3://my-bucket/ccdc_coefficients.zarr')

# We can use xr.apply_ufunc to apply the synthetic prediction across the cluster
def predict_wrapper(coefs):
    from cdts.ccdc import predict_synthetic_image
    return predict_synthetic_image(
        ccdc_coefs_stack=coefs,
        target_julian_day=(2024 * 365) + 200, # Example: Day 200 of 2024
        num_bands=6
    )

synthetic_cube = xr.apply_ufunc(
    predict_wrapper,
    coef_stack,
    input_core_dims=[['band']],
    output_core_dims=[['band']],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float]
)

# Save the synthetic cloud-free mosaic back to storage!
synthetic_cube.to_zarr('s3://my-bucket/ccdc_synthetic_day200.zarr', mode='w')
```

## 3. Best Practices for Distributed Run

1. **Spatial Chunking:** Ensure your input `spectral_cube` is chunked spatially (e.g., `chunks={'time': -1, 'y': 256, 'x': 256}`). The temporal axis `time` MUST NOT be chunked (`-1`), because CCDC needs the entire history of a single pixel to fit the harmonic model.
2. **Cluster Tuning:** CCDC is highly CPU intensive. Choose worker nodes with strong compute power (e.g., `c2-standard` instances on GCP or `c5` instances on AWS) rather than high-memory nodes, since spatial chunks can be kept small.
3. **Avoid `compute()`:** Do not call `.compute()` on large output arrays. Use `.to_zarr()` to sink the data directly from the workers to cloud storage, bypassing your local head node entirely.
