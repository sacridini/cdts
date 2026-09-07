# Distributed Processing with LandTrendr and Dask

CDTS is built on top of `xarray` and `dask`, which means it natively scales from a single machine to a distributed cluster of multiple machines (like an on-premise HPC, AWS, GCP, or a Kubernetes cluster). 

This guide demonstrates an End-to-End (E2E) workflow for running LandTrendr on a massive spatial scale using multiple machines via `dask.distributed`.

## 1. Setting up the Dask Cluster

To process across multiple machines, you need a Dask Scheduler and several Dask Workers. You can deploy this using tools like `dask-kubernetes`, `dask-yarn`, `dask-cloudprovider`, or simply by running `dask scheduler` and `dask worker` on your machines.

In Python, you connect your script to the cluster using the `Client`.

## 2. End-to-End Example

The following script connects to a remote cluster, lazily loads a massive STAC catalog, runs LandTrendr across all machines, and saves the result directly to a Cloud Storage bucket in parallel using Zarr.

```python
import xarray as xr
from dask.distributed import Client
import cdts

# 1. Connect to the multi-node Dask Cluster
# Replace with your actual Dask scheduler address
scheduler_address = 'tcp://192.168.0.100:8786'
client = Client(scheduler_address)
print(f"Connected to Cluster! Dashboard: {client.dashboard_link}")

# 2. Lazily load a massive spatial extent from AWS Earth Search
# Note: Because we use Dask, the data is NOT downloaded yet. 
# Only the metadata is parsed.
cube = cdts.build_time_series(
    source='earth_search',
    collection='sentinel-2-l2a',
    bbox=[-63.0, -11.0, -62.0, -10.0],  # Massive area
    start_date='2017-01-01',
    end_date='2023-12-31',
    bands=['red', 'nir'],
    resolution=10
)

# 3. Calculate a vegetation index (e.g., NDVI)
# This operation is lazy and added to the Dask computation graph.
ndvi = (cube.sel(band='nir') - cube.sel(band='red')) / (cube.sel(band='nir') + cube.sel(band='red'))
ndvi = ndvi.expand_dims(band=['NDVI'])

# 4. Regularize to an annual composite (Medoid)
# The `medoid` method is robust against clouds and shadows.
annual_cube = cdts.regularize_time_series(ndvi, freq='1Y', method='medoid')

# 5. Define LandTrendr execution
# xarray will distribute the apply_ufunc operations across the workers.
lt_results = cdts.run_landtrendr(
    annual_cube,
    max_segments=6,
    spike_threshold=0.9,
    recovery_threshold=0.25,
    pval_threshold=0.05,
    best_model_proportion=0.75,
    min_observations_needed=6
)

# 6. Extract Disturbance Events (YOD, Magnitude, Duration)
events = cdts.extract_events(
    lt_results,
    event_type="loss",
    sort_by="greatest",
    min_magnitude=0.1,
    min_duration=1
)

# Convert events dictionary to an xarray Dataset for distributed saving
events_ds = xr.Dataset({
    k: (['y', 'x'], v) for k, v in events.items()
})

# 7. Execute and Save in Parallel (Zarr)
# .to_zarr() is highly recommended for distributed writes to cloud buckets (S3/GCS)
# This is the moment computation actually triggers across all machines!
events_ds.to_zarr('s3://my-bucket/landtrendr_results.zarr', mode='w')

print("Distributed processing complete!")
```

## 3. Best Practices for Distributed Run

1. **Chunking is Everything:** Ensure your input cube is appropriately chunked. Spatial chunking (e.g., `chunks={'time': -1, 'y': 512, 'x': 512}`) is mandatory for LandTrendr because it requires the full time-series per pixel to run the temporal segmentation. `cdts.build_time_series` automatically attempts to chunk optimally, but you may need to rechunk depending on your cluster's RAM per worker.
2. **Avoid `compute()`:** Do not call `.compute()` on large cubes as it will try to pull all the resulting data back into the RAM of your single head node/laptop. Use `.to_zarr()` or `cdts.save_raster()` to sink the data directly from the workers to disk/cloud.
3. **Use Zarr over GeoTIFF:** When writing in a multi-machine setup, GeoTIFFs can cause write-locks. Cloud-native Zarr format writes concurrently in chunks.
