# LandTrendr in CDTS: End-to-End Tutorial

## 1. Introduction to LandTrendr

**LandTrendr** (Landsat-based detection of Trends in Disturbance and Recovery) is a highly influential trajectory-based algorithm designed to extract both abrupt changes (like deforestation or fire) and gradual changes (like forest degradation, disease, or recovery) from annual satellite imagery.

### Background

Originally developed by Robert Kennedy *et al.* (see [References](#5-references)), LandTrendr works by reducing complex, noisy, annual time series data into a sequence of simplified straight-line segments. It minimizes the residual error between the actual satellite observations and the simplified straight-line model, essentially "filtering out" inter-annual noise (like slight phenological differences or minor atmospheric effects) to reveal the true underlying landscape dynamics.

The algorithm is also natively available in Google Earth Engine — see the [eMapR Lab GitHub](https://github.com/eMapR/LT-GEE).

By integrating LandTrendr into **CDTS**, you gain the ability to run this powerful algorithm locally, on HPC clusters, or natively on massive GeoTIFF stacks without being constrained by cloud-platform quotas.

---

## 2. API Entry Points

The `cdts` library exposes LandTrendr at three different levels of abstraction, depending on your needs and data scale:

1. **`run_landtrendr` (Pixel Level):** Found in `cdts.landtrendr`. Takes a simple 1D numpy array of values. Ideal for unit testing, plotting single pixels, or integrating into your own custom map-reduce pipelines.
2. **`run_landtrendr_array` (In-Memory Array):** Found in `cdts.raster`. Takes a 3D numpy array `(Time, Rows, Cols)`. Automatically distributes the pixels across all your CPU cores using Python's multiprocessing. Ideal for small regions of interest (ROIs) that fit in your machine's RAM.
3. **`run_landtrendr_image` (Out-of-Core Image):** Found in `cdts.raster`. Takes a direct file path to a massive multi-band GeoTIFF. It internally reads the image in spatial chunks (e.g., 512x512 blocks), processes them in parallel, and writes the output directly to disk. Ideal for processing entire states or countries on a laptop without encountering `MemoryError`.

---

## 3. End-to-End Workflow (Array-based)

This tutorial will guide you through an end-to-end process: loading a real multi-band GeoTIFF, running the C++ optimized LandTrendr algorithm in Python, extracting change metrics, and visualizing the results.

### Step 3.1: Loading the Data

LandTrendr requires an annual time series of a single spectral index. The Normalized Burn Ratio (NBR) is the most commonly used index for forest disturbance because it is highly sensitive to canopy removal and moisture loss.

In LandTrendr conventions, indices are often inverted (e.g., `-NBR` or `10000 - NBR`) so that a **disturbance** is represented by a **positive increase** in value.

```python
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# 1. Define the years corresponding to our data stack
# Let's assume we have a 30-year Landsat stack from 1990 to 2019
start_year = 1990
end_year = 2019
years = np.arange(start_year, end_year + 1)

# 2. Load the GeoTIFF using rasterio
input_path = "data/annual_nbr_stack_1990_2019.tif"
with rasterio.open(input_path) as src:
    # Read the entire 3D array (Bands, Rows, Cols)
    raster_stack = src.read()
    profile = src.profile

print(f"Loaded stack with shape: {raster_stack.shape}")
# Example Output: Loaded stack with shape: (30, 2000, 2000)
```

### Step 3.2: Despiking (Optional but Recommended)

Satellite data often contains residual noise (e.g., missed cloud shadows) that appear as sharp, one-year dips in the time series. If not removed, LandTrendr might fit false segments to these spikes. The `cdts.smooth` module provides a despiking function to handle this.

```python
from cdts.smooth import desawtooth

# Despike the time series to remove 1-year anomalous drops
# This is a critical pre-processing step for LandTrendr
smoothed_stack = desawtooth(raster_stack)
```

### Step 3.3: Running the Algorithm

We use the `run_landtrendr_array` function to apply the segmentation logic across all pixels in parallel.

```python
from cdts.raster import run_landtrendr_array

print("Running LandTrendr segmentation...")
vertices_stack = run_landtrendr_array(
    years=years,
    raster_stack=smoothed_stack,
    max_segments=6,        # Allow up to 6 distinct trend segments
    pval_threshold=0.05,   # Statistical significance threshold for segments
    n_jobs=-1              # Distribute work across all CPU cores
)
print("Segmentation complete!")
```

**Understanding the Output (`vertices_stack`)**:
The output is a 3D numpy array. If `max_segments=6`, the maximum number of vertices is 7. The output will have `14` bands (2 * 7).
- **Bands 0 to 6**: The *Years* of the identified vertices.
- **Bands 7 to 13**: The *Fitted Values* corresponding to those years.

### Step 3.4: Extracting Disturbance Events

Now that we have the simplified trajectories for every pixel, we want to extract the greatest disturbance event.

```python
from cdts.metrics import extract_events

# Extract the greatest loss event (disturbance)
events = extract_events(
    vertices_stack=vertices_stack,
    event_type="loss",           # "loss" corresponds to a drop in the original NBR
    sort_by="greatest",          # Select the segment with the largest absolute change
    min_magnitude=150.0,         # Minimum change magnitude to be considered an event
    min_duration=1               # Minimum duration (1 = abrupt change, >1 = gradual)
)

# The result is a dictionary of 2D maps
yod_map = events["yod"]           # Year of Detection
magnitude_map = events["magnitude"] # Change Magnitude
duration_map = events["duration"]   # Change Duration
```

### Step 3.5: Exporting Results

You can export these 2D metrics back to GeoTIFFs using the built-in `save_raster` function from `cdts.io`. It automatically handles the GeoTIFF profiles if you provide the original array or profile reference.

```python
from cdts.io import save_raster

# Provide the original profile or xarray to inherit georeferencing
# Save Year of Detection
save_raster(
    array=yod_map.astype('uint16'), 
    output_path="results/lt_yod.tif", 
    crs=profile['crs'],
    transform=profile['transform'],
    nodata=0
)

# Save Magnitude
save_raster(
    array=magnitude_map.astype('float32'), 
    output_path="results/lt_magnitude.tif",
    crs=profile['crs'],
    transform=profile['transform'],
    nodata=0
)
```

## 3. Visualizing a Single Pixel Trajectory

To truly understand LandTrendr, it helps to plot the original data alongside the fitted vertices for a single pixel.

```python
import matplotlib.pyplot as plt

# Pick a pixel that experienced a disturbance
row, col = 500, 500

# 1. Get original time series
original_ts = smoothed_stack[:, row, col]

# 2. Get the fitted vertices for this pixel
pixel_vertices = vertices_stack[:, row, col]

# The first half of the array are the years, the second half are the values
max_v = vertices_stack.shape[0] // 2
vertex_years = pixel_vertices[:max_v]
vertex_values = pixel_vertices[max_v:]

# Filter out empty vertices (where year == 0)
valid_idx = vertex_years > 0
vertex_years = vertex_years[valid_idx]
vertex_values = vertex_values[valid_idx]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(years, original_ts, marker='o', label='Original Data (Smoothed)', color='gray', linestyle='--')
plt.plot(vertex_years, vertex_values, marker='s', label='LandTrendr Fitted Trajectory', color='red', linewidth=2)

plt.title("LandTrendr Pixel Trajectory")
plt.xlabel("Year")
plt.ylabel("Spectral Index Value")
plt.legend()
plt.grid(True)
plt.show()
```

## 4. Best Practices and Tips

1. **Index Selection**: While NBR is the standard for forest disturbance, Tasseled Cap Wetness (TCW) or Tasseled Cap Angle (TCA) are extremely effective. NDVI is generally less sensitive to structural forest changes but good for agricultural monitoring.
2. **Out-of-Core Processing**: If your input GeoTIFF is larger than your available RAM, use `cdts.raster.run_landtrendr_image` instead of `run_landtrendr_array`. The image-based function automatically chunks the raster and processes it in blocks, keeping memory usage strictly bounded.
3. **Overfitting**: A `max_segments` value of 6 is empirically proven to be optimal for a 30-year time series. Increasing it to 8 or 10 on a 30-year stack will lead to the algorithm overfitting noise, resulting in false positive disturbances.
4. **Chunking is everything for distributed runs**: spatial chunking (e.g. `chunks={'time': -1, 'y': 512, 'x': 512}`) is mandatory, since LandTrendr needs the full time series per pixel to run the temporal segmentation. `cdts.build_time_series` chunks reasonably by default, but you may need to rechunk depending on your cluster's RAM per worker.
5. **Avoid `.compute()` on large cubes**: it pulls all the resulting data back into the RAM of your single head node/laptop. Use `.to_zarr()` or `cdts.save_raster()` to sink data directly from the workers to disk/cloud instead.
6. **Prefer Zarr over GeoTIFF for multi-machine writes**: GeoTIFFs can cause write-locks; cloud-native Zarr writes concurrently in chunks.

---

## 5. Distributed Processing with Dask

CDTS is built on top of `xarray` and `dask`, which means it natively scales from a single machine to a distributed cluster of multiple machines (an on-premise HPC, AWS, GCP, or a Kubernetes cluster). This section walks through an end-to-end workflow for running LandTrendr at a massive spatial scale using multiple machines via `dask.distributed`.

### 5.1. Setting up the Dask Cluster

To process across multiple machines, you don't need to hardcode the IPs of all machines in your script. Dask uses a **Scheduler-Worker** architecture:

1. **The Scheduler** (e.g., `192.168.0.100`) coordinates the work.
2. **The Workers** (the other machines) connect to the Scheduler to ask for work.

**On the Main Machine (Scheduler):**
```bash
dask scheduler
# It will print out its address, e.g., tcp://192.168.0.100:8786
```

**On the Worker Machines:**
Open a terminal on each machine and connect it to the scheduler. **This is where you configure the cores and memory for each machine:**
```bash
dask worker tcp://192.168.0.100:8786 --nworkers 4 --nthreads 2 --memory-limit 16GB
```
*(In this example, the machine dedicates 4 processes, 2 threads each, and a 16GB RAM limit to the cluster.)*

In your script, you only need to connect your `Client` to the Scheduler. The image is divided automatically into "chunks" (e.g., blocks of 512x512 pixels), and the Scheduler automatically sends different chunks to different worker machines as they become available.

### 5.2. End-to-End Example

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

---

## 6. References

- Kennedy, R. E., Yang, Z., & Cohen, W. B. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr—Temporal segmentation algorithms. **Remote Sensing of Environment**, 114(12), 2897–2910. [https://doi.org/10.1016/j.rse.2010.07.008](https://doi.org/10.1016/j.rse.2010.07.008)
