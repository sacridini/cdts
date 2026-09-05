# Parallel & Distributed Cloud Processing

While CDTS uses a highly optimized C++ engine capable of utilizing all cores on a single machine, processing entire countries or continents across decades requires scaling out to multiple computers.

CDTS achieves this seamlessly by integrating natively with **Xarray** and **Dask**. You do not need to rewrite your algorithms or learn C++ to scale your workflows. By using the built-in Xarray accessor (`.cdts`), the same code that runs on your laptop will run perfectly across a massive cloud computing cluster or a local network of office desktops.

---

## 1. The Power of the `.cdts` Xarray Accessor

When you load a lazy, Dask-backed DataCube (using `build_time_series`, `stackstac`, or `xarray.open_zarr`), CDTS extends the Xarray API with its own methods.

Instead of writing complex loops, you simply call `.cdts.run_ccdc()` or `.cdts.run_landtrendr()`. CDTS will automatically map the underlying C++ algorithm across thousands of spatial "chunks" (blocks of pixels) and send them to the Dask workers for parallel execution.

```python
import xarray as xr
import cdts # Registers the .cdts accessor

# Load a lazy Dask-backed cube
cube = xr.open_zarr("s3://my-bucket/Rondonia_Landsat_Stack.zarr")

# Apply CCDC across the entire distributed cluster
ccdc_results = cube.cdts.run_ccdc(dates=fractional_years_array)
```

---

## 2. Setting up a Distributed Cluster

A Dask cluster consists of one **Scheduler** (the boss) and one or more **Workers** (the employees). You can set this up on a single machine, across multiple cloud servers (AWS/GCP), or even across old desktops connected to the same office Wi-Fi!

### Option A: Local Office Network (LAN / Wi-Fi)

You can turn any group of computers sharing a network into a supercomputer:

1. **On the Main Computer (Scheduler):**
   Open the terminal and start the scheduler. It will output an IP address (e.g., `tcp://192.168.1.10:8786`).
   ```bash
   dask-scheduler
   ```

2. **On the Secondary Computers (Workers):**
   Ensure CDTS is installed. Open the terminal and connect them to the main computer's IP:
   ```bash
   dask-worker tcp://192.168.1.10:8786
   ```

3. **In your Python Script (on the Main Computer):**
   ```python
   from dask.distributed import Client
   
   # Connect to the scheduler
   client = Client("tcp://127.0.0.1:8786")
   ```

### Option B: Cloud Computing (AWS/GCP/Kubernetes)

For enterprise-scale, you can rent virtual machines using tools like `dask-cloudprovider`, `dask-kubernetes`, or managed services like `Coiled`.

```python
from dask_kubernetes import KubeCluster
from dask.distributed import Client

cluster = KubeCluster.from_yaml('worker-spec.yml')
cluster.scale(50) # Spin up 50 servers in the cloud!
client = Client(cluster)
```

---

## 3. Dask vs OpenMP: Avoiding Thread Oversubscription

When integrating Dask with CDTS's highly optimized C++ engine, there is a classic trap: **Thread Oversubscription**. 
By default, the C++ engine uses OpenMP to spawn one thread per CPU core. If Dask also spawns multiple worker processes on the same machine, and each worker spawns its own OpenMP threads, you will end up with `(Dask Workers) * (OpenMP Threads)` competing for the same CPU cores. This causes catastrophic context-switching and makes processing slower.

To solve this, the `.cdts` accessor allows you to orchestrate the parallelism using the `n_jobs` parameter:

* **Strategy A (Cluster-Level / Recommended):** Configure Dask to spawn only **1 worker per physical machine**. Then, use `n_jobs=-1` in CDTS. Dask will simply hand over a chunk of the image to a machine, and OpenMP will use 100% of the cores on that machine to process it at maximum C++ speed.
* **Strategy B (Process-Level):** If Dask is spawning multiple workers per machine (e.g., 8 workers on an 8-core CPU), you MUST pass `n_jobs=1` to CDTS. This turns off OpenMP, allowing Dask to manage the parallelism across its worker processes.

---

## 4. Why Zarr is Crucial (Say Goodbye to GeoTIFF)

When a single laptop processes data, saving to a single `.tif` file is fine. However, when 50 cloud servers or 3 office PCs are processing data simultaneously, they **cannot** all write to the same GeoTIFF file at the same time—it creates a massive bottleneck and corrupts the file.

The modern standard for cloud processing is **Zarr**. Zarr stores multi-dimensional arrays in a structured directory of thousands of tiny chunk files. This allows an unlimited number of servers to read and write data simultaneously without stepping on each other's toes.

To make this frictionless, CDTS provides `.cdts.to_zarr_optimized()`. This method encapsulates Spatial Engineering best practices: it forces ideal spatial chunking (e.g., 512x512) and consolidates metadata into a single JSON, making cloud reads up to 10x faster.

### Complete Practical Workflow

Here is an end-to-end example of connecting to a distributed cluster, loading a Zarr cube, running LandTrendr in parallel, and saving the output directly back to a cloud storage bucket as Zarr.

```python
import dask.distributed
import xarray as xr
import cdts

def run_distributed_analysis():
    # 1. Connect to our distributed Dask cluster
    client = dask.distributed.Client("tcp://192.168.1.10:8786")
    print(f"Cluster connected! View Dashboard at: {client.dashboard_link}")
    
    # 2. Load the input data from Cloud Storage (Zarr format)
    # The 'chunks' argument ensures data is streamed in small pieces
    cube = xr.open_zarr('gs://my-bucket/Landsat_Timeseries.zarr')
    
    years = [2020, 2021, 2022, 2023, 2024]
    
    # 3. Disperse the C++ algorithm across the cluster
    # We use Strategy A (n_jobs=-1), assuming 1 Dask Worker per Node
    print("Mapping LandTrendr across the cluster...")
    lt_results = cube.cdts.run_landtrendr(years=years, n_jobs=-1)
    
    # 4. Execute and stream the output to Cloud Storage in parallel
    # Workers write chunks directly to the Zarr bucket with consolidated metadata
    print("Executing distributed processing and saving...")
    lt_results.cdts.to_zarr_optimized('gs://my-bucket/LandTrendr_Results.zarr')
    
    print("Analysis complete!")

if __name__ == "__main__":
    run_distributed_analysis()
```

> [!TIP]
> Always check the Dask Dashboard (usually available at `http://localhost:8787`). It provides a beautiful real-time visualization of all your servers, CPUs, memory usage, and task streams as the C++ engine crushes the pixels!
