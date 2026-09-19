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

## 3. Zarr Format for Cloud Processing

In a distributed environment where multiple workers process data concurrently, writing outputs to a single GeoTIFF file can result in file corruption or I/O bottlenecks. 

Zarr is a format designed for cloud storage that represents multi-dimensional arrays as a directory of compressed chunk files. Because each chunk is a separate file, multiple distributed workers can write their respective chunks in parallel without encountering race conditions.

To assist with exporting data to this format, CDTS provides the `.cdts.to_zarr_optimized()` method. This helper function allows for custom spatial chunking (defaulting to 512x512) and consolidates the dataset metadata into a single file to improve read performance from object storage (like AWS S3 or Google Cloud Storage).

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

---

## 4. Troubleshooting a Multi-Machine LAN Cluster

Connecting a mixed-OS cluster (e.g. a Windows desktop as scheduler + a macOS laptop as a worker) over a home/office network hits a handful of predictable snags. Here's what to check, roughly in the order you'll hit them.

### 4.1. Windows Firewall blocks the remote worker

Windows often marks a home/office network as **Public**, which blocks unsolicited inbound connections by default — the scheduler will start fine locally, but a worker on another machine will simply never be able to reach it.

Fix it with a firewall rule scoped to your LAN subnet (run in an **elevated** PowerShell), rather than opening the ports to the world:

```powershell
New-NetFirewallRule -DisplayName "Dask Scheduler (LAN)" -Direction Inbound -Protocol TCP `
    -LocalPort 8786,8787 -Action Allow -RemoteAddress 192.168.1.0/24
```

Replace `192.168.1.0/24` with your actual subnet. `8786` is the scheduler port, `8787` the dashboard.

### 4.2. Keep Python, dask and distributed versions aligned on every machine

The scheduler/worker wire protocol assumes matching (or very close) `dask`/`distributed` versions; a mismatch triggers a `VersionMismatchWarning` and can cause hard-to-diagnose failures under load. Before connecting a new worker, check:

```bash
python -c "import sys, dask, distributed; print(sys.version, dask.__version__, distributed.__version__)"
```

...and make sure it's close to what the scheduler machine reports. `cdts`'s published PyPI wheels currently cover Python 3.9–3.12 (Windows, Linux, macOS arm64) — there is no prebuilt 3.13 wheel yet, so standardize on a **Python 3.12** environment (venv or conda) on every machine to avoid an accidental from-source build.

### 4.3. macOS: a worker crash-loops silently the moment a task touches `cdts`

**Symptom:** `dask worker` starts and registers with the scheduler fine. But as soon as a real task imports `cdts` (e.g. the first `.cdts.run_landtrendr()` call), the worker process vanishes and Dask's Nanny silently respawns it with a new port — forever. No Python traceback reaches the scheduler or client; calling `client.run(...)` against that worker just raises `CommClosedError: ... Stream is closed`.

**Cause:** `cdts` depends on `torch`, and on macOS both `torch` and cdts's own compiled C++ extension (`cdts._core`) link their own copy of the OpenMP runtime (`libomp`/`libiomp`). Loading both inside the same process aborts the whole process (`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`) instead of raising a catchable Python exception — which is exactly what a Nanny-managed silent restart loop looks like from the outside.

**Fix:** set these two environment variables before launching the worker on macOS:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 dask worker tcp://<scheduler-ip>:8786 --nworkers <n> --nthreads 1
```

If a worker is still crash-looping and you need to see the actual OS-level error, look at the raw terminal where `dask worker` runs directly — `Segmentation fault`, `Illegal instruction`, or the `OMP: Error #15` line only ever prints there, never through the Dask protocol.

### 4.4. `dask worker` can't find `cdts` even though you just installed it

If `pip install cdts` (or `pip install -e .`) reported success but a worker still throws `ModuleNotFoundError: No module named 'cdts'`, the `dask` command on your `PATH` is almost certainly resolving to a *different* Python installation (a different conda env, a system Python, a pyenv shim) than the one you installed `cdts` into.

Force it explicitly — activate the right environment, then launch via `python -m dask` instead of the bare `dask` binary, so it always uses the currently active interpreter:

```bash
conda activate cdts-worker   # or: source your-venv/bin/activate
python -m dask worker tcp://<scheduler-ip>:8786 --nworkers <n> --nthreads 1
```

### 4.5. Sanity-check every worker before submitting real work

From the client/head node, verify `cdts` actually imports on every connected worker *before* kicking off a real job — it's much faster to catch a broken worker this way than to debug a stuck/slow distributed run:

```python
from dask.distributed import Client

client = Client("tcp://<scheduler-ip>:8786")

def check():
    import socket, sys
    try:
        from cdts.raster import run_landtrendr_array
        return (socket.gethostname(), sys.executable, "ok")
    except Exception as e:
        return (socket.gethostname(), sys.executable, repr(e))

print(client.run(check, on_error="return"))
```

A clean `"ok"` (or a normal, readable Python exception) per worker means you're good to go. A `CommClosedError` / `Stream is closed` here is the macOS crash-loop symptom from 4.3 — fix that first.
