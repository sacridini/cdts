# CDTS: Continuous Monitoring of Land Cover and Land Use using Dense Time Series

**CDTS** is a high-performance Python package for Earth Observation (EO) data cube processing and time series analysis. It bridges the gap between modern cloud-native data formats (STAC, Xarray, Dask) and state-of-the-art pixel-based trajectory algorithms (TWDTW, CCDC, LandTrendr, SOM). 

Built with highly optimized C++ extensions (OpenMP and Eigen SIMD) bound to Python via `pybind11`, CDTS is designed to handle massive multi-spectral satellite image time series efficiently while keeping memory footprints strictly bounded.

---

## Key Capabilities

- **ARD Data Cube Ingestion:** Fetch cloud-native STAC catalogs (via MGRS/WRS tiles or Bounding Boxes) or parse local TIFF directories into lazy Dask-backed `xarray` Datacubes.
- **Semantic Cloud Masking:** Automated extraction and translation of Quality Assessment (QA) bands for Landsat and Sentinel-2 directly inside the query pipeline.
- **Temporal Regularization:** Mathematical composite generation (e.g., Medoid, Median) to align irregular satellite acquisitions into uniform time steps (crucial for Deep Learning and DTW).
- **High-Performance C++ Algorithms:**
  - **TWDTW** (Time-Weighted Dynamic Time Warping): Highly optimized with LB_Keogh lower bounding, early abandonment, Sakoe-Chiba constraints, and multivariate Eigen vectorization.
  - **Batch SOM** (Self-Organizing Maps): Unsupervised multi-threaded clustering of massive spectral-temporal arrays.
  - **CCDC / COLD**: Continuous Change Detection and Classification via robust harmonic modeling.
  - **LandTrendr**: Trajectory-based disturbance and recovery detection.
- **Deep Learning (`cdts.ai`):** Pre-built PyTorch architectures tailored for spatio-temporal Earth Observation (U-TAE, TempCNN, Siamese Networks).

---

## Installation

```bash
pip install cdts
```
*(Note: Requires a C++14 compatible compiler installed on your system to build the optimized extensions).*

---

## 1. Cloud-Native ARD Cubes (STAC)

Fetch lazy evaluated, Dask-backed analysis-ready data cubes directly from STAC providers (e.g., Earth Search, Planetary Computer, Brazil Data Cube).

```python
import cdts

# Build a lazy DataArray using MGRS/WRS tiles or Bounding Boxes
cube = cdts.build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    tiles=["22JFQ"], # Sentinel-2 MGRS or Landsat WRS-2 (e.g., "215065")
    start_date="2022-01-01",
    end_date="2022-12-31",
    bands=["red", "green", "blue", "nir"],
    apply_cloud_mask=True # Automatically fetches QA band and natively masks clouds/shadows
)

print(cube) # Returns an xarray.DataArray (Time, Band, Y, X)
```

## 2. Temporal Regularization

Algorithms like TWDTW, SOM, and Deep Learning expect temporally aligned data. `cdts` natively regularizes irregular STAC acquisitions.

```python
from cdts import regularize_time_series

# Aggregate observations into 16-day Medoid composites 
# (Maintains xarray lazy evaluation via Dask graphs)
cube_16d = regularize_time_series(cube, freq="16D", method="medoid")
```

## 3. Time-Weighted Dynamic Time Warping (TWDTW)

The C++ TWDTW engine handles multivariate sequences simultaneously using Eigen's $L^2$ norms and aggressively skips non-matching pixels using $O(N)$ Lower Bounding techniques.

```python
from cdts.twdtw import classify_twdtw
import numpy as np

# Suppose you have regularized data (Y, X, Time, Bands)
dates = np.arange(1, 366, 16) # Day of year

# Define temporal patterns (Signatures)
patterns = {
    "Forest": (forest_signature_array, dates),
    "Agriculture": (soy_signature_array, dates)
}

# Run classification block-by-block using OpenMP
# n_jobs=-1 automatically uses all CPU cores minus 1 to prevent OS lockup
classes_map, dist_map, class_names = classify_twdtw(
    cube_16d.values, 
    dates, 
    patterns, 
    n_jobs=-1 
)
```

## 4. Self-Organizing Maps (SOM)

Unsupervised classification and dimensionality reduction of time series using a fast Batch SOM algorithm implemented in C++.

```python
from cdts.ai import train_som_batch, predict_bmus

# Flatten cube to (Pixels, Features)
X_train = cube_16d.values.reshape(-1, cube_16d.shape[2] * cube_16d.shape[3])

# Train a 10x10 SOM grid
som_weights = train_som_batch(
    data=X_train,
    grid_rows=10,
    grid_cols=10,
    num_epochs=100,
    n_jobs=-1
)

# Predict Best Matching Units (BMUs) for new data
bmus = predict_bmus(X_train, som_weights, n_jobs=-1)
```

## 5. LandTrendr & CCDC

Continuous structural monitoring using robust breakpoint and harmonic regression models directly on xarray Datacubes via pandas-like accessors (`cube.cdts.run_...`).

### LandTrendr (Trajectory-based Disturbance)
Identify structural breakpoints in time-series (e.g., detecting exactly when deforestation occurred). CDTS scales LandTrendr to massive datasets using C++ OpenMP and Dask `map_blocks`.

```python
import numpy as np

# 1. Prepare annual NBR data (Time, Y, X)
years = np.array([2018, 2019, 2020, 2021, 2022, 2023])

# 2. Run LandTrendr across the entire Dask datacube natively
# max_segments=4 means up to 5 vertices (breakpoints) per pixel.
lt_results = cube_nbr.cdts.run_landtrendr(
    years=years, 
    max_segments=4, 
    pval_threshold=0.05, 
    n_jobs=-1
)

# Trigger Dask computation (runs the C++ core in parallel)
# Output shape is (2 * max_vertices, Y, X).
# The first half of the layers are Vertex Years, the second half are Fitted Values.
lt_array = lt_results.compute()

# Slicing the layers
max_vertices = 5
vertex_years = lt_array[0 : max_vertices, :, :]
vertex_fitted_values = lt_array[max_vertices : 2 * max_vertices, :, :]

# 3. Analyze disturbances (e.g., finding the biggest drop in NBR)
magnitude_of_change = np.diff(vertex_fitted_values, axis=0)

# Identify the segment with the most negative change (greatest vegetation loss)
biggest_loss_idx = np.argmin(magnitude_of_change, axis=0)

# Extract the specific year that this major disturbance began
disturbance_year = np.take_along_axis(
    vertex_years, 
    np.expand_dims(biggest_loss_idx, axis=0), 
    axis=0
).squeeze(0)

# Now you have a 2D Map of Disturbance Years ready to export!
```

### CCDC / COLD (Harmonic Modeling)
Extracts harmonic coefficients (Intercept, Slopes, Sine, Cosine) and detects intra-annual changes by fitting mathematical curves to multi-spectral data.

```python
# 1. Provide Julian dates and a Quality Assurance mask (Cloud/Shadow)
# cube_multi: 4D array (Bands, Time, Y, X)
# qa_mask: 3D array (Time, Y, X) with 0 for clear sky, 1 for clouds
dates_julian = np.array([100, 116, 132, 148, 164, 180])

# 2. Run CCDC directly as an xarray accessor
ccdc_results = cube_multi.cdts.run_ccdc(
    dates=dates_julian,
    qa_stack=qa_mask, # Automatically skips clouded pixels in regression
    max_segments=6, 
    conseq_anom=3, 
    return_coefs=True, 
    n_jobs=-1
)

# Run the C++ engine
coef_stack = ccdc_results.compute()

# 3. Extract Physical Attributes
# The returned coefficients (intercepts) can be used to map persistent physics.
# For example, mapping permanent water bodies by comparing Green and SWIR intercepts:
from cdts import extract_water_mask
water_map = extract_water_mask(
    coef_stack.values, 
    green_band_idx=1, 
    swir_band_idx=4
)
```

## 6. Pre and Post-Processing

Before classifying, it is highly recommended to smooth temporal trajectories. After classifying, pixel-based maps often suffer from noise. CDTS provides fast functions to regularize your data in both dimensions:

```python
from cdts import apply_savgol_filter, apply_majority_filter, apply_mmu_filter

# Temporal Smoothing (Savitzky-Golay, Whittaker, or Bayesian)
smoothed_array = apply_savgol_filter(raw_array, window_length=5, polyorder=2)

# Spatial Regularization (Mode filter)
regularized_map = apply_majority_filter(classified_map, size=3)

# Minimum Mapping Unit (MMU): Erase isolated patches smaller than 10 pixels
final_map = apply_mmu_filter(regularized_map, min_pixels=10)
```

## 7. Exporting Geospatial Data

Seamlessly dump predicted arrays back to the disk, preserving the metadata from the original STAC cube.

```python
from cdts import save_raster

save_raster(
    array=final_map, 
    output_path="output/land_cover.tif", 
    reference_cube=cube, # Copies Affine Transform and CRS
    nodata=255
)
```

---

## Architecture & Threading Safety

CDTS safely blends Python-based distributed workflows (Dask) with highly parallel C++ routines:
- **OpenMP CPU Scaling**: All C++ algorithms expose the `n_jobs` parameter. When `n_jobs=-1`, CDTS automatically reserves one CPU core (`std::max(1, max_threads - 1)`) to ensure the host Operating System remains responsive during intensive workloads.
- **Memory Footprint**: Algorithms like TWDTW are strictly optimized via a 2-Row Dynamic Programming algorithm, restricting mathematical matrices to the CPU's L1 cache and avoiding heavy allocations.
- **Cross-Platform Compatibility**: Uses safe `#ifdef _OPENMP` boundaries to gracefully fallback to single-threaded operations on macOS environments using Apple Clang (which lacks native `libomp`), allowing `pip install` to succeed universally.
