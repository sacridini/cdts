<p align="center">
  <img src="docs/assets/logo.png" alt="CDTS Logo" width="400">
</p>

# CDTS: Change Detection and Time Series for Python

[![Build Wheels](https://github.com/sacridini/cdts/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sacridini/cdts/actions/workflows/build_wheels.yml)
[![Tests](https://github.com/sacridini/cdts/actions/workflows/tests.yml/badge.svg)](https://github.com/sacridini/cdts/actions/workflows/tests.yml)
[![Docs](https://github.com/sacridini/cdts/actions/workflows/docs.yml/badge.svg)](https://github.com/sacridini/cdts/actions/workflows/docs.yml)
[![PyPI version](https://badge.fury.io/py/cdts.svg)](https://badge.fury.io/py/cdts)

**CDTS** is a high-performance Python package for Earth Observation (EO) data cube processing and time series analysis. It bridges the gap between modern cloud-native data formats (STAC, Xarray, Dask) and state-of-the-art pixel-based trajectory algorithms (TWDTW, CCDC, LandTrendr). 

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
  - **BFAST Monitor**: Near-real-time structural change monitoring (ported from R's `bfast`/`strucchangeRcpp`).
- **Deep Learning (`cdts.ai`):** Pre-built PyTorch architectures tailored for spatio-temporal Earth Observation (U-TAE, TempCNN, Siamese Networks).

---

## Installation

```bash
pip install cdts
```
*(Note: Wheels are provided for Windows, Linux, and macOS. macOS runs in single-threaded mode by default due to Apple Clang lacking OpenMP).*

**For macOS users who want C++ OpenMP multi-threading:**
Apple's default Clang compiler disables OpenMP. To achieve maximum performance and enable multi-threading, you must install the `libomp` library and manually export the compilation flags *before* forcing a local compilation:

```bash
brew install libomp
export CFLAGS="-I$(brew --prefix libomp)/include"
export CXXFLAGS="-I$(brew --prefix libomp)/include"
export LDFLAGS="-L$(brew --prefix libomp)/lib -lomp"
pip install --no-binary cdts cdts
```

---

## Cloud-Native ARD Cubes (STAC)

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

## Temporal Regularization

Algorithms like TWDTW, SOM, and Deep Learning expect temporally aligned data. `cdts` natively regularizes irregular STAC acquisitions.

```python
from cdts import regularize_time_series

# Aggregate observations into 16-day Medoid composites 
# (Maintains xarray lazy evaluation via Dask graphs)
cube_16d = regularize_time_series(cube, freq="16D", method="medoid")
```

## Change Detection (LandTrendr & CCDC)

Continuous structural monitoring using robust breakpoint and harmonic regression models directly on xarray Datacubes via pandas-like accessors (`cube.cdts.run_...`).

### LandTrendr (Trajectory-based Disturbance)
Identify structural breakpoints in time-series (e.g., detecting exactly when deforestation occurred). CDTS scales LandTrendr to massive datasets using C++ OpenMP and Dask `map_blocks`.

```python
import numpy as np
from cdts.metrics import extract_events

# 1. Prepare annual NBR data (Time, Y, X)
years = np.array([2018, 2019, 2020, 2021, 2022, 2023])

# 2. Run LandTrendr across the entire Dask datacube natively
lt_results = cube_nbr.cdts.run_landtrendr(
    years=years, 
    max_segments=4, 
    pval_threshold=0.05, 
    n_jobs=-1
)

# Trigger Dask computation (runs the C++ core in parallel)
# Output shape is (2 * max_vertices, Y, X).
lt_array = lt_results.compute()

# 3. Analyze disturbances (e.g., finding the biggest drop in NBR)
events = extract_events(
    vertices_stack=lt_array, 
    event_type="loss",      # Look for drops in the index (e.g., vegetation loss)
    sort_by="greatest",     # Get the segment with the largest magnitude
    min_magnitude=0.1       # Optional noise filter
)

# You now have 2D maps ready to be exported to GeoTIFF!
yod_map = events["year"]        # Year of Disturbance (YOD)
mag_map = events["magnitude"]   # Magnitude of the disturbance
dur_map = events["duration"]    # How many years the disturbance took
pre_map = events["pre_val"]     # Value before disturbance
post_map = events["post_val"]   # Value after disturbance
```

### CCDC / COLD (Harmonic Modeling)
Extracts harmonic coefficients (Intercept, Slopes, Sine, Cosine) and detects intra-annual changes by fitting mathematical curves to multi-spectral data.

```python
import numpy as np
from cdts.ccdc import predict_synthetic_image
from cdts.classify import train_ccdc_classifier, classify_ccdc_stack

# 1. Provide Julian dates and a Quality Assurance mask (Cloud/Shadow)
# cube_multi: 4D array (Bands, Time, Y, X)
# qa_mask: 3D array (Time, Y, X) with 0 for clear sky, 1 for clouds
dates_julian = np.array([100, 116, 132, 148, 164, 180])

# 2. Run CCDC directly as an xarray accessor
ccdc_results = cube_multi.cdts.run_ccdc(
    dates=dates_julian,
    qa_stack=qa_mask, # Automatically skips clouded pixels in regression
    max_segments=6, 
    return_coefs=True, 
    n_jobs=-1
)

# Run the C++ engine to generate the harmonic coefficient stack
coef_stack = ccdc_results.compute()

# 3. Generate Synthetic Images (Harmonic Reconstruction)
# Predict what the surface should look like on any arbitrary date without clouds!
# Output shape: (Bands, Y, X)
synthetic_image = predict_synthetic_image(
    ccdc_coefs_stack=coef_stack.values, 
    target_julian_day=200, # Predict for Julian day 200
    num_bands=6
)

# 4. Land Cover Classification using the Harmonic Coefficients
# Train a Random Forest using harmonic coefficients as features
rf_model = train_ccdc_classifier(
    X_train=training_coefs, # Your extracted training samples
    y_train=training_labels, 
    n_estimators=100
)

# Classify the entire CCDC cube into a categorical land cover map block-by-block
# (Handles memory efficiently by reading/writing chunks)
classify_ccdc_stack(
    clf=rf_model,
    coef_stack_path="output/ccdc_coefs.tif",
    output_path="output/land_cover_map.tif",
    chunk_size=512
)
```

## Phenology Extraction

Extract 19 simultaneous phenological metrics (Gu, Zhang, Thresholds, Derivatives, LOS, POP) across massive datasets using optimized C++ curve-fitting models (Beck, Elmore, Gu, Zhang, Asymmetric Gaussian, Double Logistic) over Dask clusters.

> The smoothing, curve-fitting, and metric-extraction methodology is based on the R package [`phenofit`](https://github.com/eco-hydro/phenofit) (Kong *et al.*, 2022, *Methods in Ecology and Evolution*, [doi:10.1111/2041-210X.13870](https://doi.org/10.1111/2041-210X.13870)), reimplemented in C++/Eigen/OpenMP. See the [Phenology tutorial](https://sacridini.github.io/cdts/tutorials/phenology/#7-references) for the full reference list and a real-world walkthrough.

```python
import numpy as np
from cdts._core.phenology import CurveType

# 1. Provide dates corresponding to the time steps
dates_julian = np.arange(1, 366, 16) # Day of year

# 2. Run phenology curve fitting natively via the cdts accessor
pheno_results = cube_16d.cdts.run_phenology(
    dates=dates_julian,
    curve_type=int(CurveType.BECK), # Enum mapping to CurveType::BECK
    max_seasons=2,                  # Extract up to 2 growing seasons per year
    
    # Smoothing Configuration
    apply_whittaker=False,          # Turn off Whittaker
    apply_hants=True,               # Use HANTS (Fourier-based) instead
    hants_frequencies=3,
    
    # Fine-Grained Season Control
    min_season_length=90,           # Ignore noisy peaks shorter than 90 days
    min_amplitude=0.2,              # Ignore seasons with less than 0.2 NDVI growth
    return_annual=False,            # Return as purely sequential seasons
    
    n_jobs=-1                       # C++ multithreading (leave cores for OS)
)

# Trigger computation (runs C++ optimizer across Dask blocks)
# Output shape: (metric=19, season=2, Y, X)
pheno_array = pheno_results.compute()

# Extract Zhang's Greenup transition date for the first season
greenup_map = pheno_array.sel(metric="Greenup", season=0)
```

**Down-weighting cloud/snow-contaminated observations:** `cdts.qc` decodes a sensor's QA/QC band into per-observation reliability weights in `[0, 1]` (ported from phenofit's `qcFUN.R`), which feed the Whittaker/HANTS smoothing and the iterative curve fit instead of trusting every observation equally:

```python
from cdts.qc import qc_modis_summary

# qa_cube: (time, y, x) MOD13 SummaryQA band, aligned with cube_16d
weights = qc_modis_summary(qa_cube)  # 0=good, 1=marginal, 2=snow/ice, 3=cloudy -> [1.0, 0.5, 0.2, 0.2]

pheno_results = cube_16d.cdts.run_phenology(
    dates=dates_julian,
    curve_type=int(CurveType.BECK),
    weights=weights,       # down-weights unreliable observations during smoothing/fitting
    season_retry=True,     # relax the trough threshold once if a pixel finds no season at all
)
```

## Trend Analysis (Mann-Kendall)

Pixel-wise Mann-Kendall trend test + Theil-Sen slope, ported from [`pymannkendall`](https://github.com/mmhs013/pymannkendall) (Hussain & Mahmud, 2019) to a C++/OpenMP backend, with the same Dask distribution strategy as Phenology Extraction. Useful for "is there a statistically significant greening/browning trend at this pixel?" questions on multi-year composite stacks. See the [Mann-Kendall tutorial](https://sacridini.github.io/cdts/tutorials/mann_kendall/) for the full method comparison (autocorrelation-corrected variants, seasonal test) and a real-world walkthrough.

```python
import numpy as np

# annual_ndvi: (year, y, x) DataArray, one max-NDVI composite per year
trend = annual_ndvi.cdts.run_mann_kendall(
    method="hamed_rao",  # autocorrelation-corrected (recommended for annual composites)
    alpha=0.05,
)

result = trend.compute()
slope_map = result.sel(metric="slope")        # NDVI change per year
significant = result.sel(metric="h") == 1.0   # statistically significant at alpha=0.05
declining = (result.sel(metric="trend") == -1) & significant
```

## Change Monitoring (BFAST Monitor)

Pixel-wise near-real-time structural change monitoring, ported from the R package [`bfast`](https://github.com/bfast2/bfast) (Verbesselt *et al.*) to a C++/OpenMP backend, with the same Dask distribution strategy as Mann-Kendall. Unlike LandTrendr/CCDC (retrospective, whole-series segmentation), `bfastmonitor` fits a trend+harmonic model on a stable history period and asks "is a disturbance happening *right now*, in the most recent observations?" — verified bit-for-bit-scale accurate against R's `bfastmonitor()`. See the [BFAST Monitor tutorial](https://sacridini.github.io/cdts/tutorials/bfast_monitor/) for the full method background and scope (only `type="OLS-MOSUM"` + `history="all"` are ported so far).

```python
# annual_ndvi: (time, y, x) DataArray, 16-day composites (frequency=23/year) from 2010
result = annual_ndvi.cdts.run_bfast_monitor(
    start_time=2010.0,
    monitor_start_time=2022.0,  # monitor everything from 2022 onward
    frequency=23,
)

result = result.compute()
disturbed = result.sel(metric="has_break") == 1.0
break_time = result.sel(metric="breakpoint")  # fractional-year time of the first detected break
```

## Time-Series Classification (TWDTW)

The C++ TWDTW engine handles multivariate sequences simultaneously using Eigen's $L^2$ norms and aggressively skips non-matching pixels using $O(N)$ Lower Bounding techniques.

```python
from cdts.twdtw import classify_twdtw
import numpy as np

# 1. Prepare your regularized data (Y, X, Time, Bands) and temporal axis
dates = np.arange(1, 366, 16) # Day of the year (DOY) for a 16-day composite

# 2. Extract or define temporal patterns (Signatures)
# A signature is a 1D or 2D array representing the expected phenological curve of a class.
# Example: 23 time steps, 4 bands (Red, Green, Blue, NIR)
forest_sig = np.random.rand(23, 4)  
soy_sig = np.random.rand(23, 4)

patterns = {
    "Forest": (forest_sig, dates),
    "Agriculture": (soy_sig, dates)
}

# 3. Run the TWDTW Classifier using the C++ OpenMP engine
# It calculates the multi-dimensional distance using the L2 Norm (Euclidean) 
# and aligns the series dynamically in time, bounded by max_time_warp.
classes_map, dist_map, class_names = classify_twdtw(
    values_array=cube_16d.values, 
    dates_array=dates, 
    patterns=patterns, 
    alpha=0.1,             # Steepness of the time penalty
    beta=0.05,             # Midpoint of the time penalty
    max_time_warp=60,      # Max allowed temporal shift in days
    n_jobs=-1              # Use all CPU cores minus 1 to keep OS responsive
)

# 4. Filter predictions by similarity (distance)
# TWDTW distance represents similarity (lower is better).
# Mask out pixels that matched poorly with all known signatures (Unclassified)
max_acceptable_distance = 15.0
final_classification = np.where(
    dist_map < max_acceptable_distance, 
    classes_map, 
    -1 # Assign -1 for Unclassified/Unknown pixels
)
```

## Unsupervised Clustering (SOM)

Unsupervised classification and dimensionality reduction of time series using a fast Batch SOM algorithm implemented in C++.

```python
from cdts.ai import SOM

# Flatten cube to (Pixels, Features)
X_train = cube_16d.values.reshape(-1, cube_16d.shape[2] * cube_16d.shape[3])

# Train a 10x10 SOM grid
som = SOM(x=10, y=10, input_len=X_train.shape[1])
som.train(X_train, num_iters=100, n_jobs=-1)

# Predict Best Matching Units (BMUs) for new data
bmus = som.predict(X_train, n_jobs=-1)
```

## Pre and Post-Processing

Before classifying, it is highly recommended to smooth temporal trajectories. After classifying, pixel-based maps often suffer from noise. CDTS provides fast functions to regularize your data in both dimensions:

```python
from cdts import apply_savgol_filter, apply_majority_filter, apply_mmu_filter, save_raster
from cdts.smooth import apply_whittaker_filter

# Temporal Smoothing: Savitzky-Golay (fast, general-purpose)...
smoothed_array = apply_savgol_filter(raw_array, window_length=5, polyorder=2)

# ...or Whittaker (often better for NDVI/EVI, supports per-observation weights)
smoothed_array = apply_whittaker_filter(raw_array, lmbd=10.0, weights=clear_sky_weights)

# Spatial Regularization (Mode filter)
regularized_map = apply_majority_filter(classified_map, size=3)
save_raster(regularized_map, "results/classified_regularized.tif", reference_cube=cube)

# Minimum Mapping Unit (MMU): operates on a GeoTIFF on disk, not an in-memory array
# Erase isolated patches smaller than 11 pixels
apply_mmu_filter(
    input_path="results/classified_regularized.tif",
    output_path="results/classified_final.tif",
    mmu_pixels=11,
)
```

## Exporting Geospatial Data

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
