# Phenology Extraction

The `cdts` package features an incredibly fast and highly optimized C++ backend for **Phenology Extraction**, designed specifically to handle large-scale Earth Observation datasets. This module allows you to monitor and extract cyclical patterns in vegetation dynamics—essential for agriculture, forestry, and climate change studies.

Built heavily on robust concepts adapted from state-of-the-art tools (like the R package `phenofit`), our implementation leverages **Eigen** for sparse linear algebra and **OpenMP** for native multi-threading. This allows the extraction of land surface phenology metrics from gigabytes of satellite time series with unparalleled speed, bypassing the Python Global Interpreter Lock (GIL).

---

## 1. What is Land Surface Phenology?

Phenology is the study of periodic biological events in the animal and plant world as influenced by the environment, especially seasonal variations in temperature and precipitation. In remote sensing, **Land Surface Phenology (LSP)** refers to the seasonal pattern of variation in vegetated land surfaces observed from satellite imagery.

By analyzing vegetation indices (such as EVI or NDVI) over time, we can extract key phenological metrics:

- **SOS (Start of Season)**: The start of the vegetative cycle (e.g., spring green-up or planting).
- **POP (Peak of Season)**: The moment of maximum vegetative vigor (maximum canopy closure).
- **EOS (End of Season)**: The end of the vegetative cycle (senescence or harvesting).
- **LOS (Length of Season)**: The duration of the growing season, calculated as `EOS - SOS`.

These metrics are crucial for mapping crop types, predicting yields, detecting climate-induced shifts in ecosystems, and monitoring double-cropping agricultural systems.

---

## 2. The Extraction Workflow

Extracting phenology from noisy, cloud-contaminated satellite time series requires a robust, multi-step mathematical pipeline.

![Phenology Extraction Process](../assets/phenology_flow.jpg)

### 2.1. Time Series Smoothing

Raw satellite data is often noisy due to atmospheric interference, clouds, or sensor errors. The first step is to smooth the curve to extract the general seasonal trend.
- **Whittaker Smoother**: We use a highly optimized Sparse Matrix implementation of the Whittaker smoother. It balances fidelity to the original data with the smoothness of the resulting curve, controlled by a `lambda` parameter.
- **HANTS (Harmonic Analysis of Time Series)**: An alternative smoothing method that models the time series using sine and cosine waves (Fourier analysis), which is highly effective at removing cloud gaps.

### 2.2. Curve Fitting (Levenberg-Marquardt)

Once the general shape of the growing season is identified, we fit a mathematical model to it. This allows us to continuously evaluate the curve at any given day. The optimization is done using the **Levenberg-Marquardt** non-linear least squares algorithm.

We provide several asymmetric Gaussian and double-logistic functions:
- **`BECK` (Beck et al., 2006)**: Excellent for general forest and crop phenology.
- **`ELMORE` (Elmore et al., 2012)**: Handles varying winter backgrounds effectively.
- **`GU` (Gu et al., 2009)**: Uses an advanced recovery model.
- **`KLOSTERMAN` (Klosterman et al., 2014)**: Uses analytical geometry to detect transition dates.
- **`ZHANG` (Zhang et al., 2003)**: A piecewise logistic model.
- **`AG`**: Standard Asymmetric Gaussian.
- **`DL`**: Standard Double Logistic.

### 2.3. Metric Extraction Methods

Once the curve is fitted perfectly, how do we define the "Start", "Peak", and "End" of the season? The `cdts` C++ backend calculates and returns **19 distinct phenological variables simultaneously** for every season, covering all major state-of-the-art extraction methodologies at no extra computational cost:

- **Threshold Methods (`TRS`)**:
  - **`TRS2.sos` / `TRS2.eos`**: Start and End of Season defined when the curve reaches **20%** of its seasonal amplitude.
  - **`TRS5.sos` / `TRS5.eos`**: Start and End of Season defined at **50%** amplitude.
  - **`TRS6.sos` / `TRS6.eos`**: Start and End of Season defined at **60%** amplitude.
- **Derivative Method (`DER`)**:
  - **`DER.sos` / `DER.eos`**: Mathematically defined as the points where the rate of change of the curve (the 1st derivative) reaches its local maximum (spring green-up acceleration) and local minimum (senescence deceleration).
  - **`DER.pos`**: The exact day of the peak, where the 1st derivative crosses zero.
- **Gu Method (2nd Derivative)**:
  - **`UD`** (Upward), **`SD`** (Senescence Downward), **`DD`** (Downward), **`RD`** (Recovery Downward): Key transition points defined by the local maxima and minima of the curve's 2nd derivative.
- **Zhang Method (Curvature Rate)**:
  - **`Greenup`**, **`Maturity`**, **`Senescence`**, **`Dormancy`**: Transition dates extracted using the physical curvature formula `K = f'' / (1 + (f')^2)^1.5`, searching for local valleys and peaks of the curvature rate.
- **General**:
  - **`LOS`** (Length of Season): Duration of the season in days.
  - **`POP`** (Peak of Season): General peak location based on curve shape max values.

---

## 3. Practical Example: Processing a Raster (End-to-End)

The `cdts` package natively integrates with `xarray` through a custom accessor (`.cdts.run_phenology`). This abstracts away all the complex array reshaping and memory management, allowing you to process large MODIS/Landsat time series elegantly.

By default, the pipeline automatically maps the continuous days back into calendar DOYs if you pass `return_annual=True`.

```python
import rioxarray
import numpy as np
import pandas as pd
import cdts # Automatically registers the .cdts accessor in xarray
from cdts._core.phenology import CurveType

# 1. Load the dense time series raster (Shape: Time, Y, X)
ds = rioxarray.open_rasterio('MODIS_EVI_Series.tif')

# 2. Prepare the Time Array (Continuous Day of Year)
dates = pd.date_range(start='2001-01-01', periods=ds.shape[0], freq='16D')
dates_doy = np.array([d.timetuple().tm_yday + (d.year - 2001) * 365 for d in dates], dtype=np.float64)

# 3. Run the Phenology Engine directly on the xarray DataArray
# This leverages Dask internally for parallel out-of-core execution
print("Extracting 19 phenology metrics...")
metrics_da = ds.cdts.run_phenology(
    dates=dates_doy,
    curve_type=int(CurveType.BECK), # Use Beck's double logistic
    max_seasons=25,                 # Process 25 years of data
    whittaker_lambda=10.0,          # Whittaker smoothness parameter
    apply_whittaker=True,           # Apply Whittaker before fitting
    min_season_length=7,            # A season must last at least 7 days
    min_amplitude=0.0,              
    min_pixel_amplitude=0.1,        # Skip dead/water pixels entirely
    return_annual=True,             # Return variables aligned to calendar Years
    base_year=2001,
    n_jobs=14                       # Use 14 CPU cores (OpenMP)
).compute()

# 4. Save to disk using rioxarray
# metrics_da shape is (metric=19, year=25, y, x).
# We can loop through the 19 variables and export them as 25-band TIF files
metrics_da.rio.write_crs(ds.rio.crs, inplace=True)

for metric_name in metrics_da.metric.values:
    # Select the specific metric, resulting in a 3D array (year, y, x)
    single_metric_da = metrics_da.sel(metric=metric_name)
    
    # Save a multi-band TIF where each band is a year
    filename = f"cdts_{metric_name}.tif"
    single_metric_da.rio.to_raster(filename)
    print(f"Saved {filename}")
```

---

## 5. Advanced Configuration & Double Cropping

### Multiple Seasons (Double/Triple Cropping)
In regions with intense agricultural activity (like Mato Grosso, Brazil), a single pixel might feature two or even three distinct crop harvests within a single year (e.g., Soybeans followed by Corn).

To capture these dynamics directly without `return_annual=True`, simply increase `max_seasons`:
```python
metrics_tensor = ds.cdts.run_phenology(
    # ...
    max_seasons=3,
    return_annual=False
    # ...
)
```
This returns arrays of shape `(19_metrics, 3_seasons, Y, X)`. You can then map `season=0` as the first harvest, `season=1` as the second (safrinha), and so on.

### Adjusting Quality Control Parameters
- **`whittaker_lambda`**: Higher values create stiffer, smoother curves. Lower values allow the curve to bend sharply to follow the raw data closely. For 16-day composites, values between `1.0` and `5.0` are standard.
- **`min_season_length`**: Useful for filtering out high-frequency noise spikes that mistakenly look like a very short 2-day growing season.
- **`min_amplitude`**: Prevents the optimizer from fitting curves on background noise (e.g., bare soil that fluctuates slightly with rain). If the peak of the smoothed curve minus the base is less than this value, the season is rejected.

---

## 6. Execution Performance

The `cdts` phenology engine is engineered to maximize performance:
1. **No GIL**: The `fit_phenology_batch` completely releases the Python Global Interpreter Lock.
2. **OpenMP C++**: Pixel loops are chunked and scheduled dynamically across all logical CPU cores.
3. **No Dynamic Allocation**: Temporary vectors inside the optimization loop are pre-allocated and map directly to memory via Eigen.

**Operating System Notes:**
- **Windows / Linux**: OpenMP multi-threading works perfectly out of the box with `pip install cdts`.
- **macOS (Apple Silicon / Intel)**: Apple's default Clang compiler disables OpenMP. To achieve maximum performance, install the library (`brew install libomp`) and export the compilation flags *before* installing the package:
  ```bash
  export CFLAGS="-I$(brew --prefix libomp)/include"
  export CXXFLAGS="-I$(brew --prefix libomp)/include"
  export LDFLAGS="-L$(brew --prefix libomp)/lib -lomp"
  pip install cdts
  ```
