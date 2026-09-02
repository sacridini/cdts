# cdts - Change Detection Python Library 🌍🛰️

`cdts` is an ultra-fast, cloud-native Python framework for **Remote Sensing Time-Series Analysis and Change Detection**. 

Designed to replace heavy dependencies on Google Earth Engine, `cdts` handles the entire geospatial pipeline locally or on cloud clusters: from directly streaming satellite imagery via **STAC** APIs, to scaling memory lazily with **Dask/Xarray**, down to executing heavy statistical regression in native **C++**.

It provides state-of-the-art algorithms:
*   **LandTrendr** (Landsat-based detection of Trends in Disturbance and Recovery)
*   **CCDC** (Continuous Change Detection and Classification)
*   **COLD** (Continuous monitoring of Land Disturbance)

### 🐳 Enterprise / Production Deployment (Docker)
Because `cdts` relies on heavy C++ compilation and GPU-accelerated PyTorch, the easiest way to deploy it to the cloud or share it with other researchers is via our official Docker container.

```bash
# This will build the C++ engines, install PyTorch with CUDA, and launch a JupyterLab environment on port 8888.
docker-compose up --build
```

---

## ⚡ Core Architecture

1.  **C++ Engine (`pybind11` & `Eigen3`)**: The core statistical fitting (OLS, Robust Iteratively Reweighted Least Squares, Exact F-Statistics, Chi-Square CDFs) is fully written in C++ for maximum single-core speed.
2.  **Cloud-Native Data Fetching (`STAC` & `stackstac`)**: Download-free pipelines! Query AWS or Microsoft servers for imagery and stream only the exact pixels you need.
3.  **Horizontal Scaling (`xarray` & `dask`)**: Data is lazily chunked. Run processing over 100,000 km² without blowing up your RAM by distributing tasks across multiple CPU threads or remote Dask workers.

---

## 📡 Supported Cloud Data Services (STAC)

Because `cdts` relies on the open **SpatioTemporal Asset Catalog (STAC)** standard, it can pull time-series data from virtually any modern satellite provider.

| Provider / Service | Default ID in `cdts` | Highlights |
| :--- | :--- | :--- |
| **AWS Earth Search (Element84)** | `"earth_search"` | Free, public Sentinel-2 L2A and Landsat Collection 2. No authentication needed! |
| **Microsoft Planetary Computer** | `"planetary_computer"` | Huge catalog (ALOS, MODIS, NAIP, Sentinel, Landsat). *Note: Requires a free token for heavy downloads.* |
| **Brazil Data Cube (INPE)** | `"brazil_data_cube"` | High-quality ARD cubes for Brazil (CBERS-4/4A, Amazonia-1). *Note: Requires INPE token.* |
| **Copernicus Data Space** | Custom URL | The official European Space Agency hub for Sentinel 1/2/3/5P. |

---

## 🛠️ Installation

**Prerequisites:** Python 3.8+ and a C++ Compiler (GCC, Clang, or MSVC).

```bash
git clone https://github.com/your-username/cdts.git
cd cdts
pip install -e .
```
*(This will automatically compile the C++ backend and install Python dependencies like `xarray`, `dask`, `scikit-learn`, `rasterio`, and `pystac-client`)*.

---

## 🚀 Advanced Tutorial: End-to-End Pipeline

Here is a complete workflow demonstrating how to go from zero data to a classified map of persistent water using `cdts`.

### 1. Stream Virtual Data (STAC)
No downloading required. We define a Bounding Box in Mato Grosso (Brazil) and request 2 years of Sentinel-2 data.

```python
from cdts import build_time_series
import cdts.xarray_api # Registers the .cdts accessor on Xarray

# Builds a Dask-backed virtual datacube
cube = build_time_series(
    source="earth_search",       
    collection="sentinel-2-l2a",
    bbox=[-54.0, -12.0, -53.9, -11.9],
    start_date="2020-01-01",
    end_date="2022-12-31",
    cloud_cover_max=30,
    bands=["red", "green", "blue", "nir", "swir16"],
    epsg=3857,
    resolution=30
)
print(cube.shape) # e.g., (Time: 45, Bands: 5, Y: 1000, X: 1000)
```

### 2. Run CCDC / COLD (Distributed via Dask)
Using the Xarray accessor, we pipe the virtual cube directly into our C++ engine.

```python
# 'conseq_anom=6' activates the rigorous COLD algorithm (6 consecutive anomalies required to flag deforestation).
# This returns a lazy map of harmonic coefficients (Intercept, Slopes, Sine, Cosine).
ccdc_lazy_results = cube.cdts.run_ccdc(max_segments=6, conseq_anom=6, return_coefs=True)

# Actually execute the download + calculation in parallel threads
coef_stack = ccdc_lazy_results.compute()
```

### 3. Extract Physical Masks & Classify
With the harmonic coefficients calculated, we can derive physical parameters. For instance, extracting persistent rivers/lakes by comparing the Intercept coefficients of Green (index 1) and SWIR (index 4).

```python
from cdts import extract_water_mask, predict_synthetic_image

# 1. Physical Extraction: Isolate rivers and lakes
water_mask = extract_water_mask(coef_stack.values, green_band_idx=1, swir_band_idx=4)

# 2. Synthetic Imagery: Generate a cloud-free image for Julian Day 150
cloud_free_rgb = predict_synthetic_image(coef_stack.values, target_julian_day=150, num_bands=3)
```

If you have training data, you can run a full Random Forest classification on the coefficients:
```python
from cdts import train_ccdc_classifier, classify_ccdc_stack

clf = train_ccdc_classifier(X_train_data, y_train_labels)
classify_ccdc_stack(clf, coef_stack_path="output/coefs.tif", output_path="landcover.tif")
```

### 🧹 Pre & Post-Processing (Smoothing & Spatial Filters)
Before classifying, it is highly recommended to smooth the temporal trajectories to remove atmospheric noise. After classifying, pixel-based maps often suffer from "salt and pepper" noise. `cdts` provides fast functions to regularize your data in both dimensions:

```python
from cdts import apply_savgol_filter, apply_majority_filter, apply_mmu_filter

# 1. Temporal Smoothing: Apply Savitzky-Golay filter across the time axis (e.g. axis 0)
smoothed_cube = apply_savgol_filter(raw_cube, window_length=5, polyorder=2)

# ... (Run Classification to get `land_cover_map`) ...

# 2. Spatial Regularization: Force pixels to match their 3x3 neighborhood (Mode filter)
regularized_map = apply_majority_filter(land_cover_map, size=3)

# 3. Minimum Mapping Unit (MMU): Erase any isolated patches smaller than 10 pixels
final_map = apply_mmu_filter(regularized_map, min_pixels=10)
```

---

## 🌲 LandTrendr Specifics (FTV)
If your focus is on forest recovery, `cdts` natively supports LandTrendr. A key feature is **FTV (Fitted to Vertices)**, which allows you to find structural breakpoints in an index (like NBR) and apply them to smooth out noisy raw bands (like SWIR).

```python
from cdts import run_landtrendr, apply_vertices

# 1. Fit the trajectory on the main index to find the breakpoint years
vertices = run_landtrendr(years, nbr_time_series)
vertex_years = [v["year"] for v in vertices]

# 2. Force the raw SWIR band to conform to the NBR breakpoints!
swir_fitted = apply_vertices(vertex_years, years, raw_swir_time_series)
```

---

## 🧠 Deep Learning & Foundation Models (`cdts.ai`)
Beyond statistical algorithms like CCDC, `cdts` embraces the next generation of Spatio-Temporal Artificial Intelligence. Built on **PyTorch**, the new `cdts.ai` module provides modern neural network architectures tailored for earth observation:

### 1. U-TAE and TempCNN (Time-Series Neural Networks)
Instead of processing individual pixels, **U-TAE** consumes entire 3D Data Cubes (Spatial + Temporal) simultaneously to naturally ignore cloud noise. If you prefer pixel-based time-series classification, `cdts` also provides **TempCNN**, a lightweight 1D-CNN (inspired by INPE's `sits` package) that is incredibly fast to train.

```python
from cdts.ai import UTAE, TempCNN
import torch

# U-TAE: Input shape (Batch, Time, Channels, Height, Width)
model_3d = UTAE(in_channels=6, num_classes=5)

# TempCNN: Input shape (Batch, Channels, Time)
model_1d = TempCNN(in_channels=6, num_classes=5)
```

### 2. Bi-Temporal Siamese CNNs
Perfect for disaster mapping (floods, fires, landslides). A Siamese Network processes a T0 ("Before") image and a T1 ("After") image through shared convolutional weights, then extracts absolute differences deep in the feature space.

```python
from cdts.ai import SiameseChangeDetector

model = SiameseChangeDetector(in_channels=4, num_classes=2)
img_before = torch.randn(1, 4, 512, 512)
img_after = torch.randn(1, 4, 512, 512)

# Outputs a spatial change map directly
change_map = model(img_before, img_after)
```

### 3. Geospatial Foundation Models (ViT)
`cdts.ai.GeoFoundationViT` acts as a wrapper/stub to plug in large-scale Vision Transformers (like the **NASA/IBM Prithvi** model). It enables you to take pre-trained planetary representations and fine-tune them for specific downstream tasks like deforestation or crop classification.

### 4. Specialized Change Detection Losses
Remote sensing datasets are highly imbalanced (usually >99% unchanged pixels). Standard Cross-Entropy fails here. `cdts.ai.losses` provides battle-tested loss functions specifically for Change Detection:
```python
from cdts.ai.losses import FocalLoss, TverskyLoss, ContrastiveSiameseLoss

# Focal Loss: Forces the network to focus gradients on hard-to-detect subtle changes
criterion1 = FocalLoss(alpha=0.25, gamma=2.0)

# Tversky Loss: Penalizes False Negatives heavier than False Positives (beta=0.7)
criterion2 = TverskyLoss(alpha=0.3, beta=0.7)
```

---

## ☁️ Tmask: Time-Series Cloud Masking
Before running CCDC or deep learning models, you must have clean data. While STAC APIs provide QA bands (like Fmask), `cdts` natively implements **Tmask** (Zhu & Woodcock 2014) to dynamically find undetected clouds and shadows.

Tmask runs a robust harmonic regression on Green and SWIR bands. If a pixel suddenly flashes bright green or dark SWIR without altering the long-term structural trajectory, it is flagged as noise.

```python
from cdts import apply_tmask_stack

# Outputs a Boolean mask (True = Clear, False = Cloud/Shadow)
# Uses robust Huber regression under the hood
clear_sky_mask = apply_tmask_stack(dates_julian, green_cube, swir_cube)
```

---

## 💾 Exporting & Saving Data (IO)
Instead of dealing with complex `rasterio` profiles, `cdts` includes a powerful `save_raster` utility that automatically extracts the geotransform and CRS from the downloaded STAC Datacube and exports your PyTorch/Numpy predictions into professional, ready-to-use GeoTIFFs.

```python
from cdts import save_raster, get_georef

# Extract geospatial reference explicitly if needed
geo_info = get_georef(cube)
print(geo_info["crs"]) # e.g. "EPSG:3857"

# Or save the array seamlessly using the original cube as reference!
# Handles 2D, 3D, and even 4D cubes out-of-the-box.
save_raster(prediction_array, "output/final_map.tif", reference_cube=cube, nodata=255)
```

---

## 📚 End-to-End Examples
We provide **6 complete example scripts** in the `examples/` directory. They cover everything from downloading STAC data to temporal smoothing and classification using both statistical and AI models. Running these scripts will automatically output georeferenced GeoTIFFs into `examples/data/`.

- `example_01_landtrendr.py` (LandTrendr Disturbance Year)
- `example_02_ccdc_cold.py` (CCDC / COLD Synthetic Image Generation)
- `example_03_ai_siamese.py` (Siamese Neural Network)
- `example_04_ai_utae.py` (U-TAE 4D processing)
- `example_05_ai_tempcnn.py` (TempCNN time-series)
- `example_06_ai_vit.py` (Geospatial Foundation Model)

```bash
# Try one!
python examples/example_05_ai_tempcnn.py
```

---

## 💻 Command Line Interface (CLI)

Prefer the terminal? If you already have a massive GeoTIFF locally, you can process it chunk-by-chunk using the CLI.

```bash
# Run LandTrendr (Extracting just the break years)
cdts landtrendr input_stack.tif output_folder/ \
    --start-year 1990 --event-type loss --jobs -1

# Run COLD (Extracting the harmonic coefficient matrix)
cdts ccdc multi_band_stack.tif output_folder/ \
    --num-bands 6 --max-segments 6 --cold --jobs -1
```
