# CDTS: Change Detection Time-Series for Python

**CDTS** is an ultra-fast, cloud-native Python framework for **Remote Sensing Time-Series Analysis and Change Detection**. 

Designed to overcome heavy dependencies on platforms like Google Earth Engine, CDTS handles the entire geospatial pipeline locally or on cloud clusters. It scales seamlessly from directly streaming satellite imagery via STAC APIs, to scaling memory lazily with Dask and Xarray, down to executing heavy statistical regression in native C++.

## Key Capabilities

*   **Cloud-Native Data Fetching:** Query AWS, Microsoft Planetary Computer, or other STAC-compliant servers for imagery, streaming only the exact pixels needed without full downloads.
*   **High-Performance Computing:** The core statistical fitting (OLS, Robust Iteratively Reweighted Least Squares, Exact F-Statistics, Chi-Square CDFs) is fully written in C++ (via `pybind11` and `Eigen3`) for maximum single-core speed.
*   **Horizontal Scaling:** Leverage `xarray` and `dask` to lazily chunk data. Distribute tasks across multiple CPU threads or remote Dask workers to process large areas without memory exhaustion.
*   **Deep Learning & Foundation Models (`cdts.ai`):** Built on PyTorch, providing modern neural network architectures tailored for earth observation, including U-TAE, TempCNN, Bi-Temporal Siamese CNNs, and wrappers for Geospatial Foundation Models (ViT).

## Algorithms

CDTS natively implements industry-standard change detection algorithms:

*   **LandTrendr**: Landsat-based detection of Trends in Disturbance and Recovery.
*   **CCDC**: Continuous Change Detection and Classification.
*   **COLD**: Continuous monitoring of Land Disturbance.
*   **Tmask**: Time-Series Cloud Masking to dynamically find undetected clouds and shadows.

## Supported Cloud Data Services

CDTS relies on the SpatioTemporal Asset Catalog (STAC) standard and can pull time-series data from virtually any modern satellite provider, including AWS Earth Search, Microsoft Planetary Computer, Brazil Data Cube, and Copernicus Data Space.
