<div align="center" markdown>

<img src="assets/logo.png" alt="CDTS Logo" width="220">

# CDTS

**Change Detection and Time-Series for Python**

[![Build and Publish Wheels](https://github.com/sacridini/cdts/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sacridini/cdts/actions/workflows/build_wheels.yml)
[![Tests](https://github.com/sacridini/cdts/actions/workflows/tests.yml/badge.svg)](https://github.com/sacridini/cdts/actions/workflows/tests.yml)
[![Docs](https://github.com/sacridini/cdts/actions/workflows/docs.yml/badge.svg)](https://github.com/sacridini/cdts/actions/workflows/docs.yml)
[![PyPI version](https://badge.fury.io/py/cdts.svg)](https://badge.fury.io/py/cdts)

An ultra-fast, cloud-native Python library for **Remote Sensing Time-Series Analysis and Change Detection**.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[Browse Tutorials](tutorials/index.md){ .md-button }
[View on GitHub :material-github:](https://github.com/sacridini/cdts){ .md-button }

</div>

---

Designed to overcome heavy dependencies on platforms like Google Earth Engine, CDTS handles the entire geospatial pipeline locally or on cloud clusters. It scales seamlessly from directly streaming satellite imagery via STAC APIs, to lazily scaling memory with Dask and Xarray, down to executing heavy statistical regression in native C++.

## Quick Install

```bash
pip install cdts
```

Pre-compiled wheels are provided for Windows, macOS, and Linux — no C++ compiler required. See the [installation guide](getting-started/installation.md) for GPU, macOS OpenMP, and from-source options.

## Key Capabilities

<div class="grid cards" markdown>

-   :material-cloud-download:{ .lg .middle } **Cloud-Native Data Fetching**

    ---

    Query AWS, Microsoft Planetary Computer, or other STAC-compliant servers for imagery, streaming only the exact pixels needed without full downloads.

-   :material-speedometer:{ .lg .middle } **High-Performance Computing**

    ---

    Core statistical fitting (OLS, Robust IRLS, Exact F-Statistics, Chi-Square CDFs) is fully written in C++ via `pybind11` and `Eigen3` for maximum single-core speed.

-   :material-server-network:{ .lg .middle } **Horizontal Scaling**

    ---

    Leverage `xarray` and `dask` to lazily chunk data, distributing work across CPU threads or remote Dask workers to process large areas without memory exhaustion.

-   :material-brain:{ .lg .middle } **Deep Learning & Foundation Models**

    ---

    Built on PyTorch, `cdts.ai` provides modern architectures for earth observation — U-TAE, TempCNN, Bi-Temporal Siamese CNNs — plus wrappers for Geospatial Foundation Models (ViT).

</div>

## Algorithms

CDTS natively implements industry-standard algorithms for **Time-Series Analysis**, **Change Detection**, and **Deep Learning**:

| Algorithm | Category | What it does |
|---|---|---|
| [LandTrendr](tutorials/landtrendr.md) | Change Detection | Landsat-based detection of trends in disturbance and recovery. |
| [CCDC](tutorials/ccdc.md) | Change Detection | Continuous Change Detection and Classification via robust harmonic modeling. |
| [BFAST family](tutorials/bfast.md) | Change Detection | Iterative trend + season break detection, near-real-time monitoring ([Monitor](tutorials/bfast_monitor.md)), and single-pass multi-breakpoint detection ([Lite](tutorials/bfast_lite.md)). |
| [Tmask](tutorials/tmask.md) | Change Detection | Time-series cloud masking to dynamically find clouds and shadows missed by native QA bands. |
| [TWDTW](tutorials/twdtw.md) | Time-Series Analysis | Time-Weighted Dynamic Time Warping for pattern matching against reference curves. |
| [Mann-Kendall](tutorials/mann_kendall.md) | Time-Series Analysis | Non-parametric trend test and Theil-Sen slope estimation for greening/browning trends. |
| [Phenology Extraction](tutorials/phenology.md) | Time-Series Analysis | Simultaneous phenological metrics from optimized curve-fitting models. |
| [SOM](tutorials/som.md) | Time-Series Analysis | Batch Self-Organizing Maps for unsupervised clustering of spectral-temporal arrays. |
| [U-TAE / LTAE](tutorials/utae.md) | Deep Learning | Attention-based architectures for spatio-temporal satellite image classification. |
| [Siamese Networks](tutorials/siamese.md) | Deep Learning | Bi-temporal CNNs for pixel-wise change detection. |
| [GeoFoundationViT](tutorials/geo_foundation_vit.md) | Deep Learning | Wrappers for Vision Transformer geospatial foundation models. |

## Supported Cloud Data Services

CDTS relies on the SpatioTemporal Asset Catalog (STAC) standard and can pull time-series data from virtually any modern satellite provider, including AWS Earth Search, Microsoft Planetary Computer, Brazil Data Cube, and Copernicus Data Space — plus [Google Earth Engine](tutorials/gee-downloads.md) integration.

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Install CDTS**

    ---

    Get up and running with pip, Docker, or a from-source build.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-school:{ .lg .middle } **Follow a Tutorial**

    ---

    Full walkthroughs with theory, code, and validation against reference implementations.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-console:{ .lg .middle } **Use the CLI**

    ---

    Run every core algorithm as a `cdts` subcommand from bash scripts, cron jobs, or HPC environments.

    [:octicons-arrow-right-24: CLI Reference](cli.md)

-   :material-api:{ .lg .middle } **Browse the API**

    ---

    Full reference for every public class and function in the `cdts` package.

    [:octicons-arrow-right-24: API Reference](api.md)

</div>
