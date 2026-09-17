# Tutorials

CDTS covers a lot of ground — from classic pixel-based change detection to deep learning and cloud data acquisition. Pick the group below that matches what you're trying to do; each tutorial is self-contained and covers theory, data prep, a full code walkthrough, and validation against reference implementations where one exists.

<div class="grid cards" markdown>

-   :material-chart-timeline-variant:{ .lg .middle } **Trajectory & Change Detection**

    ---

    Pixel-based statistical algorithms that model a time series' trajectory to flag disturbances, breakpoints, and land-cover change.

    - [LandTrendr](landtrendr.md)
    - [CCDC](ccdc.md)
    - [Tmask](tmask.md)
    - [BFAST](bfast.md)
    - [BFAST Monitor](bfast_monitor.md)
    - [BFAST Lite](bfast_lite.md)

-   :material-chart-bell-curve:{ .lg .middle } **Time-Series Analysis**

    ---

    Tools for comparing, clustering, and extracting statistical patterns from time series — independent of any particular change-detection algorithm.

    - [TWDTW (Time-Weighted DTW)](twdtw.md)
    - [Phenology Extraction](phenology.md)
    - [Mann-Kendall Trend Test](mann_kendall.md)
    - [SOM (Self-Organizing Maps)](som.md)

-   :material-brain:{ .lg .middle } **AI & Deep Learning**

    ---

    Native PyTorch architectures for per-pixel classification, spatio-temporal segmentation, and bi-temporal change detection, plus foundation-model transfer learning.

    - [Overview](ai.md)
    - [LTAE & LightTAE](ltae.md)
    - [UTAE](utae.md)
    - [TempCNN](tempcnn.md)
    - [Siamese Change Detector](siamese.md)
    - [GeoFoundationViT](geo_foundation_vit.md)

-   :material-cloud-download:{ .lg .middle } **Data Acquisition & Integration**

    ---

    Streaming imagery straight from cloud catalogs into analysis-ready data cubes, without full downloads.

    - [STAC & ARD Integration](stac-downloads.md)
    - [Google Earth Engine (GEE)](gee-downloads.md)

-   :material-server:{ .lg .middle } **Infrastructure**

    ---

    Scaling any of the above from a single machine to a distributed cluster.

    - [Parallel & Cloud Processing](parallel-cloud-processing.md)

</div>

## Not sure where to start?

- Want to detect disturbances or land-cover change over time? Start with [LandTrendr](landtrendr.md) or [CCDC](ccdc.md).
- Have labeled data and want to train a classifier? Start with [AI & Deep Learning](ai.md).
- Need to pull satellite imagery into a data cube first? Start with [STAC & ARD Integration](stac-downloads.md) or [Google Earth Engine](gee-downloads.md).
- Your area is too large to process on one machine? See [Parallel & Cloud Processing](parallel-cloud-processing.md).
