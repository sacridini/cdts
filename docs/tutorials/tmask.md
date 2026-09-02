# Tmask: Time-Series Cloud Masking

## 1. Introduction to Tmask

While algorithms like LandTrendr and CCDC are incredibly powerful for detecting land cover changes, their accuracy is entirely dependent on having clean, cloud-free data. Traditional cloud masking algorithms (like Fmask or Sen2Cor) operate on a single image at a time. This often leads to missed thin clouds, small cloud shadows, or false positives over bright urban areas.

**Tmask (Time-series based Automated Cloud Masking)** solves this by using the temporal dimension. By understanding the natural seasonal phenology of a pixel across time, it can accurately identify anomalies that represent missed clouds or shadows.

### Background and References
Developed alongside CCDC by Zhe Zhu and Curtis Woodcock, Tmask estimates a robust harmonic (Fourier) baseline for the Green and SWIR (Shortwave Infrared) bands. 
- **Clouds** generally exhibit abnormally high reflectance in the Green band compared to the seasonal expectation.
- **Shadows** exhibit abnormally low reflectance in the SWIR band compared to the seasonal expectation.

- **Original Paper**: [Zhu, Z. and Woodcock, C.E., 2014. Automated cloud, cloud shadow, and snow detection in multitemporal Landsat data: An algorithm designed specifically for monitoring land cover change. Remote Sensing of Environment, 152, pp.217-234.](https://doi.org/10.1016/j.rse.2014.06.012)

---

## 2. Using Tmask in CDTS

The `cdts` package provides an easy-to-use implementation of Tmask designed to generate QA (Quality Assessment) masks dynamically, which can then be fed directly into CCDC or LandTrendr.

### Step 2.1: Loading the Required Bands

Tmask strictly requires the **Green** band and the **SWIR** band (typically SWIR1, ~1.6 µm) stacked across time.

```python
import numpy as np
import rasterio
from datetime import datetime

# Define dates
dates_str = ["2020-01-15", "2020-02-01", "2020-02-17", "2020-03-05", "2020-03-21"]
dates_julian = np.array([datetime.strptime(d, "%Y-%m-%d").toordinal() for d in dates_str])

# Load raster stacks (Shape: Time, Rows, Cols)
# Note: You can extract these from a larger stacked GeoTIFF
with rasterio.open("data/green_band_stack.tif") as src:
    green_stack = src.read()
    profile = src.profile

with rasterio.open("data/swir_band_stack.tif") as src:
    swir_stack = src.read()

print(f"Stack shape: {green_stack.shape}")
```

### Step 2.2: Applying the Algorithm

Use the `apply_tmask_stack` function from the `cdts.tmask` module.

```python
from cdts.tmask import apply_tmask_stack

# The default scale_factor is 10000.0 (standard for Landsat/Sentinel-2 L2A).
# If your data is already float reflectance (0.0 - 1.0), set scale_factor=1.0.

print("Running Tmask...")
qa_mask_stack = apply_tmask_stack(
    dates=dates_julian,
    green_stack=green_stack,
    swir_stack=swir_stack,
    scale_factor=10000.0
)
print("Masking complete!")
```

### Understanding the Output

The `qa_mask_stack` is a boolean (True/False) NumPy array with the exact same shape as the input stacks `(Time, Rows, Cols)`.

*   **`True`**: The pixel is considered **CLEAR** (Valid observation).
*   **`False`**: The pixel is considered **CLOUD or SHADOW** (Anomaly detected).

You can easily invert this mask depending on the specific requirements of the downstream algorithm. For example, CCDC expects `0` for clear and `1` for clouds:

```python
# Convert boolean mask to uint8 for CCDC (0 = clear, 1 = cloud/shadow)
ccdc_qa_mask = (~qa_mask_stack).astype(np.uint8)
```

### Step 2.3: Exporting the QA Mask

You can save this dynamically generated mask back to a GeoTIFF using the `cdts.io.save_raster` function, ensuring it's ready to be used alongside your spectral data.

```python
from cdts.io import save_raster

# Save the multi-temporal QA mask
save_raster(
    array=ccdc_qa_mask,
    output_path="results/tmask_qa_stack.tif",
    crs=profile['crs'],
    transform=profile['transform']
)
```

## 3. Best Practices

1. **Input Data**: Tmask works best as a *secondary* pass. It is highly recommended to apply a preliminary mask (like the standard Landsat QA_PIXEL or Sentinel-2 SCL) to remove obvious thick clouds, and then use Tmask to catch the subtle, missed shadows and thin cirrus clouds that escaped the first pass.
2. **Minimum Observations**: Because Tmask relies on robust harmonic regression (`HuberRegressor`), it requires a minimum number of clear observations across the year to establish a baseline. If a pixel has fewer than 5 valid observations, the algorithm defaults to accepting all remaining pixels as clear.
3. **Computational Cost**: Fitting robust regression models for every pixel is computationally intensive. Consider using parallel processing (like Dask) or chunking when applying Tmask to massive regional datasets.
