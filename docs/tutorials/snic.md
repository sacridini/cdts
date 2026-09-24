# SNIC Superpixel Segmentation

## 1. Introduction

**SNIC** (Simple Non-Iterative Clustering; Achanta & Süsstrunk, CVPR 2017) partitions an image into compact, connected regions of similar pixels — *superpixels*, or *segments*. In Earth Observation these segments are used as objects: instead of classifying every pixel, you classify one mean time series per segment, which removes salt-and-pepper noise and cuts the number of samples by orders of magnitude. This is the object-based workflow of `sits_segment(seg_fn = sits_snic())` in R's [sits](https://github.com/e-sensing/sits).

`cdts.run_snic` works on a single image *and* on a whole time series cube: every `(time, band)` pair becomes one feature, so two pixels are close when their **trajectories** are close, not just their values on one date. Two fields with the same mean NDVI but opposite seasons fall in different segments.

cdts implements the algorithm of the paper in C++. Given the same seeds, it **produces the same labels as the authors' reference implementation** ([github.com/achanta/SNIC](https://github.com/achanta/SNIC)), pixel for pixel. `tests/test_snic.py` checks this against outputs of the original C code.

---

## 2. Background: How SNIC Works

1. Seeds are placed on a grid. Each seed starts a cluster.
2. A single priority queue holds candidate `(pixel, cluster, distance)` entries; it starts with every seed at distance 0.
3. The closest entry is popped. If its pixel is still free, it joins that cluster, the cluster's running means are updated, and the pixel's free 4-neighbours are pushed with their distance to the *updated* cluster.
4. Repeat until every pixel is labelled — one pass, no iterations (unlike SLIC's k-means).

The distance combines feature and image space:

$$
d = \lVert \mathbf{c}_i - \mathbf{c}_k \rVert^2 + \left(\frac{M}{S}\right)^2 \lVert \mathbf{p}_i - \mathbf{p}_k \rVert^2,
\qquad S = \sqrt{N / K}
$$

where $\mathbf{c}$ are the features (all bands at all dates), $\mathbf{p}$ the row/column position, $N$ the number of valid pixels, $K$ the number of seeds and $M$ the **compactness**.

!!! tip "Choosing `compactness`"
    The feature term is in data units, so $M$ must follow the scale of your data. Small $M$ follows the data closely (irregular segments); large $M$ approaches a regular grid. `0.5` is the sits default for reflectance-scaled cubes; the reference implementation's `10` suits 0-255 CIELAB images. If bands have very different ranges, standardise them first — every feature counts equally in the distance.

---

## 3. Using SNIC in CDTS

### 3.1 On a NumPy cube

```python
import numpy as np
from cdts import run_snic, snic_to_polygons

# cube: (time, band, y, x), e.g. 23 dates x (NDVI, EVI) from a regularised cube
res = run_snic(cube.astype(np.float32), spacing=10, compactness=0.5, grid="hexagonal")

res.labels     # (y, x) int32 segment ids, -1 = unlabelled (NaN pixels)
res.means      # (n_seeds, time, band) mean trajectory of every segment
res.centroids  # (n_seeds, 2) row/col centre of mass
res.sizes      # (n_seeds,) pixel counts
```

Seeds come from one of two places:

| Argument | Seeds |
| :--- | :--- |
| `spacing=` (+ `grid`, `padding`) | the grids of the R `snic` package used by `sits_snic()` (`snic_grid`): `"rectangular"` (default), `"diamond"`, `"hexagonal"`, `"random"`; default `spacing=10`, `padding=spacing/2` |
| `seeds=` | your own `(n, 2)` `(row, col)` pixel positions (takes precedence) |

### 3.2 Via the xarray accessor

```python
import cdts.xarray_api  # registers .cdts

ds = cube_da.cdts.run_snic(spacing=10, compactness=0.5)   # dims (..., y, x)
ds["labels"]          # (y, x)
ds["means"]           # (segment, time, band) with the cube's coordinates
```

### 3.3 Polygons, as `sits_segment()` returns

```python
from cdts.io import get_georef
geo = get_georef(cube_da)
gdf = snic_to_polygons(res, transform=geo["transform"], crs=geo["crs"], include_means=True)
# columns: supercells, x, y (centroid), n_pixels, f0..fN (flattened means), geometry
```

The segment means can go straight to any classifier (TempCNN, LTAE, TWDTW, random forest…) as one sample per segment.

---

## 4. Large Images: Tiles and Parallelism

SNIC is inherently sequential — each step depends on the previous pop — so a single image is segmented on one core. The C++ core is still 1.5–4× faster than the reference C code on one thread (more so with more features) (pixel-major feature layout for contiguous, vectorised Eigen distance computations; float32 input used without a float64 copy), and OpenMP parallelises the data reordering.

For large scenes, use `tile_size`: the image is split into tiles that are segmented **independently and in parallel** (OpenMP), each seed belonging to the tile that contains it — the same block-wise strategy `sits_segment()` uses. Segments never cross tile edges, and labels remain the global seed index, so the output is identical for any `n_jobs`.

```python
res = run_snic(cube, spacing=10, compactness=0.5, tile_size=512, n_jobs=-1)
```

Memory: the core keeps one pixel-major copy of each tile being processed (`rows × cols × features` values of the input dtype) plus the priority queue.

---

## 5. Fidelity to the Reference Implementation

From the same seeds, the labels match those of `snic.c` (Achanta, EPFL). That includes how ties are broken: when two clusters reach a pixel at exactly the same cost, the heap's order decides the winner the same way. The test suite gives cdts the seeds the original placed and compares labels on RGB-like, noisy, piecewise-constant (many ties) and time-series inputs, in float32 and float64. The original's code is not included in cdts.

Differences from the original:

- **Seed grid**: seeds come from the sits grids (`spacing`) or your own list, not from the original's `FindSeeds`. To compare with the original, pass its seeds explicitly, as the test suite does.
- **NaN handling** (as in the R `snic` package used by sits): pixels with a NaN in any feature are left unlabelled (`-1`), and $N$ counts valid pixels only. A seed on a NaN pixel gives an empty segment. Gap-fill the cube first (`regularize_time_series`, `apply_whittaker_filter`) if you want every pixel labelled.
- **Heap bug fix**: the reference `pop()` never removes the heap's last node. With a single seed it reads an uninitialised pixel index and crashes; on tiny images it can return a label that does not exist. Here the heap drains normally.
- **No RGB→CIELAB conversion**: the reference's optional `doRGBtoLAB` is specific to 8-bit RGB photos; convert beforehand if you need it.

---

## 6. References

- Achanta, R., & Süsstrunk, S. (2017). *Superpixels and Polygons Using Simple Non-Iterative Clustering*. CVPR 2017, 4651–4660.
- Simoes, R., et al. *snic: Superpixel Segmentation with the Simple Non-Iterative Clustering Algorithm* (R package), used by `sits_snic()`.
- Simoes, R., Camara, G., et al. (2021). *Satellite Image Time Series Analysis for Big Earth Observation Data*. Remote Sensing, 13(13), 2428.
