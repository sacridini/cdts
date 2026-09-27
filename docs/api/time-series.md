# Time-Series Analysis

<p class="lead">Trend tests, phenology, pattern matching, segmentation and clustering.</p>

## Trends

### `run_mann_kendall_dask` { .api }

<!-- sig: zeit.trend.run_mann_kendall_dask -->
```python
zeit.trend.run_mann_kendall_dask(
    arr, method="hamed_rao", alpha=0.05, lag=None, period=1,
    min_valid=4, n_jobs=-1,
)
```

Mann-Kendall test and Theil-Sen slope for each pixel of a `(time, y, x)` Dask array. A C++ port of `pymannkendall`. Tutorial: [Trend Analysis](../tutorials/mann_kendall.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arr` | `dask.array.Array` | required | `(time, y, x)`; NaN for gaps. |
| `method` | `str` | `"hamed_rao"` | `"original"`, `"hamed_rao"`, `"yue_wang"` or `"seasonal"`. |
| `alpha` | `float` | `0.05` | Significance level. |
| `lag` | `int` | `None` | Lags used by the autocorrelation corrections. `None`: all. |
| `period` | `int` | `1` | Observations per cycle, for `"seasonal"` (e.g. `23` for 16-day data). |
| `min_valid` | `int` | `4` | Pixels with fewer valid values are NaN. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** `(9, y, x)`: `trend`, `h`, `p`, `z`, `tau`, `s`, `var_s`, `slope`, `intercept` (`zeit.trend.MK_METRIC_NAMES`). The slope is per array step, or per `period` for `"seasonal"`.

```python
from zeit.trend import run_mann_kendall_dask, MK_METRIC_NAMES

out = run_mann_kendall_dask(annual_dask_array).compute()
slope = out[MK_METRIC_NAMES.index("slope")]
```

### `run_mann_kendall_image` { .api }

<!-- sig: zeit.raster.run_mann_kendall_image -->
```python
zeit.raster.run_mann_kendall_image(
    input_path, output_dir, method="hamed_rao", alpha=0.05, lag=None,
    period=1, min_valid=4, chunk_size=512, n_jobs=-1,
    prefix="mann_kendall",
)
```

The same test on a GeoTIFF (one band per time step), block by block. Writes `<output_dir>/<prefix>.tif` with one band per metric. Also exported as `zeit.run_mann_kendall_image`; CLI: `zeit mann-kendall`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_path` | `str` | required | Input GeoTIFF. |
| `output_dir` | `str` | required | Output folder. |
| `method`, `alpha`, `lag`, `period`, `min_valid` | | `"hamed_rao"`, `0.05`, `None`, `1`, `4` | As in `run_mann_kendall_dask`. |
| `chunk_size` | `int` | `512` | Block size in pixels. |
| `n_jobs` | `int` | `-1` | Threads. |
| `prefix` | `str` | `"mann_kendall"` | Output file name. |

</div>

## Phenology

### `run_phenology_dask` { .api }

<!-- sig: zeit.phenology.run_phenology_dask -->
```python
zeit.phenology.run_phenology_dask(
    arr, dates, curve_type, extraction_method=0, max_seasons=2,
    whittaker_lambda=10.0, apply_whittaker=True, apply_hants=False,
    hants_frequencies=3, hants_threshold=0.1, min_season_length=0,
    min_amplitude=0.0, min_pixel_amplitude=0.1, return_annual=True,
    base_year=2001, n_jobs=-1, weights=None, season_retry=True,
)
```

Smoothing, curve fitting and extraction of 19 phenology metrics (plus fit R² and RMSE) for each pixel of a `(time, y, x)` Dask array. Follows the methodology of R `phenofit`. Tutorial: [Phenology](../tutorials/phenology.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arr` | `dask.array.Array` | required | `(time, y, x)` vegetation index. |
| `dates` | 1-D array | required | Days since 1 January of `base_year` (`doy + (year - base_year) * 365`). |
| `curve_type` | `int` | required | `int(CurveType.BECK)` etc., from `zeit._core.phenology`: `BECK`, `ELMORE`, `GU`, `KLOS`, `ZHANG`, `AG`, `DL`. |
| `extraction_method` | `int` | `0` | Kept at `0`; all metrics are always returned. |
| `max_seasons` | `int` | `2` | Season slots (per year with `return_annual=True`). |
| `whittaker_lambda` | `float` | `10.0` | Whittaker smoothness. |
| `apply_whittaker` | `bool` | `True` | Smooth with Whittaker before fitting. |
| `apply_hants` | `bool` | `False` | Smooth with HANTS (harmonics) instead. |
| `hants_frequencies` | `int` | `3` | HANTS harmonics. |
| `hants_threshold` | `float` | `0.1` | HANTS outlier threshold. |
| `min_season_length` | `int` | `0` | Drop seasons shorter than this many days. |
| `min_amplitude` | `float` | `0.0` | Drop seasons with a smaller amplitude. |
| `min_pixel_amplitude` | `float` | `0.1` | Skip pixels whose whole series varies less than this (water, urban). |
| `return_annual` | `bool` | `True` | Align seasons to calendar years (`year` axis) instead of sequential slots. |
| `base_year` | `int` | `2001` | First year of `dates`. |
| `n_jobs` | `int` | `-1` | Threads. |
| `weights` | `dask.array.Array` | `None` | `(time, y, x)` observation weights in `[0, 1]`, e.g. from `zeit.qc`. |
| `season_retry` | `bool` | `True` | Retry pixels with no season once with a relaxed trough threshold. |

</div>

**Returns** `(21, max_seasons, y, x)`: `TRS2.sos`, `TRS2.eos`, `TRS5.sos`, `TRS5.eos`, `TRS6.sos`, `TRS6.eos`, `DER.sos`, `DER.pos`, `DER.eos`, `UD`, `SD`, `DD`, `RD`, `Greenup`, `Maturity`, `Senescence`, `Dormancy`, `LOS`, `POP`, `R2`, `RMSE`. The accessor form, `DataArray.zeit.run_phenology`, returns the same array labelled with these names.

```python
from zeit._core.phenology import CurveType

pheno = ndvi_16d.zeit.run_phenology(dates=days, curve_type=int(CurveType.BECK),
                                    max_seasons=3, base_year=2019).compute()
sos = pheno.sel(metric="TRS5.sos")
```

## Pattern matching (TWDTW)

Tutorial: [Pattern Matching](../tutorials/twdtw.md). The time weight is `alpha / (1 + exp(-beta * (Δt - gamma)))`, with `Δt` in days.

### `classify_twdtw` { .api }

<!-- sig: zeit.twdtw.classify_twdtw -->
```python
zeit.twdtw.classify_twdtw(
    values_array, dates_array, patterns, alpha=0.1, beta=0.05,
    gamma=50.0, max_time_warp=365, subsequence_matching=False,
    n_jobs=-1,
)
```

Compares every pixel with every class pattern and keeps the closest.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `values_array` | `np.ndarray` | required | `(rows, cols, time)` or `(rows, cols, time, bands)`. Time comes after space. |
| `dates_array` | 1-D array | required | Day number of each observation. |
| `patterns` | `dict` | required | `{class_name: (pattern_values, pattern_days)}`. |
| `alpha` | `float` | `0.1` | Maximum time cost. |
| `beta` | `float` | `0.05` | Steepness of the time cost, per day. |
| `gamma` | `float` | `50.0` | Gap in days at which the time cost is half of `alpha`. |
| `max_time_warp` | `int` | `365` | Largest allowed shift, in days. |
| `subsequence_matching` | `bool` | `False` | Match a short pattern anywhere in a longer series. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** `(classes, distance, names)`: the index of the best class per pixel, its distance, and the class names in index order.

```python
classes, distance, names = zeit.twdtw.classify_twdtw(np.moveaxis(stack, 0, -1), days, patterns)
```

### `run_twdtw` { .api }

<!-- sig: zeit.twdtw.run_twdtw -->
```python
zeit.twdtw.run_twdtw(
    ts_values, ts_dates, pattern_values, pattern_dates, alpha=0.1,
    beta=0.05, gamma=50.0, max_time_warp=365,
    subsequence_matching=False, abort_threshold=inf,
    return_path=False,
)
```

TWDTW distance between one series and one pattern. Also exported as `zeit.run_twdtw`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ts_values` | array | required | `(time,)` or `(time, bands)`. |
| `ts_dates` | 1-D array | required | Day numbers of the series. |
| `pattern_values` | array | required | `(time,)` or `(time, bands)`. |
| `pattern_dates` | 1-D array | required | Day numbers of the pattern. |
| `alpha`, `beta`, `gamma`, `max_time_warp`, `subsequence_matching` | | `0.1`, `0.05`, `50.0`, `365`, `False` | As in `classify_twdtw`. |
| `abort_threshold` | `float` | `inf` | Stop early once the distance exceeds this. |
| `return_path` | `bool` | `False` | Also return the alignment as `(series_index, pattern_index)` pairs. |

</div>

**Returns** the distance, or `(distance, path)`.

### `run_twdtw_batch` { .api }

<!-- sig: zeit.twdtw.run_twdtw_batch -->
```python
zeit.twdtw.run_twdtw_batch(
    values_array, dates_array, pattern_values, pattern_dates,
    alpha=0.1, beta=0.05, gamma=50.0, max_time_warp=365,
    subsequence_matching=False, abort_threshold=inf, n_jobs=-1,
)
```

Distance from every pixel to one pattern. Also exported as `zeit.run_twdtw_batch`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `values_array` | `np.ndarray` | required | `(rows, cols, time)` or `(rows, cols, time, bands)`. |
| `dates_array` | 1-D array | required | Day numbers. |
| `pattern_values`, `pattern_dates` | array | required | The pattern. |
| `alpha`, `beta`, `gamma`, `max_time_warp`, `subsequence_matching`, `abort_threshold` | | `0.1`, `0.05`, `50.0`, `365`, `False`, `inf` | As in `run_twdtw`. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** a `(rows, cols)` distance map.

## Segmentation (SNIC)

Tutorial: [Segmentation](../tutorials/snic.md).

### `run_snic` { .api }

<!-- sig: zeit.segmentation.run_snic -->
```python
zeit.segmentation.run_snic(
    data, spacing=10, compactness=0.5, seeds=None, grid="rectangular",
    padding=None, tile_size=None, random_state=None, n_jobs=-1,
)
```

SNIC superpixels of an image or a whole cube; every leading axis becomes a feature. Given the same seeds, labels match the reference C implementation. Also exported as `zeit.run_snic`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `np.ndarray` | required | `(y, x)`, `(feature, y, x)` or `(time, band, y, x)`. Pixels with any NaN are left unlabelled (`-1`). |
| `spacing` | `float` or pair | `10` | Seed spacing in pixels. |
| `compactness` | `float` | `0.5` | Regularity of the segments, in data units. Higher is more compact. |
| `seeds` | `(n, 2)` array | `None` | Explicit `(row, col)` seeds; overrides the grid. |
| `grid` | `str` | `"rectangular"` | `"rectangular"`, `"diamond"`, `"hexagonal"` or `"random"`. |
| `padding` | `float` or pair | `None` | Seed-free margin. Default `spacing / 2`. |
| `tile_size` | `int` or pair | `None` | Segment tiles of this size independently, in parallel. |
| `random_state` | `int` | `None` | Seed for `grid="random"`. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** a `SnicResult` with `labels` `(y, x)`, `means` `(n_seeds, *features)`, `centroids`, `sizes` and `seeds`.

### `snic_to_polygons` { .api }

<!-- sig: zeit.segmentation.snic_to_polygons -->
```python
zeit.segmentation.snic_to_polygons(
    result, transform=None, crs=None, include_means=False,
)
```

Converts SNIC labels to a GeoDataFrame, one polygon per segment, like `sits_segment()`. Also exported as `zeit.snic_to_polygons`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `result` | `SnicResult` | required | Output of `run_snic`. |
| `transform` | `Affine` | `None` | Geotransform of the image (identity: pixel coordinates). |
| `crs` | | `None` | Coordinate reference system. |
| `include_means` | `bool` | `False` | Add each segment's mean features as columns `f0, f1, …`. |

</div>

### `snic_grid` { .api }

<!-- sig: zeit.segmentation.snic_grid -->
```python
zeit.segmentation.snic_grid(
    shape, spacing, padding=None, type="rectangular",
    random_state=None,
)
```

The seed grids of the R `snic` package, as `(n, 2)` `(row, col)` positions. Also exported as `zeit.snic_grid`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shape` | `(int, int)` | required | Image `(rows, cols)`. |
| `spacing` | `float` or pair | required | Seed spacing. |
| `padding` | `float` or pair | `None` | Seed-free margin. |
| `type` | `str` | `"rectangular"` | Grid type, as in `run_snic`'s `grid`. |
| `random_state` | `int` | `None` | Seed for `"random"`. |

</div>

## Clustering (SOM)

### `SOM` { .api .cls }

<!-- sig: zeit.ai.SOM -->
```python
class zeit.ai.SOM(
    x, y, input_len, sigma=1.0, learning_rate=0.5,
    decay_function="asymptotic_decay",
    neighborhood_function="gaussian", topology="rectangular",
    random_seed=42, sigma_decay_function="asymptotic_decay",
)
```

Self-organizing map in C++ (online and batch, OpenMP). An operation-by-operation port of Python [`minisom`](https://github.com/JustGlowing/minisom) 2.3: with the same `random_seed` and arguments the trained codebook is bit-for-bit identical to `MiniSom.train` (`algorithm="online"`) or `MiniSom.train_batch_offline` (`algorithm="batch"`), 30–190 times faster. Tutorial: [Clustering](../tutorials/som.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `x`, `y` | `int` | required | Grid size. |
| `input_len` | `int` | required | Features per sample. |
| `sigma` | `float` | `1.0` | Initial spread of the neighbourhood function. |
| `learning_rate` | `float` | `0.5` | Initial learning rate. |
| `decay_function` | `str` | `"asymptotic_decay"` | `"asymptotic_decay"`, `"inverse_decay_to_zero"` or `"linear_decay_to_zero"`. |
| `neighborhood_function` | `str` | `"gaussian"` | `"gaussian"`, `"mexican_hat"`, `"bubble"` or `"triangle"`. |
| `topology` | `str` | `"rectangular"` | `"rectangular"` or `"hexagonal"`. |
| `random_seed` | `int` | `42` | Seed of the `numpy.random.RandomState` used for initialisation and sample order (the same draws as `minisom`). |
| `sigma_decay_function` | `str` | `"asymptotic_decay"` | `"asymptotic_decay"`, `"inverse_decay_to_one"` or `"linear_decay_to_one"`. |

</div>

| Method | Description |
| :--- | :--- |
| `random_weights_init(data)` / `pca_weights_init(data)` | Initialise the weights from random samples / from the first two principal components. |
| `train(data, num_iters, n_jobs=-1, algorithm="online", random_order=False, use_epochs=False)` | Train on `(samples, features)`, continuing from the current weights. `"online"`: `num_iters` single-sample updates (epochs with `use_epochs=True`). `"batch"`: `num_iters` passes over the data, parallel with `n_jobs`, same result for any `n_jobs`. |
| `predict(data, n_jobs=-1)` | Flat index `i * y + j` of each sample's best-matching unit. |
| `winner(x)` | Grid coordinates `(i, j)` of one sample's best-matching unit. |
| `quantization(data, n_jobs=-1)` / `quantization_error(data, n_jobs=-1)` | Best-matching codebook vector of each sample / mean distance to it. |
| `get_weights()` | Codebook, shape `(x, y, input_len)`. |
| `filter_noisy_samples(data, labels, n_jobs=-1)` | Boolean mask of samples to **keep**: `False` where a sample's label disagrees with the majority label of its neuron. |

```python
from zeit.ai import SOM

som = SOM(x=10, y=10, input_len=X.shape[1], sigma=1.5)
som.random_weights_init(X)
som.train(X, num_iters=20, algorithm="batch")    # 20 passes, OpenMP-parallel
clusters = som.predict(X)
```
