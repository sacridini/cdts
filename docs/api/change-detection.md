# Change Detection

<p class="lead">LandTrendr, CCDC and the BFAST family, at every level: one pixel, an in-memory array, a GeoTIFF on disk, or a Dask array. The xarray accessor forms are listed in <a href="../xarray/">Xarray Accessor</a>.</p>

## LandTrendr

Tutorial: [LandTrendr](../tutorials/landtrendr.md).

### `run_landtrendr_array` { .api }

<!-- sig: cdts.raster.run_landtrendr_array -->
```python
cdts.raster.run_landtrendr_array(
    years, raster_stack, max_segments=6, pval_threshold=0.05,
    n_jobs=-1, recovery_threshold=0.25, prevent_fast_recovery=True,
    spike_threshold=0.9, best_model_proportion=1.25,
    vertex_count_overshoot=3, min_observations_needed=6,
    no_data_value=0.0, return_rmse=False, modifier=1.0,
)
```

Segments every pixel of an in-memory stack in parallel (C++ / OpenMP). Also exported as `cdts.run_landtrendr_array`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `years` | 1-D array | required | Year of each layer. |
| `raster_stack` | `np.ndarray` | required | `(time, rows, cols)` index values. |
| `max_segments` | `int` | `6` | Maximum segments per pixel. |
| `pval_threshold` | `float` | `0.05` | Maximum p-value of an accepted model. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |
| `recovery_threshold` | `float` | `0.25` | Rejects recoveries faster than `1 / recovery_threshold` years. |
| `prevent_fast_recovery` | `bool` | `True` | Kept for compatibility; has no effect (the recovery check always applies, as in the original). |
| `spike_threshold` | `float` | `0.9` | Spike dampening (`desawtooth`). `1.0` disables it. |
| `best_model_proportion` | `float` | `1.25` | Prefer the model with more vertices whose p-value is within this factor of the best. |
| `vertex_count_overshoot` | `int` | `3` | Extra candidate vertices before pruning. |
| `min_observations_needed` | `int` | `6` | Pixels with fewer valid years are not segmented. |
| `no_data_value` | `float` | `0.0` | Value treated as missing (NaN always is). |
| `return_rmse` | `bool` | `False` | Also return each pixel's fit RMSE (for `dsnr` in `extract_events`). |
| `modifier` | `float` | `1.0` | `-1.0` when disturbance makes the index **fall** (NDVI, NBR); `1.0` when it makes it rise. |

</div>

**Returns** `vertices`, a `(2 × (max_segments + 1), rows, cols)` float32 array: vertex years in the first half, fitted values in the second (`0` for unused slots). With `return_rmse=True`, returns `(vertices, rmse)`.

```python
vertices, rmse = cdts.run_landtrendr_array(years, stack, modifier=-1.0, return_rmse=True)
```

### `extract_events` { .api }

<!-- sig: cdts.metrics.extract_events -->
```python
cdts.metrics.extract_events(
    vertices_stack, event_type="loss", sort_by="greatest",
    min_magnitude=0.0, min_duration=1, pre_val_threshold=0.0,
    rmse_map=None,
)
```

Turns LandTrendr vertices into maps of one event per pixel. Also exported as `cdts.extract_events`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vertices_stack` | `np.ndarray` | required | Output of `run_landtrendr_array` (or the computed accessor output). |
| `event_type` | `str` | `"loss"` | `"loss"` (index fell) or `"gain"` (index rose). |
| `sort_by` | `str` | `"greatest"` | Which event to keep: `"greatest"`, `"newest"`, `"fastest"`, `"longest"` or `"dsnr"` (needs `rmse_map`). |
| `min_magnitude` | `float` | `0.0` | Ignore events smaller than this, in data units. |
| `min_duration` | `int` | `1` | Ignore events shorter than this many years. |
| `pre_val_threshold` | `float` | `0.0` | For losses, ignore events starting below this value (for gains, above). `0` disables. |
| `rmse_map` | `np.ndarray` | `None` | `(rows, cols)` RMSE from `run_landtrendr_array(..., return_rmse=True)`. Adds the `dsnr` output. |

</div>

**Returns** a dict of `(rows, cols)` arrays: `yod` (year of the vertex where the event starts, i.e. the last year before it; `0` = none), `magnitude`, `duration`, `pre_val`, `post_val`, `rate`, and `dsnr` when `rmse_map` is given.

```python
loss = cdts.extract_events(vertices, event_type="loss", min_magnitude=2000, rmse_map=rmse)
first_year_of_loss = np.where(loss["yod"] > 0, loss["yod"] + 1, 0)
```

### `run_landtrendr_image` { .api }

<!-- sig: cdts.raster.run_landtrendr_image -->
```python
cdts.raster.run_landtrendr_image(
    input_path, output_dir, start_year=2000, max_segments=6,
    chunk_size=512, n_jobs=-1, save_vertices=False, event_type="loss",
    sort_by="greatest", min_mag=0.0, min_dur=1, pre_val_thresh=0.0,
    prefix="lt_event", pval_threshold=0.05, output_scale_factor=1.0,
    recovery_threshold=0.25, prevent_fast_recovery=True,
    spike_threshold=0.9, best_model_proportion=1.25,
    vertex_count_overshoot=3, min_observations_needed=6,
    no_data_value=0.0, modifier=None,
)
```

Runs LandTrendr and `extract_events` on a GeoTIFF (one band per year) block by block, so the file can be larger than memory. Writes `<prefix>_yod.tif`, `_magnitude`, `_duration`, `_pre_val`, `_post_val`, `_rate`, `_dsnr` and, optionally, `lt_vertices.tif`. Also exported as `cdts.run_landtrendr_image`; CLI: `cdts landtrendr`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_path` | `str` | required | Input GeoTIFF, one band per year. |
| `output_dir` | `str` | required | Output folder. |
| `start_year` | `int` | `2000` | Year of the first band. |
| `max_segments` | `int` | `6` | As in `run_landtrendr_array`. |
| `chunk_size` | `int` | `512` | Block size in pixels. |
| `n_jobs` | `int` | `-1` | Threads. |
| `save_vertices` | `bool` | `False` | Also write the vertex stack. |
| `event_type`, `sort_by` | `str` | `"loss"`, `"greatest"` | As in `extract_events`. |
| `min_mag`, `min_dur`, `pre_val_thresh` | | `0.0`, `1`, `0.0` | `extract_events`' `min_magnitude`, `min_duration`, `pre_val_threshold`. |
| `prefix` | `str` | `"lt_event"` | Output file prefix. |
| `pval_threshold` | `float` | `0.05` | As in `run_landtrendr_array`. |
| `output_scale_factor` | `float` | `1.0` | Multiplies value outputs, e.g. `0.0001` to write NDVI instead of NDVI × 10000. |
| `recovery_threshold`, `prevent_fast_recovery`, `spike_threshold`, `best_model_proportion`, `vertex_count_overshoot`, `min_observations_needed`, `no_data_value` | | `0.25`, `True`, `0.9`, `1.25`, `3`, `6`, `0.0` | As in `run_landtrendr_array`. |
| `modifier` | `float` | `None` | Defaults to `-1.0` for `event_type="loss"` and `1.0` for `"gain"`. |

</div>

```python
cdts.run_landtrendr_image("ndvi_1985_2024.tif", "results/", start_year=1985, min_mag=2000)
```

### `run_landtrendr` { .api }

<!-- sig: cdts.landtrendr.run_landtrendr -->
```python
cdts.landtrendr.run_landtrendr(
    years, values, max_segments=6, pval_threshold=0.05,
    recovery_threshold=0.25, prevent_fast_recovery=True,
    spike_threshold=0.9, best_model_proportion=1.25,
    vertex_count_overshoot=3, min_observations_needed=6, modifier=1.0,
)
```

LandTrendr for a single series. Useful for exploring parameters and plotting. Also exported as `cdts.run_landtrendr`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `years` | 1-D array | required | Years. |
| `values` | 1-D array | required | Index values. `NaN` marks a missing year. |
| `max_segments`, `pval_threshold`, `recovery_threshold`, `prevent_fast_recovery`, `spike_threshold`, `best_model_proportion`, `vertex_count_overshoot`, `min_observations_needed`, `modifier` | | `6`, `0.05`, `0.25`, `True`, `0.9`, `1.25`, `3`, `6`, `1.0` | As in `run_landtrendr_array`. |

</div>

**Returns** a list of vertices, `[{"year": 1985, "value": 7972.0}, ...]`.

```python
from cdts.landtrendr import run_landtrendr

run_landtrendr(years, values, modifier=-1.0)
```

### `apply_vertices` { .api }

<!-- sig: cdts.landtrendr.apply_vertices -->
```python
cdts.landtrendr.apply_vertices(
    vertex_years, other_band_years, other_band_values,
)
```

"Fit to vertices": describes another band with the vertex years found on the primary index, by interpolating that band at those years. Also exported as `cdts.apply_vertices`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vertex_years` | 1-D array | required | Vertex years from the primary index. |
| `other_band_years` | 1-D array | required | Years of the other band's series. |
| `other_band_values` | 1-D array | required | The other band's values. |

</div>

```python
vertices = run_landtrendr(years, nbr, modifier=-1.0)
ftv = cdts.apply_vertices([v["year"] for v in vertices], years, swir1)
```

## CCDC

Tutorial: [CCDC](../tutorials/ccdc.md). All CCDC functions expect **surface reflectance × 10,000**, **Python ordinal days**, and **Fmask QA codes** (`0` clear, `1` water, `2` shadow, `3` snow, `4` cloud, `255` fill).

### `run_ccdc` { .api }

<!-- sig: cdts.ccdc.run_ccdc -->
```python
cdts.ccdc.run_ccdc(
    dates, values, qa, min_obs=12, conseq_anom=6,
    chi2_prob_threshold=0.99, tmax_cg_prob_threshold=0.999999,
    detection_bands=None, num_c=8, tmask_bands=None,
    thermal_band=None, valid_range=(0.0, 10000.0),
    thermal_range=(-9320.0, 7070.0),
)
```

CCDC for a single pixel.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dates` | 1-D array | required | Ordinal days. |
| `values` | array | required | `(bands, dates)`: Blue, Green, Red, NIR, SWIR1, SWIR2 [, thermal]. |
| `qa` | 1-D array | required | Fmask codes per date. |
| `min_obs` | `int` | `12` | Kept for compatibility; the original fixes it at 12. |
| `conseq_anom` | `int` | `6` | Consecutive anomalies that confirm a break. |
| `chi2_prob_threshold` | `float` | `0.99` | Change probability for the chi-squared threshold. |
| `tmax_cg_prob_threshold` | `float` | `0.999999` | Outlier probability. |
| `detection_bands` | `list[int]` | `None` | Bands used for detection. Default: Green to SWIR2 (`[1, 2, 3, 4, 5]`). |
| `num_c` | `int` | `8` | Maximum coefficients (4, 6 or 8). |
| `tmask_bands` | `list[int]` | `None` | Bands for the internal Tmask screen. Default `[1, 4]`. |
| `thermal_band` | `int` | `None` | Index of a brightness-temperature band (°C × 100). |
| `valid_range` | `tuple` | `(0.0, 10000.0)` | Valid range of the optical bands. |
| `thermal_range` | `tuple` | `(-9320.0, 7070.0)` | Valid range of the thermal band. |

</div>

**Returns** a list of models, one dict each: `t_start`, `t_end`, `t_break` (`0` if none), `coefs` (bands × 8), `rmse`, `magnitude`, `change_prob`, `category`, `num_obs`.

### `run_ccdc_array` { .api }

<!-- sig: cdts.raster.run_ccdc_array -->
```python
cdts.raster.run_ccdc_array(
    dates, raster_stack, qa_stack, max_segments=6, n_jobs=-1,
    return_coefs=True, conseq_anom=6, **ccdc_kwargs,
)
```

CCDC for every pixel of an in-memory stack, in parallel. Also exported as `cdts.run_ccdc_array`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dates` | 1-D array | required | Ordinal days. |
| `raster_stack` | `np.ndarray` | required | `(bands, time, rows, cols)` reflectance × 10,000. |
| `qa_stack` | `np.ndarray` | required | `(time, rows, cols)` Fmask codes. |
| `max_segments` | `int` | `6` | Model slots in the output. |
| `n_jobs` | `int` | `-1` | Threads. |
| `return_coefs` | `bool` | `True` | Return full models; `False` returns only break dates. |
| `conseq_anom` | `int` | `6` | As in `run_ccdc`. |
| `chi2_prob_threshold`, `tmax_cg_prob_threshold`, `detection_bands`, `num_c`, `tmask_bands`, `thermal_band`, `valid_range`, `thermal_range` | | | Passed through (`**ccdc_kwargs`), as in `run_ccdc`. |

</div>

**Returns** `(max_segments, 3 + 9 × bands, rows, cols)`: per model `t_start`, `t_end`, `t_break`, then for each band its RMSE and 8 coefficients.

### `run_ccdc_image` { .api }

<!-- sig: cdts.raster.run_ccdc_image -->
```python
cdts.raster.run_ccdc_image(
    input_path, output_dir, dates, num_bands=6, qa_band_idx=-1,
    max_segments=6, chunk_size=512, n_jobs=-1, prefix="ccdc_break",
    return_coefs=True, conseq_anom=6,
)
```

CCDC on a date-interleaved GeoTIFF (all bands of date 1, then of date 2, …), block by block. Writes `<prefix>_coefs.tif` with `max_segments × (3 + 9 × num_bands)` bands. Also exported as `cdts.run_ccdc_image`; CLI: `cdts ccdc`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_path` | `str` | required | Date-interleaved GeoTIFF. |
| `output_dir` | `str` | required | Output folder. |
| `dates` | 1-D array | required | Ordinal days, one per date in the file. |
| `num_bands` | `int` | `6` | Bands per date in the file (including the QA band, if any). |
| `qa_band_idx` | `int` | `-1` | Position of the QA band (Fmask codes) within each date. `-1`: no QA band, everything clear. |
| `max_segments` | `int` | `6` | Model slots. |
| `chunk_size` | `int` | `512` | Block size in pixels. |
| `n_jobs` | `int` | `-1` | Threads. |
| `prefix` | `str` | `"ccdc_break"` | Output file prefix. |
| `return_coefs` | `bool` | `True` | As in `run_ccdc_array`. |
| `conseq_anom` | `int` | `6` | As in `run_ccdc`. |

</div>

### `predict_synthetic_image` { .api }

<!-- sig: cdts.ccdc.predict_synthetic_image -->
```python
cdts.ccdc.predict_synthetic_image(
    ccdc_coefs_stack, target_julian_day, num_bands=6,
)
```

Evaluates the CCDC model active on a given date for every pixel, giving a cloud-free image for any day. Also exported as `cdts.predict_synthetic_image`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ccdc_coefs_stack` | `np.ndarray` | required | Output of `run_ccdc_array` (or the `_coefs.tif` reshaped to `(segments, params, rows, cols)`). |
| `target_julian_day` | `int` | required | Ordinal day to predict. |
| `num_bands` | `int` | `6` | Bands to predict. |

</div>

**Returns** `(num_bands, rows, cols)` float32.

```python
from datetime import date
img = cdts.predict_synthetic_image(results, date(2019, 7, 15).toordinal())
```

### `predict` { .api }

<!-- sig: cdts.ccdc.predict -->
```python
cdts.ccdc.predict(coefs, dates)
```

Evaluates one band's 8 coefficients (from `run_ccdc`'s `coefs`) at any ordinal day or days.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `coefs` | array | required | 8 coefficients of one band. |
| `dates` | `int` or array | required | Ordinal day(s). |

</div>

## BFAST family

All three take a regular series: observation `i` is at `start_time + i / frequency`. Tutorials: [BFAST](../tutorials/bfast.md), [BFAST Monitor](../tutorials/bfast_monitor.md), [BFAST Lite](../tutorials/bfast_lite.md).

### `run_bfast_monitor_dask` { .api }

<!-- sig: cdts.bfast.run_bfast_monitor_dask -->
```python
cdts.bfast.run_bfast_monitor_dask(
    arr, start_time, monitor_start_time, frequency, order=3, h=0.25,
    period=10, alpha=0.05, min_valid=10, n_jobs=-1,
)
```

Near-real-time monitoring (`bfastmonitor`, OLS-MOSUM, `history="all"`) for each pixel of a `(time, y, x)` Dask array.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arr` | `dask.array.Array` | required | `(time, y, x)`; NaN for gaps. |
| `start_time` | `float` | required | Time of the first observation (a whole year, e.g. `2010.0`). |
| `monitor_start_time` | `float` | required | Start of the monitoring period. |
| `frequency` | `int` | required | Observations per year. |
| `order` | `int` | `3` | Seasonal harmonics. |
| `h` | `float` | `0.25` | MOSUM window: `0.25`, `0.5` or `1.0`. |
| `period` | `int` | `10` | Boundary horizon: `2`, `4`, `6`, `8` or `10`. |
| `alpha` | `float` | `0.05` | Significance level. |
| `min_valid` | `int` | `10` | Minimum valid history observations. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** `(7, y, x)`: `breakpoint`, `breakpoint_idx`, `magnitude`, `sigma`, `n_history`, `has_break`, `valid` (`cdts.bfast.BFM_METRIC_NAMES`).

### `run_bfast_lite_dask` { .api }

<!-- sig: cdts.bfast.run_bfast_lite_dask -->
```python
cdts.bfast.run_bfast_lite_dask(
    arr, start_time, frequency, order=3, h=0.15, max_breaks_output=5,
    min_valid=20, n_jobs=-1,
)
```

Optimal multiple breakpoints (`bfastlite`, LWZ criterion) for each pixel.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arr` | `dask.array.Array` | required | `(time, y, x)`; NaN for gaps. |
| `start_time`, `frequency` | | required | Regular time axis. |
| `order` | `int` | `3` | Seasonal harmonics. |
| `h` | `float` | `0.15` | Minimum segment size, fraction of the observations. |
| `max_breaks_output` | `int` | `5` | Break slots in the output. |
| `min_valid` | `int` | `20` | Minimum valid observations. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** `(5 + max_breaks_output, y, x)`: `n_breaks`, `rss`, `lwz`, `n_valid`, `valid`, `breakpoint_idx_1…` (`cdts.bfast.bfl_metric_names(max_breaks_output)`).

### `run_bfast_dask` { .api }

<!-- sig: cdts.bfast.run_bfast_dask -->
```python
cdts.bfast.run_bfast_dask(
    arr, start_time, frequency, order=3, h=0.15, max_breaks_trend=5,
    max_breaks_season=5, max_iter=10, level=0.05, min_valid=20,
    n_jobs=-1,
)
```

Classic iterative BFAST (trend and seasonal breaks) for each pixel.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arr` | `dask.array.Array` | required | `(time, y, x)`; NaN for gaps. |
| `start_time`, `frequency` | | required | Regular time axis. |
| `order` | `int` | `3` | Seasonal harmonics. |
| `h` | `float` | `0.15` | Minimum segment size. |
| `max_breaks_trend`, `max_breaks_season` | `int` | `5`, `5` | Break slots in the output. |
| `max_iter` | `int` | `10` | Maximum trend/season iterations. |
| `level` | `float` | `0.05` | Significance of the stability pre-test (`1.0` always searches). |
| `min_valid` | `int` | `20` | Minimum valid observations. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** `n_trend_breaks`, `n_season_breaks`, `magnitude`, `time`, `n_iter`, `n_valid`, `valid`, then the trend and season break indices (`cdts.bfast.bf_metric_names(max_breaks_trend, max_breaks_season)`).

### GeoTIFF versions

`run_bfast_monitor_image`, `run_bfast_lite_image` and `run_bfast_image` take the same parameters as their Dask counterparts, with `input_path`, `output_dir`, `chunk_size` and `prefix` instead of `arr`. The input has one band per time step; the output is `<output_dir>/<prefix>.tif` with one band per metric (band descriptions set to the metric names). All three are exported at the top level and available in the [CLI](../cli.md).

```python
cdts.run_bfast_monitor_image("ndvi_16d.tif", "results/", start_time=2010.0,
                             monitor_start_time=2022.0, frequency=23)
```
