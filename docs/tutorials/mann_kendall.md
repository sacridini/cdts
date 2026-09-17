# Mann-Kendall Trend Test

## 1. Introduction

The Mann-Kendall (MK) test is a non-parametric statistical test used to detect monotonic trends in a time series without assuming a particular data distribution or a linear relationship. Combined with the **Theil-Sen slope estimator** (a robust, outlier-resistant estimate of the trend's magnitude), it is the standard tool in remote sensing for questions like "has NDVI been declining over this region for the last 20 years?" or "is there a statistically significant greening/browning trend at this pixel?".

`cdts` provides a pixel-wise, C++/OpenMP-accelerated implementation of the Mann-Kendall test family, ported from the reference Python implementation [`pymannkendall`](https://github.com/mmhs013/pymannkendall) (Hussain & Mahmud, 2019 — see [References](#7-references)), with the same Dask distribution strategy used by [Phenology Extraction](phenology.md) and the other batch algorithms: OpenMP parallelizes across pixels within a chunk, Dask distributes chunks across cores/workers.

### Why not just fit a linear regression?

Ordinary least-squares regression assumes normally distributed, independent, homoscedastic residuals — assumptions satellite time series routinely violate (skewed index distributions, autocorrelated annual composites, occasional extreme outliers from residual cloud contamination). Mann-Kendall only asks whether values tend to *rank* higher later in the series, and Sen's slope takes the *median* of all pairwise slopes rather than a least-squares fit, making both far more robust to outliers and non-normality.

---

## 2. Background: Which Method Should I Use?

`cdts` exposes four variants of the test, selected via the `method` argument:

| Method | When to use it |
| :--- | :--- |
| `"original"` | Classic Mann-Kendall (Mann 1945, Kendall 1975). Assumes the observations are serially independent. |
| `"hamed_rao"` | **Recommended default.** Hamed & Rao (1998) correct the test's variance for serial autocorrelation. Annual satellite composites are usually autocorrelated (a wet year tends to be followed by another wet-ish year), which inflates the false-positive rate of the original test — this correction accounts for that. |
| `"yue_wang"` | Yue & Wang (2004), an alternative autocorrelation correction (pre-whitening-free, like Hamed-Rao). Occasionally more powerful than Hamed-Rao depending on the autocorrelation structure; worth comparing if results are borderline. |
| `"seasonal"` | Hirsch & Slack (1984). Reshapes the series into `period` season slots (e.g. `period=23` for MODIS 16-day annual cycles, `period=12` for monthly data) and pools the per-season MK scores. Lets you test a **raw sub-annual series directly**, without pre-aggregating to one observation per year first. |

All four return the same 9 metrics: `trend`, `h`, `p`, `z`, `tau`, `s`, `var_s`, `slope`, `intercept` — see [Understanding the Output](#4-understanding-the-output).

!!! warning "Units of `slope`/`intercept`"
    For `"original"`/`"hamed_rao"`/`"yue_wang"`, the slope is in units of your data **per time step** (i.e. per array index), *not* per calendar year — pass one observation per year (e.g. an annual max-NDVI composite) if you want a directly interpretable per-year trend. For `"seasonal"`, the slope is already per full `period` cycle (e.g. per year, if `period` spans one year), so a raw 16-day or monthly series gives a per-year slope without needing pre-aggregation. This is the same kind of unit caveat as `min_season_length` in [Phenology Extraction](phenology.md) (see its "Advanced Configuration" section) — always check what "one time step" means for your input array.

---

## 3. Using Mann-Kendall in CDTS

### Step 3.1: Via the xarray accessor (recommended)

The `.cdts.run_mann_kendall()` accessor runs the test pixel-wise across a Dask-backed `(time, y, x)` DataArray and returns a `(metric, y, x)` result.

```python
import numpy as np
import xarray as xr
import dask.array as da

# Synthetic annual max-NDVI composite cube: 20 years, 50x50 pixels.
# Half the pixels show a clear upward trend; the rest are flat/noisy.
rng = np.random.RandomState(0)
n_years, rows, cols = 20, 50, 50

ndvi = rng.normal(0.5, 0.03, (n_years, rows, cols))
ndvi[:, :25, :] += (np.arange(n_years) * 0.01)[:, None, None]  # top half trends up

cube = xr.DataArray(
    da.from_array(ndvi, chunks=(n_years, 25, 25)),
    dims=["time", "y", "x"],
    coords={"y": np.arange(rows), "x": np.arange(cols)},
)

trend = cube.cdts.run_mann_kendall(method="hamed_rao", alpha=0.05)

print(trend.coords["metric"].values)
# ['trend' 'h' 'p' 'z' 'tau' 's' 'var_s' 'slope' 'intercept']

result = trend.compute()

slope_map = result.sel(metric="slope")          # NDVI change per year
significant = result.sel(metric="h") == 1.0     # statistically significant at alpha=0.05
```

### Step 3.2: Via the Dask array wrapper

If you already have a raw `dask.array.Array` (rather than an `xarray.DataArray`), `cdts.trend.run_mann_kendall_dask` skips the xarray wrapping:

```python
from cdts.trend import run_mann_kendall_dask

out = run_mann_kendall_dask(cube.data, method="hamed_rao")  # dask.array, shape (9, y, x)
```

### Step 3.3: Single-series testing/debugging

For inspecting a single pixel's time series directly (no Dask/array machinery), use the low-level C++ binding:

```python
from cdts._core.mannkendall import mk_test_single, MKMethod

y = [0.41, 0.44, 0.39, 0.47, 0.52, 0.49, 0.55, 0.58, 0.61, 0.60]

trend, h, p, z, tau, s, var_s, slope, intercept = mk_test_single(
    y, method=int(MKMethod.HAMED_RAO), alpha=0.05
)
print(f"trend={trend} (h={h}), p={p:.4f}, slope={slope:.4f}/step")
```

### Step 3.4: Sub-annual data with `method="seasonal"`

If you'd rather test a raw 16-day or monthly time series without pre-aggregating to annual composites first, use `method="seasonal"` with `period` set to the number of steps per cycle:

```python
# 15 years of MODIS 16-day composites (23 steps/year), tested directly
trend_seasonal = ndvi_16d.cdts.run_mann_kendall(method="seasonal", period=23)
```

---

## 4. Understanding the Output

Every call returns a DataArray/array with 9 rows along the `metric` dimension:

| Metric | Meaning |
| :--- | :---: |
| `trend` | `-1` decreasing, `0` no significant trend, `1` increasing. |
| `h` | `1.0` if the trend is statistically significant at `alpha`, else `0.0`. |
| `p` | Two-tailed p-value of the test. |
| `z` | Standardized test statistic. |
| `tau` | Kendall's Tau (rank correlation, `-1` to `1`). |
| `s` | Mann-Kendall's raw S score (sum of pairwise concordant minus discordant signs). |
| `var_s` | Variance of `s` (autocorrelation-corrected, for `hamed_rao`/`yue_wang`). |
| `slope` | Theil-Sen slope — see the units warning in [Section 2](#2-background-which-method-should-i-use). |
| `intercept` | Intercept of the Kendall-Theil robust line (Conover, 1980). |

Pixels with fewer than `min_valid` non-NaN observations (default `4`) are returned as all-NaN — this mirrors `min_pixel_amplitude` in Phenology Extraction as a cheap early skip for water/urban/persistently-masked pixels.

---

## 5. Practical Example: Regional Vegetation Trend Map

Combining `cdts`'s STAC ingestion with `run_mann_kendall` gives an end-to-end degradation/greening trend map with a handful of lines:

```python
import cdts
import numpy as np
from cdts._core.phenology import CurveType

# 1. Build an annual max-NDVI composite series for the region of interest
#    (20 growing seasons)
cube = cdts.build_time_series(
    source="planetary_computer",
    collection="sentinel-2-l2a",
    bbox=[-52.10, -12.55, -51.95, -12.40],
    start_date="2004-01-01",
    end_date="2023-12-31",
    cloud_cover_max=20,
    bands=["red", "nir"],
    resolution=10,
)

ndvi = (cube.sel(band="nir") - cube.sel(band="red")) / (cube.sel(band="nir") + cube.sel(band="red"))
annual_max_ndvi = ndvi.groupby("time.year").max(skipna=True)  # (year, y, x)
annual_max_ndvi = annual_max_ndvi.rename({"year": "time"})

# 2. Run the trend test — one observation per year, so slope is already per-year
trend = annual_max_ndvi.cdts.run_mann_kendall(method="hamed_rao", alpha=0.05)
result = trend.compute()

# 3. Build a "significant decline" mask (e.g. a degradation/deforestation-adjacent signal)
declining = (result.sel(metric="trend") == -1) & (result.sel(metric="h") == 1.0)

cdts.save_raster(
    declining.astype("uint8").values,
    "results/ndvi_decline_2004_2023.tif",
    reference_cube=cube,
)
```

### Interpreting the result

- **`trend == 1` and `h == 1`**: statistically significant greening — could reflect regrowth, agricultural expansion, or irrigation onset.
- **`trend == -1` and `h == 1`**: statistically significant decline — worth cross-checking against disturbance products like [LandTrendr](landtrendr.md) or [CCDC](ccdc.md) to see *when* and *how abruptly* the decline happened, since Mann-Kendall only tells you a monotonic trend exists over the whole window, not when it started.
- **`h == 0`**: no significant trend at the chosen `alpha` — most of a typical landscape, and expected.

---

## 6. Best Practices

1. **Prefer `"hamed_rao"` over `"original"` for annual EO composites.** Serial correlation between consecutive years is the norm, not the exception, and the uncorrected test's false-positive rate rises quickly as autocorrelation increases.
2. **One observation per year for a per-year slope**, unless you use `method="seasonal"` with an explicit `period` — see the units warning in [Section 2](#2-background-which-method-should-i-use).
3. **`min_amplitude`-style pre-filtering isn't built in** — if you want to skip water/urban/bare-soil pixels before running the test (for speed, or to avoid meaningless trends on near-constant series), mask them beforehand (e.g. with a simple amplitude threshold, or reuse a mask from Phenology Extraction's `min_pixel_amplitude`).
4. **Combine with LandTrendr/CCDC for "when", not just "if".** Mann-Kendall answers "is there a monotonic trend over this whole period", while LandTrendr/CCDC answer "when did a change happen and how abrupt was it" — they're complementary, not competing.
5. **Use `"seasonal"` to avoid throwing away sub-annual data.** Pre-aggregating a dense 16-day time series down to one value per year discards information; `method="seasonal"` lets the test use every observation while still producing a per-year-equivalent slope.

---

## 7. References

`cdts`'s implementation is a direct C++/OpenMP port of **pymannkendall**, validated field-by-field against it on hundreds of synthetic series (exact match to floating-point precision):

- Hussain, M. M., & Mahmud, I. (2019). pyMannKendall: a python package for non parametric Mann Kendall family of trend tests. **Journal of Open Source Software**, 4(39), 1556. [https://doi.org/10.21105/joss.01556](https://doi.org/10.21105/joss.01556)

The underlying statistical methods:

- Mann, H. B. (1945). Nonparametric tests against trend. **Econometrica**, 13(3), 245–259. [https://doi.org/10.2307/1907187](https://doi.org/10.2307/1907187)
- Kendall, M. G. (1975). *Rank Correlation Methods* (4th ed.). Griffin, London.
- Theil, H. (1950). A rank-invariant method of linear and polynomial regression analysis. **Indagationes Mathematicae**, 12, 85–91.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. **Journal of the American Statistical Association**, 63(324), 1379–1389. [https://doi.org/10.1080/01621459.1968.10480934](https://doi.org/10.1080/01621459.1968.10480934)
- Hamed, K. H., & Rao, A. R. (1998). A modified Mann-Kendall trend test for autocorrelated data. **Journal of Hydrology**, 204(1–4), 182–196. [https://doi.org/10.1016/S0022-1694(97)00125-X](https://doi.org/10.1016/S0022-1694(97)00125-X)
- Yue, S., & Wang, C. (2004). The Mann-Kendall test modified by effective sample size to detect trend in serially correlated hydrological series. **Water Resources Management**, 18(3), 201–218. [https://doi.org/10.1023/B:WARM.0000043140.61082.60](https://doi.org/10.1023/B:WARM.0000043140.61082.60)
- Hirsch, R. M., & Slack, J. R. (1984). A nonparametric trend test for seasonal data with serial dependence. **Water Resources Research**, 20(6), 727–732. [https://doi.org/10.1029/WR020i006p00727](https://doi.org/10.1029/WR020i006p00727)
