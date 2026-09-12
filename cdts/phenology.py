import numpy as np
import xarray as xr
import dask.array as da
from typing import Optional
from cdts._core.phenology import fit_phenology_batch

def run_phenology_dask(
    arr: da.Array,
    dates: np.ndarray,
    curve_type: int,
    extraction_method: int = 0,
    max_seasons: int = 2,
    whittaker_lambda: float = 10.0,
    apply_whittaker: bool = True,
    apply_hants: bool = False,
    hants_frequencies: int = 3,
    hants_threshold: float = 0.1,
    min_season_length: int = 0,
    min_amplitude: float = 0.0,
    min_pixel_amplitude: float = 0.1,
    return_annual: bool = True,
    base_year: int = 2001,
    n_jobs: int = -1
) -> da.Array:
    """
    Applies fit_phenology_batch across a Dask array.
    Input array shape: (time, y, x).
    Output array shape: (4, max_seasons, y, x).
    The first dimension of output represents: 0: SOS, 1: EOS, 2: LOS, 3: POP.
    """
    def _phenology_block(block):
        if block.size == 0:
            return np.zeros((19, max_seasons, block.shape[1], block.shape[2]), dtype=np.float32)
        
        # block is (time, y, x)
        time_steps, rows, cols = block.shape
        pixels = rows * cols
        
        # reshape to (pixels, time) and ensure C-contiguous
        values_2d = np.ascontiguousarray(block.reshape(time_steps, pixels).T)
        
        # run batch fitting
        out_3d = fit_phenology_batch(
            values_array=values_2d,
            dates_array=dates,
            curve_type=curve_type,
            extraction_method=extraction_method,
            max_seasons=max_seasons,
            whittaker_lambda=whittaker_lambda,
            apply_whittaker=apply_whittaker,
            apply_hants=apply_hants,
            hants_frequencies=hants_frequencies,
            hants_threshold=hants_threshold,
            min_season_length=min_season_length,
            min_amplitude=min_amplitude,
            min_pixel_amplitude=min_pixel_amplitude,
            n_jobs=n_jobs
        )
        
        # out_3d is (19, pixels, max_seasons)
        # We need (19, max_seasons, rows, cols)
        # Transpose to (19, max_seasons, pixels) and then reshape
        out_transposed = out_3d.transpose(0, 2, 1)
        out = out_transposed.reshape(19, max_seasons, rows, cols)
        
        if return_annual:
            out_annual = np.full_like(out, np.nan)

            # `val` encodes a date as "days since `base_year`-01-01, 1-indexed"
            # (i.e. datetime(base_year, 1, 1) + timedelta(days=val - 1)), matching
            # the C++ core's day numbering. Vectorized with numpy datetime64 instead
            # of a pure-Python per-pixel loop, since this runs per Dask block and
            # scales with total pixel count across the whole (potentially global) cube.
            valid = np.isfinite(out) & (out > 0)
            if np.any(valid):
                m_idx, s_idx, r_idx, c_idx = np.nonzero(valid)
                vals = out[valid]

                epoch = np.datetime64(f"{base_year}-01-01", "D")
                # timedelta(days=val - 1) only ever carries a sub-day remainder,
                # so the calendar date depends solely on floor(val - 1) days.
                day_offset = np.floor(vals).astype(np.int64) - 1
                event_dates = epoch + day_offset.astype("timedelta64[D]")

                years = event_dates.astype("datetime64[Y]").astype(np.int64) + 1970
                year_idx = years - base_year

                year_start = event_dates.astype("datetime64[Y]")
                doy = (event_dates - year_start).astype(np.int64) + 1  # 1-based day-of-year

                in_range = (year_idx >= 0) & (year_idx < max_seasons)
                m_idx, r_idx, c_idx = m_idx[in_range], r_idx[in_range], c_idx[in_range]
                year_idx = year_idx[in_range]
                doy = doy[in_range]
                vals = vals[in_range]

                is_los = (m_idx == 17)  # LOS stores the raw duration, not a date
                store_val = np.where(is_los, vals, doy.astype(np.float64))

                # For a given (m, r, c), later seasons must win ties on the same
                # calendar year — matches the original loop's `for s in range(...)` order,
                # which `np.nonzero` preserves (C-order over (m, s, r, c)).
                out_annual[m_idx, year_idx, r_idx, c_idx] = store_val

            out = out_annual

        return out.astype(np.float32)
        
    out = da.map_blocks(
        _phenology_block,
        arr,
        dtype=np.float32,
        drop_axis=[0], # remove time
        new_axis=[0, 1], # add metrics (19) and max_seasons
        chunks=(19, max_seasons, arr.chunks[1], arr.chunks[2])
    )
    return out
