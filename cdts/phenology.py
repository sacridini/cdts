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
            return np.zeros((4, max_seasons, block.shape[1], block.shape[2]), dtype=np.float32)
        
        # block is (time, y, x)
        time_steps, rows, cols = block.shape
        pixels = rows * cols
        
        # reshape to (pixels, time) and ensure C-contiguous
        values_2d = np.ascontiguousarray(block.reshape(time_steps, pixels).T)
        
        # run batch fitting
        sos, eos, los, pop = fit_phenology_batch(
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
        
        # reshape outputs to (max_seasons, y, x)
        sos_3d = sos.T.reshape(max_seasons, rows, cols)
        eos_3d = eos.T.reshape(max_seasons, rows, cols)
        los_3d = los.T.reshape(max_seasons, rows, cols)
        pop_3d = pop.T.reshape(max_seasons, rows, cols)
        
        # stack to (4, max_seasons, y, x)
        out = np.stack([sos_3d, eos_3d, los_3d, pop_3d], axis=0)
        return out.astype(np.float32)
        
    out = da.map_blocks(
        _phenology_block,
        arr,
        dtype=np.float32,
        drop_axis=[0], # remove time
        new_axis=[0, 1], # add metrics (4) and max_seasons
        chunks=(4, max_seasons, arr.chunks[1], arr.chunks[2])
    )
    return out
