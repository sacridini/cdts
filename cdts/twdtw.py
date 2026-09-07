import numpy as np
from typing import List, Dict, Any, Union, Tuple
from . import _core

def run_twdtw(
    ts_values: Union[np.ndarray, List[float]], 
    ts_dates: Union[np.ndarray, List[int]], 
    pattern_values: Union[np.ndarray, List[float]], 
    pattern_dates: Union[np.ndarray, List[int]], 
    alpha: float = 0.1, 
    beta: float = 0.05, 
    gamma: float = 50.0,
    max_time_warp: int = 365,
    abort_threshold: float = float('inf')
) -> float:
    """
    Run Time-Weighted Dynamic Time Warping (TWDTW) on a single time series against a pattern.
    
    Args:
        ts_values (np.ndarray or list): Time series values.
        ts_dates (np.ndarray or list): Time series dates (ordinal or DOY).
        pattern_values (np.ndarray or list): Pattern time series values.
        pattern_dates (np.ndarray or list): Pattern time series dates.
        alpha (float): Amplitude of the time weight penalty.
        beta (float): Steepness of the logistic time weight function.
        gamma (float): Midpoint (inflection point) of the logistic function.
        max_time_warp (int): Sakoe-Chiba constraint (maximum time distortion allowed).
        abort_threshold (float): Early abandonment limit. Returns inf if cost exceeds this.
        
    Returns:
        float: The TWDTW distance between the time series and the pattern.
    """
    ts_values_list = [float(v) for v in (ts_values.tolist() if isinstance(ts_values, np.ndarray) else list(ts_values))]
    ts_dates_list = [int(d) for d in (ts_dates.tolist() if isinstance(ts_dates, np.ndarray) else list(ts_dates))]
    pat_values_list = [float(v) for v in (pattern_values.tolist() if isinstance(pattern_values, np.ndarray) else list(pattern_values))]
    pat_dates_list = [int(d) for d in (pattern_dates.tolist() if isinstance(pattern_dates, np.ndarray) else list(pattern_dates))]
    
    params = _core.twdtw.TWDTWParams()
    params.alpha = alpha
    params.beta = beta
    params.gamma = gamma
    params.max_time_warp = max_time_warp
    
    distance = _core.twdtw.fit_twdtw(
        ts_values_list, 
        ts_dates_list, 
        pat_values_list, 
        pat_dates_list, 
        params, 
        float(abort_threshold)
    )
    
    return distance


def run_twdtw_batch(
    values_array: np.ndarray, 
    dates_array: np.ndarray, 
    pattern_values: np.ndarray, 
    pattern_dates: np.ndarray, 
    alpha: float = 0.1, 
    beta: float = 0.05, 
    gamma: float = 50.0,
    max_time_warp: int = 365,
    abort_threshold: float = float('inf'),
    n_jobs: int = -1
) -> np.ndarray:
    """
    Run highly optimized TWDTW on a batch of pixels (3D array) using OpenMP.
    
    Args:
        values_array (np.ndarray): 3D array of spectral values [Y, X, Time].
        dates_array (np.ndarray): 1D array of dates [Time].
        pattern_values (np.ndarray): 1D array of pattern spectral values.
        pattern_dates (np.ndarray): 1D array of pattern dates.
        alpha (float): Amplitude of the time weight penalty.
        beta (float): Steepness of the logistic time weight function.
        gamma (float): Midpoint (inflection point) of the logistic function.
        max_time_warp (int): Sakoe-Chiba constraint (maximum time distortion allowed).
        abort_threshold (float): Early abandonment limit.
        n_jobs (int): Number of threads for OpenMP to use. Default -1 (use all).
        
    Returns:
        np.ndarray: 2D array [Y, X] of TWDTW distances.
    """
    params = _core.twdtw.TWDTWParams()
    params.alpha = alpha
    params.beta = beta
    params.gamma = gamma
    params.max_time_warp = max_time_warp
    
    values_array = np.ascontiguousarray(values_array, dtype=np.float64)
    dates_array = np.ascontiguousarray(dates_array, dtype=np.int32)
    pattern_values = np.ascontiguousarray(pattern_values, dtype=np.float64)
    pattern_dates = np.ascontiguousarray(pattern_dates, dtype=np.int32)
    
    result = _core.twdtw.fit_twdtw_batch(
        values_array, 
        dates_array, 
        pattern_values, 
        pattern_dates, 
        params,
        float(abort_threshold),
        n_jobs
    )
    
    return result
