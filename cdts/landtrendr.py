import numpy as np
from typing import List, Dict, Union
from . import _core

def desawtooth(values: Union[np.ndarray, List[float]], stopat: float = 0.9) -> np.ndarray:
    """
    Remove spikes from a time series using LandTrendr's desawtooth algorithm.
    """
    values_list = values.tolist() if isinstance(values, np.ndarray) else list(values)
    filtered_list = _core.landtrendr.desawtooth(values_list, stopat)
    return np.array(filtered_list)

def run_landtrendr(years: Union[np.ndarray, List[int]], values: Union[np.ndarray, List[float]], max_segments: int = 6, pval_threshold: float = 0.05) -> List[Dict[str, Union[int, float]]]:
    """
    Run LandTrendr algorithm on a 1D time series of a single pixel.
    
    Args:
        years (np.ndarray): 1D array of years.
        values (np.ndarray): 1D array of spectral values.
        max_segments (int): Maximum number of segments to fit.
        pval_threshold (float): P-value threshold for segment significance.
        
    Returns:
        list of dicts containing the fitted vertices (year, value).
    """
    params = _core.landtrendr.LandTrendrParams()
    params.max_segments = max_segments
    params.pval_threshold = pval_threshold
    
    # Ensure lists for C++ vector binding (or we could use pybind11::array in C++ directly for zero-copy)
    years_list = years.tolist() if isinstance(years, np.ndarray) else list(years)
    values_list = values.tolist() if isinstance(values, np.ndarray) else list(values)
    
    vertices = _core.landtrendr.fit_trajectory(years_list, values_list, params)
    
    return [{"year": v.year, "value": v.value} for v in vertices]

def run_landtrendr_batch(years: np.ndarray, values: np.ndarray, max_segments: int = 6, pval_threshold: float = 0.05, no_data_value: float = -9999.0, n_jobs: int = -1):
    """
    Run LandTrendr algorithm on a batch of pixels.
    
    Args:
        years (np.ndarray): 1D array of years [Time].
        values (np.ndarray): 3D array of spectral values [Y, X, Time].
        max_segments (int): Maximum number of segments to fit.
        pval_threshold (float): P-value threshold for segment significance.
        no_data_value (float): No data value in the array.
        n_jobs (int): Number of threads for OpenMP to use. Default -1 (use all).
        
    Returns:
        tuple of (vertices_array, counts_array)
        vertices_array: [Y*X, max_segments+1, 2] containing (year, value) for each vertex
        counts_array: [Y*X] containing the number of valid vertices found for each pixel
    """
    params = _core.landtrendr.LandTrendrParams()
    params.max_segments = max_segments
    params.pval_threshold = pval_threshold
    
    years = np.ascontiguousarray(years, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.float64)
    
    return _core.landtrendr.fit_trajectory_batch(values, years, params, no_data_value, n_jobs)

def apply_vertices(vertex_years: Union[np.ndarray, List[int]], other_band_years: Union[np.ndarray, List[int]], other_band_values: Union[np.ndarray, List[float]]) -> List[Dict[str, Union[int, float]]]:
    """
    Applies LandTrendr structural vertices (FTV - Fitted to Vertices) to another spectral band.
    This effectively uses the segmentation derived from the primary index to smooth and fit the secondary index.
    """
    import numpy as np
    
    if len(vertex_years) == 0 or len(other_band_years) == 0:
        return []
        
    # Find the corresponding values in the other band for the vertex years
    # If a vertex year is missing in the other band, we interpolate it.
    fitted_values = np.interp(vertex_years, other_band_years, other_band_values)
    
    return [{"year": int(y), "value": float(v)} for y, v in zip(vertex_years, fitted_values)]
