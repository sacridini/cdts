import numpy as np
from scipy.signal import savgol_filter

def apply_savgol_filter(cube: "np.ndarray", window_length: int = 5, polyorder: int = 2, axis: int = 0) -> "np.ndarray":
    """
    Applies a Savitzky-Golay filter to smooth a time-series cube.
    This removes minor temporal noise and regularizes the trajectory before AI/classification.
    
    Args:
        cube (np.ndarray): The data cube. Expected shape (Time, Bands, H, W) or (Time, H, W).
        window_length (int): The length of the filter window (must be an odd integer).
        polyorder (int): The order of the polynomial used to fit the samples.
        axis (int): The temporal axis.
        
    Returns:
        np.ndarray: Smoothed cube with the same shape.
    """
    if cube.shape[axis] < window_length:
        raise ValueError(f"Time dimension ({cube.shape[axis]}) is smaller than window_length ({window_length}).")
        
    # Apply Savitzky-Golay filter along the time axis
    smoothed = savgol_filter(cube, window_length=window_length, polyorder=polyorder, axis=axis)
    
    # Optional: preserve NoData values (assuming 0 is NoData in original)
    smoothed[cube == 0] = 0
    
    return smoothed
