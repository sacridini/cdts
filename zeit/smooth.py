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

import scipy.sparse as sp
from scipy.sparse.linalg import splu

def apply_whittaker_filter(cube: "np.ndarray", lmbd: float = 10.0, axis: int = 0, weights: "np.ndarray" = None) -> "np.ndarray":
    """
    Applies a Whittaker smoother along the time axis.
    This is often superior to Savitzky-Golay for NDVI/EVI because it penalizes roughness directly 
    and handles missing/cloudy data elegantly when weights are provided (0 for cloud, 1 for clear).
    
    Args:
        cube: 3D numpy array [Time, Y, X].
        lmbd: Smoothing parameter (larger = smoother curve).
        axis: The temporal axis.
        weights: Optional 3D array of the same shape with weights for each observation.
    """
    if axis != 0:
        cube = np.swapaxes(cube, 0, axis)
        if weights is not None:
            weights = np.swapaxes(weights, 0, axis)
            
    T, Y, X = cube.shape
    smoothed = np.zeros_like(cube)
    
    # Pre-build difference matrix for length T
    E = sp.eye(T, format='csc')
    D = E[2:] - 2 * E[1:-1] + E[:-2]
    D_T_D = D.T @ D
    
    for y in range(Y):
        for x in range(X):
            ts = cube[:, y, x]
            if weights is None:
                w = np.ones(T)
            else:
                w = weights[:, y, x]
                
            W = sp.spdiags(w, 0, T, T)
            A = W + lmbd * D_T_D
            try:
                # Solve A * z = W * ts
                z = splu(A.tocsc()).solve(w * ts)
                smoothed[:, y, x] = z
            except RuntimeError:
                # Fallback if matrix is perfectly singular (rare)
                smoothed[:, y, x] = ts
                
    if axis != 0:
        smoothed = np.swapaxes(smoothed, 0, axis)
        
    return smoothed
