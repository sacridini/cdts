import numpy as np
from typing import List, Dict, Any, Union
from . import _core

def run_ccdc(
    dates: Union[np.ndarray, List[float]], 
    values: Union[np.ndarray, List[List[float]]], 
    qa: Union[np.ndarray, List[int]], 
    min_obs: int = 12, 
    conseq_anom: int = 3, 
    chi2_prob_threshold: float = 0.99
) -> List[Dict[str, Any]]:
    """
    Continuous Change Detection and Classification (CCDC).
    
    This function wraps the C++ implementation.
    """
    dates_list = [int(d) for d in (dates.tolist() if isinstance(dates, np.ndarray) else list(dates))]
    
    params = _core.ccdc.CCDCParams()
    params.min_obs = min_obs
    params.conseq_anom = conseq_anom
    params.chi2_prob_threshold = chi2_prob_threshold
    
    # Ensure values is 2D: (num_bands, num_dates)
    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            values_list = [values.tolist()]
        else:
            values_list = values.tolist()
    else:
        # If it's a list, check if it's 1D or 2D
        if len(values) > 0 and not isinstance(values[0], (list, tuple, np.ndarray)):
            values_list = [values]
        else:
            values_list = values
            
    qa_list = qa.tolist() if isinstance(qa, np.ndarray) else list(qa)
    
    segments = _core.ccdc.fit_ccdc(dates_list, values_list, qa_list, params)
    
    return [
        {
            "t_start": s.t_start,
            "t_end": s.t_end,
            "t_break": s.t_break,
            "coefs": s.coefs,
            "rmse": s.rmse,
            "magnitude": s.magnitude
        } for s in segments
    ]

def predict_synthetic_image(ccdc_coefs_stack: np.ndarray, target_julian_day: int, num_bands: int = 6) -> np.ndarray:
    """
    Generates a cloud-free synthetic image for a specific day using CCDC harmonic coefficients.
    ccdc_coefs_stack: 4D numpy array output from run_ccdc_array() or read from _coefs.tif.
    """
    import numpy as np
    
    _, bands_dim, rows, cols = ccdc_coefs_stack.shape
    
    W = 2.0 * np.pi / 365.25
    t = float(target_julian_day)
    
    terms = np.array([
        1.0, t, np.cos(W * t), np.sin(W * t), np.cos(2.0 * W * t), np.sin(2.0 * W * t)
    ])
    
    synthetic_image = np.zeros((num_bands, rows, cols), dtype=np.float32)
    
    for r in range(rows):
        for c in range(cols):
            best_seg = 0
            for i in range(ccdc_coefs_stack.shape[0]):
                t_start = ccdc_coefs_stack[i, 0, r, c]
                t_end = ccdc_coefs_stack[i, 1, r, c]
                
                if t_start <= t <= t_end:
                    best_seg = i
                    break
            
            idx = 3
            for b in range(num_bands):
                idx += 1 # skip RMSE
                coefs = ccdc_coefs_stack[best_seg, idx:idx+6, r, c]
                idx += 6
                synthetic_image[b, r, c] = np.dot(coefs, terms)
                
    return synthetic_image

def run_ccdc_batch(dates: np.ndarray, values: np.ndarray, qa: np.ndarray, max_segments: int = 6, return_coefs: bool = True, conseq_anom: int = 3, n_jobs: int = -1):
    """
    Run CCDC algorithm on a batch of pixels.
    
    Args:
        dates (np.ndarray): 1D array of ordinal dates [Time].
        values (np.ndarray): 4D array of spectral values [Y, X, Bands, Time].
        qa (np.ndarray): 3D array of QA values [Y, X, Time].
        max_segments (int): Maximum number of segments to fit.
        return_coefs (bool): Whether to return harmonic coefficients.
        conseq_anom (int): Consecutive anomalies to trigger a break.
        n_jobs (int): Number of threads for OpenMP to use. Default -1 (use all).
        
    Returns:
        tuple of (segments_array, counts_array)
    """
    params = _core.ccdc.CCDCParams()
    params.conseq_anom = conseq_anom
    
    dates = np.ascontiguousarray(dates, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.float64)
    qa = np.ascontiguousarray(qa, dtype=np.int32)
    
    return _core.ccdc.fit_ccdc_batch(values, qa, dates, params, max_segments, return_coefs, n_jobs)
