import numpy as np
from . import _core

def run_ccdc(dates, values, qa, min_obs=12, conseq_anom=3, chi2_prob_threshold=0.99):
    """
    Continuous Change Detection and Classification (CCDC).
    
    This function wraps the C++ implementation.
    """
    dates_list = dates.tolist() if isinstance(dates, np.ndarray) else list(dates)
    
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

def predict_synthetic_image(ccdc_coefs_stack, target_julian_day, num_bands=6):
    """
    Generates a cloud-free synthetic image for a specific day using CCDC harmonic coefficients.
    ccdc_coefs_stack: 4D numpy array output from run_ccdc_array() or read from _coefs.tif.
    """
    import numpy as np
    
    _, bands_dim, rows, cols = ccdc_coefs_stack.shape
    
    W = 2.0 * np.pi / 365.25
    t = target_julian_day
    
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
