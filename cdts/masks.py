import numpy as np

def extract_water_mask(ccdc_coefs_stack, green_band_idx, swir_band_idx):
    """
    Extracts a persistent water mask using the CCDC intercept coefficients.
    Water typically has a higher Green reflectance than SWIR reflectance.
    
    ccdc_coefs_stack: shape (max_segments, params_per_seg, rows, cols)
    green_band_idx: index of the Green band (0-based)
    swir_band_idx: index of the SWIR1 or SWIR2 band (0-based)
    
    Returns a binary mask of shape (rows, cols) where 1 is persistent water.
    """
    segments, params, rows, cols = ccdc_coefs_stack.shape
    
    # Band intercepts are at index: 3 (rmse) + 1 (intercept) = 4 + band_idx * 7
    # But wait, index map:
    # 0: t_start, 1: t_end, 2: t_break
    # For band 0: 3=rmse, 4=c0(intercept), 5=c1(slope), 6=c2, 7=c3, 8=c4, 9=c5
    # For band 1: 10=rmse, 11=c0 ...
    
    green_intercept_idx = 4 + green_band_idx * 7
    swir_intercept_idx = 4 + swir_band_idx * 7
    
    water_mask = np.zeros((rows, cols), dtype=np.uint8)
    
    # We look at the very first segment (initial state of the landscape)
    green_intercept = ccdc_coefs_stack[0, green_intercept_idx, :, :]
    swir_intercept = ccdc_coefs_stack[0, swir_intercept_idx, :, :]
    
    # A simple persistent water heuristic:
    # Water absorbs heavily in SWIR, but reflects slightly in Green.
    # Therefore, Green > SWIR, and SWIR is very low (e.g. < 500 if scaled to 10000)
    
    # Avoid nodata (where intercept == 0)
    valid = (green_intercept != 0) | (swir_intercept != 0)
    
    is_water = valid & (green_intercept > swir_intercept) & (swir_intercept < 500.0)
    water_mask[is_water] = 1
    
    return water_mask

