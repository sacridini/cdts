import numpy as np
from typing import Dict

def extract_events(vertices_stack: np.ndarray, event_type: str = "loss", sort_by: str = "greatest", 
                  min_magnitude: float = 0.0, min_duration: int = 1, pre_val_threshold: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Extract change events (e.g., largest disturbance) from a stack of vertices.
    
    Args:
        vertices_stack (np.ndarray): The 3D array output from `process_raster_stack`.
                                     Shape: (max_vertices * 2, rows, cols).
                                     First half of bands are years, second half are values.
        event_type (str): "loss" (value decreases) or "gain" (value increases).
        sort_by (str): "greatest" (magnitude), "newest" (year), "fastest" (rate), "longest" (duration).
        min_magnitude (float): Minimum magnitude to consider.
        min_duration (int): Minimum duration (years) to consider.
        pre_val_threshold (float): Pre-disturbance value threshold.
        
    Returns:
        dict: A dictionary of 2D numpy arrays for 'year', 'magnitude', 'duration', 
              'pre_val', 'post_val', and 'rate'.
    """
    bands, rows, cols = vertices_stack.shape
    max_vertices = bands // 2
    
    # Initialize output 2D arrays
    out_year = np.zeros((rows, cols), dtype=np.uint16)
    out_mag = np.zeros((rows, cols), dtype=np.float32)
    out_dur = np.zeros((rows, cols), dtype=np.uint16)
    out_pre = np.zeros((rows, cols), dtype=np.float32)
    out_post = np.zeros((rows, cols), dtype=np.float32)
    out_rate = np.zeros((rows, cols), dtype=np.float32)
    
    # Process each pixel
    # For a fully optimized version, this should be written in Cython/Numba or C++, 
    # but since vertices_stack is small, vectorized/compiled numpy is possible.
    # Here we'll use a straightforward loop optimized with numba if available, 
    # otherwise pure python.
    
    try:
        from numba import njit, prange
        
        @njit(parallel=True, fastmath=True)
        def _extract(v_stack, max_v, rows, cols, is_loss, sort_id, min_mag, min_dur, pre_thresh):
            y_out = np.zeros((rows, cols), dtype=np.uint16)
            m_out = np.zeros((rows, cols), dtype=np.float32)
            d_out = np.zeros((rows, cols), dtype=np.uint16)
            pre_out = np.zeros((rows, cols), dtype=np.float32)
            post_out = np.zeros((rows, cols), dtype=np.float32)
            r_out = np.zeros((rows, cols), dtype=np.float32)
            
            for r in prange(rows):
                for c in range(cols):
                    best_score = -999999.0
                    
                    # Iterate through segments (pairs of vertices)
                    for i in range(max_v - 1):
                        start_year = v_stack[i, r, c]
                        end_year = v_stack[i+1, r, c]
                        
                        if start_year == 0 or end_year == 0:
                            break # No more valid vertices
                            
                        duration = end_year - start_year
                        if duration <= 0:
                            continue
                            
                        pre_val = v_stack[i + max_v, r, c]
                        post_val = v_stack[i+1 + max_v, r, c]
                        
                        if is_loss:
                            magnitude = pre_val - post_val
                            if pre_thresh > 0 and pre_val < pre_thresh: continue
                        else:
                            magnitude = post_val - pre_val
                            if pre_thresh > 0 and pre_val > pre_thresh: continue
                            
                        if magnitude < min_mag: continue
                        if duration < min_dur: continue
                        
                        rate = magnitude / duration
                        
                        score = 0.0
                        if sort_id == 1: score = magnitude
                        elif sort_id == 2: score = start_year
                        elif sort_id == 3: score = rate
                        elif sort_id == 4: score = duration
                            
                        if score > best_score:
                            best_score = score
                            y_out[r, c] = start_year
                            m_out[r, c] = magnitude
                            d_out[r, c] = duration
                            pre_out[r, c] = pre_val
                            post_out[r, c] = post_val
                            r_out[r, c] = rate
                            
            return y_out, m_out, d_out, pre_out, post_out, r_out
            
        is_loss = (event_type.lower() == "loss")
        sort_map = {"greatest": 1, "newest": 2, "fastest": 3, "longest": 4}
        sort_id = sort_map.get(sort_by.lower(), 1)
        
        out_year, out_mag, out_dur, out_pre, out_post, out_rate = _extract(
            vertices_stack, max_vertices, rows, cols, is_loss, sort_id, 
            min_magnitude, min_duration, pre_val_threshold
        )
        
    except ImportError:
        # Fallback to pure python if numba is not installed
        is_loss = (event_type.lower() == "loss")
        sort_map = {"greatest": 1, "newest": 2, "fastest": 3, "longest": 4}
        sort_id = sort_map.get(sort_by.lower(), 1)
        
        for r in range(rows):
            for c in range(cols):
                best_score = -999999.0
                
                for i in range(max_vertices - 1):
                    start_year = vertices_stack[i, r, c]
                    end_year = vertices_stack[i+1, r, c]
                    
                    if start_year == 0 or end_year == 0:
                        break
                        
                    duration = end_year - start_year
                    if duration <= 0: continue
                        
                    pre_val = vertices_stack[i + max_vertices, r, c]
                    post_val = vertices_stack[i+1 + max_vertices, r, c]
                    
                    if is_loss:
                        magnitude = pre_val - post_val
                        if pre_val_threshold > 0 and pre_val < pre_val_threshold: continue
                    else:
                        magnitude = post_val - pre_val
                        if pre_val_threshold > 0 and pre_val > pre_val_threshold: continue
                        
                    if magnitude < min_magnitude: continue
                    if duration < min_duration: continue
                    
                    rate = magnitude / duration
                    
                    score = 0.0
                    if sort_id == 1: score = magnitude
                    elif sort_id == 2: score = start_year
                    elif sort_id == 3: score = rate
                    elif sort_id == 4: score = duration
                        
                    if score > best_score:
                        best_score = score
                        out_year[r, c] = start_year
                        out_mag[r, c] = magnitude
                        out_dur[r, c] = duration
                        out_pre[r, c] = pre_val
                        out_post[r, c] = post_val
                        out_rate[r, c] = rate
                        
    return {
        "yod": out_year,
        "magnitude": out_mag,
        "duration": out_dur,
        "pre_val": out_pre,
        "post_val": out_post,
        "rate": out_rate
    }
