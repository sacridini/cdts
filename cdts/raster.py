import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import rasterio
from rasterio.windows import Window

from .landtrendr import run_landtrendr
from .metrics import extract_events
from typing import Tuple, List, Dict, Any, Union

def _process_pixel_lt(args: Tuple[int, int, np.ndarray], years: Union[np.ndarray, List[int]], max_segments: int, pval_threshold: float) -> Tuple[int, int, List[Dict[str, Union[int, float]]]]:
    """Worker function for parallel processing of a single pixel with LandTrendr."""
    row, col, values = args
    
    # Mask out completely empty pixels (e.g. nodata)
    if np.all(values == 0) or np.all(np.isnan(values)):
        return row, col, []
        
    try:
        vertices = run_landtrendr(years, values, max_segments=max_segments, pval_threshold=pval_threshold)
        return row, col, vertices
    except Exception:
        # If C++ fails on a specific weird pixel, return empty
        return row, col, []

def run_landtrendr_array(years: "np.ndarray", raster_stack: "np.ndarray", max_segments: int = 6, pval_threshold: float = 0.05, n_jobs: int = -1) -> "np.ndarray":
    """
    Apply LandTrendr across a 3D numpy array (time_steps, rows, cols) using multiprocessing.
    """
    time_steps, rows, cols = raster_stack.shape
    max_vertices = max_segments + 1
    
    output = np.zeros((2 * max_vertices, rows, cols), dtype=np.float32)
    
    pixel_args = [
        (r, c, raster_stack[:, r, c]) 
        for r in range(rows) for c in range(cols)
    ]
    
    worker = partial(_process_pixel_lt, 
                     years=years, 
                     max_segments=max_segments, 
                     pval_threshold=pval_threshold)
                     
    if n_jobs == -1:
        import os as _os
        n_jobs = _os.cpu_count() or 4
        
    if n_jobs == 1:
        results = list(map(worker, pixel_args))
    else:
        with Pool(processes=n_jobs) as pool:
            results = pool.map(worker, pixel_args)
        
    for row, col, vertices in results:
        n_verts = len(vertices)
        if n_verts == 0:
            continue
            
        n_verts = min(n_verts, max_vertices)
        
        for i in range(n_verts):
            output[i, row, col] = vertices[i]['year']
            output[i + max_vertices, row, col] = vertices[i]['value']
            
    return output


# ---------------------------------------------------------
# CCDC Raster Engine
# ---------------------------------------------------------
from .ccdc import run_ccdc

def _process_pixel_ccdc(args: Tuple[int, int, np.ndarray, np.ndarray], dates: Union[np.ndarray, List[int]], max_segments: int, conseq_anom: int) -> Tuple[int, int, List[Dict[str, Any]]]:
    row, col, values, qa = args
    
    if np.all(values == 0) or np.all(np.isnan(values)):
        return row, col, []
        
    try:
        segments = run_ccdc(dates, values, qa, conseq_anom=conseq_anom)
        return row, col, segments
    except Exception:
        return row, col, []

def run_ccdc_array(dates: "np.ndarray", raster_stack: "np.ndarray", qa_stack: "np.ndarray", max_segments: int = 6, n_jobs: int = -1, return_coefs: bool = True, conseq_anom: int = 3) -> "np.ndarray":
    """
    Apply CCDC across a 4D numpy array (bands, time, rows, cols) using multiprocessing.
    """
    num_bands, time_steps, rows, cols = raster_stack.shape
    
    params_per_segment = 3 + num_bands * 7
    
    if return_coefs:
        output_stack = np.zeros((max_segments, params_per_segment, rows, cols), dtype=np.float32)
    else:
        output_stack = np.zeros((max_segments, 1, rows, cols), dtype=np.float32)
    
    pixel_args = [
        (r, c, raster_stack[:, :, r, c], qa_stack[:, r, c]) 
        for r in range(rows) for c in range(cols)
    ]
    
    worker = partial(_process_pixel_ccdc, dates=dates, max_segments=max_segments, conseq_anom=conseq_anom)
                     
    if n_jobs == -1:
        import os as _os
        n_jobs = _os.cpu_count() or 4
        
    if n_jobs == 1:
        results = list(map(worker, pixel_args))
    else:
        with Pool(processes=n_jobs) as pool:
            results = pool.map(worker, pixel_args)
        
    for row, col, segments in results:
        n_segs = len(segments)
        if n_segs == 0:
            continue
            
        n_segs = min(n_segs, max_segments)
        
        for i in range(n_segs):
            seg = segments[i]
            if return_coefs:
                output_stack[i, 0, row, col] = seg['t_start']
                output_stack[i, 1, row, col] = seg['t_end']
                output_stack[i, 2, row, col] = seg['t_break'] if seg['t_break'] > 0 else 0
                
                idx = 3
                for b in range(num_bands):
                    output_stack[i, idx, row, col] = seg['rmse'][b]
                    idx += 1
                    for c_idx in range(6):
                        output_stack[i, idx, row, col] = seg['coefs'][b][c_idx]
                        idx += 1
            else:
                output_stack[i, 0, row, col] = seg['t_break'] if seg['t_break'] > 0 else 0
            
    return output_stack

def run_ccdc_image(input_path: str, output_dir: str, dates: "np.ndarray", num_bands: int = 6, qa_band_idx: int = -1,
                   max_segments: int = 6, chunk_size: int = 512, n_jobs: int = -1, prefix: str = "ccdc_break", return_coefs: bool = True, conseq_anom: int = 3) -> None:
    """
    High-level function to process a full GeoTIFF stack using CCDC with chunking.
    Assumes the stack is interleaved by date.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(input_path) as src:
        total_layers = src.count
        num_dates = total_layers // num_bands
        
        if len(dates) != num_dates:
            raise ValueError(f"Provided {len(dates)} dates, but image has {num_dates} dates.")
            
        profile = src.profile
        height, width = src.height, src.width
        
        print(f"CCDC Image dimensions: {width}x{height} pixels, {num_dates} dates, {num_bands} bands per date")
        
        out_path = os.path.join(output_dir, f"{prefix}_coefs.tif")
        p = profile.copy()
        
        params_per_seg = (3 + num_bands * 7) if return_coefs else 1
        total_bands = max_segments * params_per_seg
        
        p.update(count=total_bands, dtype='float32', nodata=0, driver='GTiff')
        
        dst_breaks = rasterio.open(out_path, 'w', **p)
        
        print(f"Processing CCDC in chunks of {chunk_size}x{chunk_size}...")
        for row in range(0, height, chunk_size):
            for col in range(0, width, chunk_size):
                window = Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row))
                print(f"  Chunk: Row {row}-{row+window.height}, Col {col}-{col+window.width}")
                
                stack = src.read(window=window)
                stack = stack.reshape((num_dates, num_bands, window.height, window.width))
                stack = np.transpose(stack, (1, 0, 2, 3))
                
                if qa_band_idx >= 0 and qa_band_idx < num_bands:
                    qa_stack = stack[qa_band_idx, :, :, :].astype(int)
                else:
                    qa_stack = np.zeros((num_dates, window.height, window.width), dtype=int)
                    
                breaks_stack = run_ccdc_array(
                    dates=dates, 
                    raster_stack=stack, 
                    qa_stack=qa_stack,
                    max_segments=max_segments, 
                    n_jobs=n_jobs,
                    return_coefs=return_coefs,
                    conseq_anom=conseq_anom
                )
                
                # breaks_stack shape: (max_segments, params_per_seg, height, width)
                # flatten the first two dimensions for writing
                breaks_stack = breaks_stack.reshape((total_bands, window.height, window.width))
                
                dst_breaks.write(breaks_stack.astype('float32'), window=window)
                
        dst_breaks.close()
            
    print(f"Successfully processed CCDC and saved break dates to {output_dir}")

def run_landtrendr_image(input_path: str, output_dir: str, start_year: int = 2000, max_segments: int = 6,
                            chunk_size: int = 512, n_jobs: int = -1, save_vertices: bool = False,
                            event_type: str = "loss", sort_by: str = "greatest", min_mag: float = 0.0, 
                            min_dur: int = 1, pre_val_thresh: float = 0.0, prefix: str = "lt_event", pval_threshold: float = 0.05,
                            output_scale_factor: float = 1.0) -> None:
    """
    High-level function to process a full GeoTIFF file using LandTrendr with chunking to save RAM.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(input_path) as src:
        num_years = src.count
        years = np.arange(start_year, start_year + num_years)
        profile = src.profile
        height, width = src.height, src.width
        
        print(f"Image dimensions: {width}x{height} pixels, {num_years} bands/years")
        
        out_path = os.path.join(output_dir, "lt_vertices.tif")
        p = profile.copy()
        max_vertices = max_segments + 1
        p.update(count=max_vertices * 2, dtype='float32', nodata=0, driver='GTiff')
        
        p_2d = profile.copy()
        p_2d.update(count=1, driver='GTiff')
        dtypes = {
            "yod": "uint16", "magnitude": "float32", "duration": "uint16",
            "pre_val": "float32", "post_val": "float32", "rate": "float32"
        }
        
        dst_events = {}
        for metric, dt in dtypes.items():
            p_tmp = p_2d.copy()
            p_tmp.update(dtype=dt)
            if dt == 'uint16' and p_tmp.get('nodata') is not None and p_tmp['nodata'] < 0:
                p_tmp['nodata'] = 0
            dst_events[metric] = rasterio.open(os.path.join(output_dir, f"{prefix}_{metric}.tif"), 'w', **p_tmp)
            
        dst_vertices = rasterio.open(out_path, 'w', **p) if save_vertices else None
        
        print(f"Processing in chunks of {chunk_size}x{chunk_size}...")
        
        # Calcular total de chunks para a barra de progresso
        rows_range = list(range(0, height, chunk_size))
        cols_range = list(range(0, width, chunk_size))
        total_chunks = len(rows_range) * len(cols_range)
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_chunks, desc="LandTrendr", unit="chunk")
        except ImportError:
            pbar = None
            
        for row in rows_range:
            for col in cols_range:
                window = Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row))
                if pbar is None:
                    print(f"  Chunk: Row {row}-{row+window.height}, Col {col}-{col+window.width}")
                
                stack = src.read(window=window)
                
                vertices_stack = run_landtrendr_array(
                    years, stack, 
                    max_segments=max_segments, 
                    pval_threshold=pval_threshold,
                    n_jobs=n_jobs
                )
                
                events = extract_events(
                    vertices_stack,
                    event_type=event_type,
                    sort_by=sort_by,
                    min_magnitude=min_mag,
                    min_duration=min_dur,
                    pre_val_threshold=pre_val_thresh
                )
                
                if output_scale_factor != 1.0:
                    events['magnitude'] *= output_scale_factor
                    events['pre_val'] *= output_scale_factor
                    events['post_val'] *= output_scale_factor
                    events['rate'] *= output_scale_factor
                    
                    if dst_vertices:
                        max_vertices = max_segments + 1
                        vertices_stack[max_vertices:, :, :] *= output_scale_factor

                if dst_vertices:
                    dst_vertices.write(vertices_stack.astype('float32'), window=window)
                
                for metric, data in events.items():
                    dst_events[metric].write(data, 1, window=window)
                    
                if pbar is not None:
                    pbar.update(1)
                    
        if pbar is not None:
            pbar.close()
            
        if dst_vertices: 
            dst_vertices.close()
        for f in dst_events.values(): 
            f.close()
            
    print(f"Successfully processed and saved layers to {output_dir}")
