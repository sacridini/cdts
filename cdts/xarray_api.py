import xarray as xr
import dask.array as da
import numpy as np
from typing import Optional, Any
from cdts.raster import run_ccdc_array, run_landtrendr_array

@xr.register_dataarray_accessor("cdts")
class cdtsAccessor:
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def run_ccdc(self, dates: np.ndarray, qa_stack: Optional[Any] = None, max_segments: int = 6, return_coefs: bool = True, conseq_anom: int = 3) -> xr.DataArray:
        """
        Runs CCDC on an xarray DataArray using Dask for out-of-core and parallel execution.
        Assumes DataArray shape: (bands, time, y, x).
        """
        arr = self._obj.data
        bands, time_steps, rows, cols = self._obj.shape
        
        if qa_stack is None:
            qa_stack = da.zeros((time_steps, rows, cols), dtype=int, chunks=(time_steps, arr.chunks[2], arr.chunks[3]))
        elif isinstance(qa_stack, xr.DataArray):
            qa_stack = qa_stack.data
            
        params_per_seg = (3 + bands * 7) if return_coefs else 1
        
        # We need to map blocks. The input to map_blocks will be a chunk of the array.
        # But run_ccdc_array expects the entire time dimension intact.
        # Ensure chunks over bands and time are -1 (un-chunked)
        
        def _ccdc_block(block, qa_block):
            # block shape: (bands, time, y_chunk, x_chunk)
            if block.size == 0:
                return np.zeros((max_segments, params_per_seg, block.shape[2], block.shape[3]), dtype=np.float32)
            
            # Call our numpy function
            # Limit n_jobs=1 because Dask handles the multi-processing across blocks
            return run_ccdc_array(
                dates, block, qa_block, 
                max_segments=max_segments, 
                n_jobs=1, 
                return_coefs=return_coefs,
                conseq_anom=conseq_anom
            )
            
        # Using dask map_blocks
        out = da.map_blocks(
            _ccdc_block,
            arr,
            qa_stack,
            dtype=np.float32,
            drop_axis=[0, 1], # drop bands and time
            new_axis=[0, 1],  # add max_segments and params_per_seg
            chunks=(max_segments, params_per_seg, arr.chunks[2], arr.chunks[3])
        )
        
        # Return a new xarray
        return xr.DataArray(
            out,
            dims=["segment", "parameter", "y", "x"],
            coords={
                "y": self._obj.coords.get("y"),
                "x": self._obj.coords.get("x")
            }
        )

    def run_landtrendr(self, years: np.ndarray, max_segments: int = 6, pval_threshold: float = 0.05) -> xr.DataArray:
        """
        Runs LandTrendr on an xarray DataArray using Dask for out-of-core and parallel execution.
        Assumes DataArray shape: (time, y, x).
        """
        arr = self._obj.data
        time_steps, rows, cols = self._obj.shape
        
        max_vertices = max_segments + 1
        
        def _lt_block(block):
            if block.size == 0:
                return np.zeros((2 * max_vertices, block.shape[1], block.shape[2]), dtype=np.float32)
            
            # Chama a função em C++ para o bloco, limitando n_jobs=1 pois o Dask faz o paralelismo
            return run_landtrendr_array(
                years, block, 
                max_segments=max_segments, 
                pval_threshold=pval_threshold,
                n_jobs=1
            )
            
        out = da.map_blocks(
            _lt_block,
            arr,
            dtype=np.float32,
            drop_axis=[0], # remove time
            new_axis=[0],  # adiciona os vertices
            chunks=(2 * max_vertices, arr.chunks[1], arr.chunks[2])
        )
        
        return xr.DataArray(
            out,
            dims=["vertex_info", "y", "x"], # vertex_info: (anos, valores ajustados)
            coords={
                "y": self._obj.coords.get("y"),
                "x": self._obj.coords.get("x")
            }
        )

