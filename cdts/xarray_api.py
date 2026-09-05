import xarray as xr
import dask.array as da
import numpy as np
from typing import Optional, Any
from cdts.raster import run_ccdc_array, run_landtrendr_array

@xr.register_dataarray_accessor("cdts")
class cdtsAccessor:
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def run_ccdc(self, dates: np.ndarray, qa_stack: Optional[Any] = None, max_segments: int = 6, return_coefs: bool = True, conseq_anom: int = 3, n_jobs: int = -1) -> xr.DataArray:
        """
        Runs CCDC on an xarray DataArray using Dask for out-of-core and parallel execution.
        Assumes DataArray shape: (bands, time, y, x).
        
        Strategy A: Dask handles cross-node distribution (map_blocks), OpenMP handles multi-core within the node (n_jobs=-1).
        WARNING: If using n_jobs=-1, ensure Dask is configured to run with only 1 worker process per physical machine!
        """
        arr = self._obj.data
        bands, time_steps, rows, cols = self._obj.shape
        
        if qa_stack is None:
            qa_stack = da.zeros((time_steps, rows, cols), dtype=int, chunks=(time_steps, arr.chunks[2], arr.chunks[3]))
        elif isinstance(qa_stack, xr.DataArray):
            qa_stack = qa_stack.data
            
        params_per_seg = (3 + bands * 7) if return_coefs else 1
        
        def _ccdc_block(block, qa_block):
            if block.size == 0:
                return np.zeros((max_segments, params_per_seg, block.shape[2], block.shape[3]), dtype=np.float32)
            
            return run_ccdc_array(
                dates, block, qa_block, 
                max_segments=max_segments, 
                n_jobs=n_jobs, 
                return_coefs=return_coefs,
                conseq_anom=conseq_anom
            )
            
        out = da.map_blocks(
            _ccdc_block,
            arr,
            qa_stack,
            dtype=np.float32,
            drop_axis=[0, 1], # drop bands and time
            new_axis=[0, 1],  # add max_segments and params_per_seg
            chunks=(max_segments, params_per_seg, arr.chunks[2], arr.chunks[3])
        )
        
        return xr.DataArray(
            out,
            dims=["segment", "parameter", "y", "x"],
            coords={
                "y": self._obj.coords.get("y"),
                "x": self._obj.coords.get("x")
            }
        )

    def run_landtrendr(self, years: np.ndarray, max_segments: int = 6, pval_threshold: float = 0.05, n_jobs: int = -1) -> xr.DataArray:
        """
        Runs LandTrendr on an xarray DataArray using Dask for out-of-core and parallel execution.
        Assumes DataArray shape: (time, y, x).
        
        Strategy A: Dask handles cross-node distribution (map_blocks), OpenMP handles multi-core within the node (n_jobs=-1).
        WARNING: If using n_jobs=-1, ensure Dask is configured to run with only 1 worker process per physical machine!
        """
        arr = self._obj.data
        time_steps, rows, cols = self._obj.shape
        
        max_vertices = max_segments + 1
        
        def _lt_block(block):
            if block.size == 0:
                return np.zeros((2 * max_vertices, block.shape[1], block.shape[2]), dtype=np.float32)
            
            return run_landtrendr_array(
                years, block, 
                max_segments=max_segments, 
                pval_threshold=pval_threshold,
                n_jobs=n_jobs
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
            dims=["vertex_info", "y", "x"],
            coords={
                "y": self._obj.coords.get("y"),
                "x": self._obj.coords.get("x")
            }
        )
        
    def to_zarr_optimized(self, store_path: str, chunk_size: dict = {"y": 512, "x": 512}) -> None:
        """
        Optimizes and saves the DataArray directly to a Zarr store, ideal for cloud storage (S3/GCS) 
        and rapid multidimensional time-series queries.
        """
        # Ensure optimal chunking before saving
        optimized_ds = self._obj.chunk(chunk_size)
        optimized_ds.to_dataset(name="data").to_zarr(store_path, mode="w", consolidated=True)


