import xarray as xr
import numpy as np
import pandas as pd
import os
from typing import Optional

try:
    from cdts._core.utils import compute_medoid
except ImportError:
    # Fallback or just let it crash if C++ extension is not built
    def compute_medoid(array, nodata):
        raise NotImplementedError("C++ extension not built. Run 'pip install -e .' first.")

def _medoid_per_year(da_year: xr.DataArray, no_data_value: float = -9999.0) -> xr.DataArray:
    """
    Helper function to apply C++ medoid computation per year.
    da_year is expected to have dimensions (time, band, y, x).
    """
    # Xarray standard dims usually: (time, band, y, x)
    # We transpose to (y, x, time, band) for cache-locality in C++
    da_transposed = da_year.transpose("y", "x", "time", "band")
    # Must ensure the array is C-contiguous in memory so PyBind11 ptr arithmetic works!
    np_array = np.ascontiguousarray(da_transposed.values, dtype=np.float64)
    
    # Run optimized C++ Eigen Medoid
    out_np = compute_medoid(np_array, no_data_value)
    
    # out_np shape is (y, x, band)
    # Reconstruct back to (band, y, x)
    out_np_transposed = np.transpose(out_np, (2, 0, 1))
    
    return xr.DataArray(
        out_np_transposed,
        dims=["band", "y", "x"],
        coords={
            "band": da_year.coords.get("band"),
            "y": da_year.coords.get("y"),
            "x": da_year.coords.get("x")
        }
    )

def cbers_to_landtrendr(cube: xr.DataArray, no_data_value: float = -9999.0) -> xr.DataArray:
    """
    Prepares a DataCube for LandTrendr by grouping years and computing 
    an annual medoid composite using a highly optimized C++ backend.
    """
    if "time" not in cube.dims:
        raise ValueError("Cube must have a 'time' dimension")
        
    print("Grouping images by year and computing SIMD-optimized Medoid...")
    
    # Group by year and apply C++ function
    annual_cube = cube.groupby("time.year").map(lambda ds: _medoid_per_year(ds, no_data_value))
    
    # Rename 'year' back to 'time' to maintain temporal structure
    annual_cube = annual_cube.rename({"year": "time"})
    
    # Convert 'time' coord to actual datetimes instead of just year integers (optional but recommended)
    years = annual_cube.time.values
    annual_cube = annual_cube.assign_coords(time=[np.datetime64(f"{y}-07-01") for y in years])
    
    print(f"Annual Medoid cube created with {annual_cube.sizes['time']} years.")
    return annual_cube


def cbers_to_ccdc(cube: xr.DataArray, out_dir: str = ".") -> xr.DataArray:
    """
    Prepares the cube for CCDC, extracting the dense series and generating the 
    CSV of ordinal dates required by CCDC algorithms.
    """
    print("Extracting dates from dense series for CCDC...")
    
    time_values = cube.time.values
    dates_pd = pd.to_datetime(time_values)
    
    ordinal_dates = [d.toordinal() for d in dates_pd]
    dates_str = [d.strftime('%Y-%m-%d') for d in dates_pd]
    
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "cbers_ccdc_dates.csv")
    df = pd.DataFrame({
        'Date': dates_str,
        'Ordinal_Day': ordinal_dates
    })
    df.to_csv(csv_path, index=False)
    
    print(f"Dates successfully saved to: {csv_path}")
    
    return cube
