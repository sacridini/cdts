import pytest
import numpy as np
import xarray as xr
import dask.array as da
import cdts  # This registers the xarray accessor automatically

def test_xarray_ccdc_accessor():
    """
    Test that the xarray accessor correctly maps the C++ CCDC algorithm over Dask blocks.
    """
    # Create dummy data: (bands, time, y, x)
    bands, time, y, x = 3, 20, 2, 2
    
    # Random scaled reflectance data
    data = np.random.randint(0, 10000, size=(bands, time, y, x), dtype=np.int16)
    
    # Create Dask-backed xarray DataArray chunked spatially (y:1, x:1)
    dask_data = da.from_array(data, chunks=(bands, time, 1, 1))
    
    da_arr = xr.DataArray(
        dask_data, 
        dims=["band", "time", "y", "x"],
        coords={"y": [10, 20], "x": [30, 40]}
    )
    
    # Dummy fractional years
    dates = np.linspace(2000.0, 2010.0, time)
    
    # Run CCDC lazily
    max_segments = 4
    return_coefs = True
    params_per_seg = 3 + (bands * 7) if return_coefs else 1
    
    result = da_arr.cdts.run_ccdc(dates=dates, max_segments=max_segments, return_coefs=return_coefs)
    
    # Check that it's still lazy (Dask array inside)
    assert isinstance(result.data, da.Array)
    
    # Check expected output dimensions
    assert result.dims == ("segment", "parameter", "y", "x")
    assert result.shape == (max_segments, params_per_seg, y, x)
    
    # Compute the graph and verify execution completes without crashing
    computed_result = result.compute()
    assert computed_result.shape == (max_segments, params_per_seg, y, x)

def test_xarray_landtrendr_accessor():
    """
    Test that the xarray accessor correctly maps the C++ LandTrendr algorithm over Dask blocks.
    """
    # Create dummy data: (time, y, x) for LandTrendr (single index)
    time, y, x = 15, 2, 2
    
    # Generate an index with a sudden drop
    data = np.linspace(8000, 7000, time)
    data[7:] -= 3000  # Introduce a break
    data = np.broadcast_to(data[:, None, None], (time, y, x)).copy()
    
    # Chunk spatially
    dask_data = da.from_array(data, chunks=(time, 1, 1))
    
    da_arr = xr.DataArray(
        dask_data,
        dims=["time", "y", "x"],
        coords={"y": [10, 20], "x": [30, 40]}
    )
    
    years = np.arange(2000, 2000 + time)
    
    max_segments = 3
    max_vertices = max_segments + 1
    
    # Run LandTrendr lazily
    result = da_arr.cdts.run_landtrendr(years=years, max_segments=max_segments)
    
    # Check laziness
    assert isinstance(result.data, da.Array)
    
    # Check output shape: (2 * max_vertices, y, x) 
    # where the first max_vertices are years, and the next are fitted values
    assert result.dims == ("vertex_info", "y", "x")
    assert result.shape == (2 * max_vertices, y, x)
    
    # Compute and verify
    computed_result = result.compute()
    assert computed_result.shape == (2 * max_vertices, y, x)
    # The first row for pixel (0,0) should contain the start year 2000
    assert computed_result[0, 0, 0] > 0 
