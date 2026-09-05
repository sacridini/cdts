import pytest
import numpy as np
import xarray as xr
import pandas as pd
import os

from cdts.preprocessor import cbers_to_landtrendr, cbers_to_ccdc

def _create_dummy_cube():
    # Create a dummy xarray DataArray with shape (time=4, band=3, y=10, x=10)
    # 2 dates in 2020, 2 dates in 2021
    times = pd.to_datetime(['2020-01-15', '2020-07-20', '2021-02-10', '2021-08-15'])
    bands = ['SR_B2', 'SR_B3', 'SR_B4']
    y = np.arange(10)
    x = np.arange(10)
    
    # Initialize with some values
    data = np.random.rand(4, 3, 10, 10) * 1000
    
    # Insert some nodata (-9999) to test the C++ robust handling
    data[1, :, 5, 5] = -9999.0
    
    cube = xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={"time": times, "band": bands, "y": y, "x": x}
    )
    return cube

def test_cbers_to_landtrendr():
    cube = _create_dummy_cube()
    annual_cube = cbers_to_landtrendr(cube, no_data_value=-9999.0)
    
    assert 'time' in annual_cube.dims
    assert annual_cube.sizes['time'] == 2
    assert annual_cube.sizes['band'] == 3
    assert annual_cube.sizes['y'] == 10
    assert annual_cube.sizes['x'] == 10
    
    # In 2020 (index 0), slice 1 has nodata. Slice 0 is valid.
    assert annual_cube.isel(time=0, band=0, y=5, x=5).values != -9999.0

def test_cbers_to_landtrendr_all_nodata():
    cube = _create_dummy_cube()
    # Make a specific pixel completely nodata for 2020 (time indices 0 and 1)
    # Modify data explicitly using xarray loc
    cube.loc[dict(time=cube.time[0:2], y=3, x=3)] = -9999.0
    
    annual_cube = cbers_to_landtrendr(cube, no_data_value=-9999.0)
    
    # For 2020, pixel (3,3) should be entirely -9999.0
    for b in range(annual_cube.sizes['band']):
        assert annual_cube.isel(time=0, band=b, y=3, x=3).values == -9999.0
        
    # But for 2021 (time index 1), it should have valid data
    assert annual_cube.isel(time=1, band=0, y=3, x=3).values != -9999.0

def test_cbers_to_landtrendr_missing_time_dim():
    cube = _create_dummy_cube()
    cube_no_time = cube.isel(time=0).drop_vars("time")
    
    with pytest.raises(ValueError, match="Cube must have a 'time' dimension"):
        cbers_to_landtrendr(cube_no_time)

def test_cbers_to_landtrendr_math_correctness():
    """Verify that the medoid algorithm picks the mathematically correct slice."""
    times = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
    # 3 times, 1 band, 1x1 pixel
    # Value array: 100, 200, 1000
    # Median is 200. Distance to 200: (100-200)^2=10000; (200-200)^2=0; (1000-200)^2=640000
    # Medoid should pick the 2nd slice (value 200)
    data = np.array([[[[100.0]]], [[[200.0]]], [[[1000.0]]]])
    
    cube = xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={"time": times}
    )
    
    annual_cube = cbers_to_landtrendr(cube)
    
    # Expected output for 2020 is 200.0
    assert annual_cube.isel(time=0, band=0, y=0, x=0).values == 200.0

def test_cbers_to_ccdc(tmp_path):
    cube = _create_dummy_cube()
    out_cube = cbers_to_ccdc(cube, out_dir=str(tmp_path))
    
    assert out_cube.sizes['time'] == 4
    
    csv_file = tmp_path / "cbers_ccdc_dates.csv"
    assert csv_file.exists()
    
    df = pd.read_csv(csv_file)
    assert len(df) == 4
    assert 'Ordinal_Day' in df.columns
    assert 'Date' in df.columns
    
    # Verify ordinal day for the first date
    expected_ordinal = pd.to_datetime('2020-01-15').toordinal()
    assert df.loc[0, 'Ordinal_Day'] == expected_ordinal

