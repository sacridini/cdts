import xarray as xr
import numpy as np
import pandas as pd
import pytest
from cdts.regularize import regularize_time_series

def test_regularize_median():
    time = pd.date_range("2020-01-01", periods=5, freq="5D")
    data = np.random.rand(5, 2, 3, 3)
    cube = xr.DataArray(data, dims=["time", "band", "y", "x"], coords={"time": time})
    
    result = regularize_time_series(cube, freq="16D", method="median")
    assert result.dims == ("time", "band", "y", "x")
    # 5 intervals of 5 days spans 20 days total. Start 01-01. Resampling '16D' from 01-01 gives bins: 01-01 and 01-17.
    assert len(result.time) == 2

def test_regularize_medoid():
    time = pd.date_range("2020-01-01", periods=5, freq="5D")
    data = np.random.rand(5, 2, 3, 3)
    cube = xr.DataArray(data, dims=["time", "band", "y", "x"], coords={"time": time})
    
    result = regularize_time_series(cube, freq="16D", method="medoid")
    assert result.dims == ("time", "band", "y", "x")
    assert len(result.time) == 2
