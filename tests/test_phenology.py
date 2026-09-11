import numpy as np
import xarray as xr
import dask.array as da
import pytest
from unittest.mock import patch

from cdts.phenology import run_phenology_dask
import cdts.xarray_api # Registers the accessor

def mock_fit_phenology_batch(values_array, dates_array, curve_type, extraction_method, max_seasons, whittaker_lambda, apply_whittaker, apply_hants, hants_frequencies, hants_threshold, min_season_length, min_amplitude, min_pixel_amplitude, n_jobs, **kwargs):
    n_pixels = values_array.shape[0]
    out_arr = np.zeros((19, n_pixels, max_seasons), dtype=np.float32)
    for s in range(max_seasons):
        out_arr[6, :, s] = 100.0 + s * 365.25 # DER.sos is index 6
        out_arr[8, :, s] = 200.0 + s * 365.25 # DER.eos is index 8
        out_arr[17, :, s] = 100.0             # LOS is index 17
        out_arr[18, :, s] = 150.0 + s * 365.25 # POP is index 18
    return out_arr

@patch('cdts.phenology.fit_phenology_batch', side_effect=mock_fit_phenology_batch)
def test_run_phenology_dask(mock_fit):
    time_steps = 20
    rows = 10
    cols = 10
    max_seasons = 2
    
    data = da.random.random((time_steps, rows, cols), chunks=(time_steps, 5, 5))
    dates = np.arange(time_steps)
    
    result = run_phenology_dask(data, dates, curve_type=0, max_seasons=max_seasons)
    
    assert result.shape == (19, max_seasons, rows, cols)
    assert result.chunks == ((19,), (max_seasons,), (5, 5), (5, 5))
    
    res_computed = result.compute()
    
    assert res_computed.shape == (19, max_seasons, rows, cols)
    # DER.sos is at index 6
    np.testing.assert_allclose(res_computed[6, 0, 0, 0], 100.0)
    # DER.eos is at index 8
    np.testing.assert_allclose(res_computed[8, 0, 0, 0], 200.0)
    # LOS is at index 17
    np.testing.assert_allclose(res_computed[17, 0, 0, 0], 100.0)
    # POP is at index 18
    np.testing.assert_allclose(res_computed[18, 0, 0, 0], 150.0)

@patch('cdts.phenology.fit_phenology_batch', side_effect=mock_fit_phenology_batch)
def test_xarray_accessor_phenology(mock_fit):
    time_steps = 20
    rows = 10
    cols = 10
    
    data = da.random.random((time_steps, rows, cols), chunks=(time_steps, 5, 5))
    ds = xr.DataArray(data, dims=["time", "y", "x"], coords={"y": np.arange(rows), "x": np.arange(cols)})
    dates = np.arange(time_steps)
    
    res = ds.cdts.run_phenology(dates=dates, curve_type=1, max_seasons=2)
    
    assert isinstance(res, xr.DataArray)
    assert res.dims == ("metric", "year", "y", "x")
    assert res.shape == (19, 2, rows, cols)
    
    res_computed = res.compute()
    sos_mean = res_computed.loc[{"metric": "DER.sos"}].values.mean()
    assert 99.0 <= sos_mean <= 101.0
    eos_mean = res_computed.loc[{"metric": "DER.eos"}].values.mean()
    assert 199.0 <= eos_mean <= 201.0

def test_real_phenology_extraction_advanced_params():
    """
    Test the actual C++ backend (no mocks) with the new advanced parameters
    including HANTS, Derivative extraction, and season length constraints.
    """
    time_steps = 46 # 2 years of 16-day composites
    rows = 2
    cols = 2
    max_seasons = 2
    
    # Create realistic DOY dates
    dates = np.array([t * 16 + (t // 23) * 365 for t in range(time_steps)])
    
    # Create a clean synthetic curve
    np.random.seed(42)
    data = np.zeros((time_steps, rows, cols))
    for t in range(time_steps):
        doy = dates[t] % 365
        growth = np.sin((doy / 365.0) * np.pi - np.pi/2)
        growth = (growth + 1) / 2.0
        ndvi = 0.2 + growth * 0.6
        data[t, :, :] = ndvi + np.random.normal(0, 0.05, (rows, cols))
        
    ds = xr.DataArray(da.from_array(data, chunks=(time_steps, 2, 2)), 
                      dims=["time", "y", "x"], 
                      coords={"y": np.arange(rows), "x": np.arange(cols)})
    
    # Run with HANTS + Derivative Method
    res = ds.cdts.run_phenology(
        dates=dates, 
        curve_type=0, # BECK
        extraction_method=1, # DERIVATIVE
        max_seasons=max_seasons,
        apply_whittaker=False,
        apply_hants=True,
        hants_frequencies=3,
        min_season_length=3,
        min_amplitude=0.1,
        n_jobs=1 # Test single thread as well
    )
    
    res_computed = res.compute()
    
    # Ensure it didn't crash and returned valid shapes
    assert res_computed.shape == (19, max_seasons, rows, cols)
    
    # Assert at least some seasons were detected
    # Index 17 is LOS
    assert np.nanmean(res_computed.loc[{"metric": "LOS"}].values) > 0

@patch('cdts.phenology.fit_phenology_batch', side_effect=mock_fit_phenology_batch)
def test_xarray_accessor_phenology_no_annual(mock_fit):
    time_steps = 20
    rows = 5
    cols = 5

    data = da.random.random((time_steps, rows, cols), chunks=(time_steps, 5, 5))
    ds = xr.DataArray(data, dims=["time", "y", "x"], coords={"y": np.arange(rows), "x": np.arange(cols)})
    dates = np.arange(time_steps)

    res = ds.cdts.run_phenology(dates=dates, curve_type=1, max_seasons=3, return_annual=False)

    assert isinstance(res, xr.DataArray)
    assert res.dims == ("metric", "season", "y", "x")
    assert res.shape == (19, 3, rows, cols)
