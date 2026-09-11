import numpy as np
import xarray as xr
import dask.array as da
import pytest
from unittest.mock import patch

from cdts.phenology import run_phenology_dask
import cdts.xarray_api # Registers the accessor

def mock_fit_phenology_batch(values_array, dates_array, curve_type, extraction_method, max_seasons, whittaker_lambda, apply_whittaker, apply_hants, hants_frequencies, hants_threshold, min_season_length, min_amplitude, min_pixel_amplitude, n_jobs, **kwargs):
    n_pixels = values_array.shape[0]
    sos = np.zeros((n_pixels, max_seasons), dtype=np.float32)
    eos = np.zeros((n_pixels, max_seasons), dtype=np.float32)
    los = np.zeros((n_pixels, max_seasons), dtype=np.float32)
    pop = np.zeros((n_pixels, max_seasons), dtype=np.float32)
    for s in range(max_seasons):
        # 100.0 days + s * 365 days ensures each season lands in a different year
        sos[:, s] = 100.0 + s * 365.25
        eos[:, s] = 200.0 + s * 365.25
        los[:, s] = 100.0
        pop[:, s] = 150.0 + s * 365.25
    return sos, eos, los, pop

@patch('cdts.phenology.fit_phenology_batch', side_effect=mock_fit_phenology_batch)
def test_run_phenology_dask(mock_fit):
    time_steps = 20
    rows = 10
    cols = 10
    max_seasons = 2
    
    data = da.random.random((time_steps, rows, cols), chunks=(time_steps, 5, 5))
    dates = np.arange(time_steps)
    
    result = run_phenology_dask(data, dates, curve_type=0, max_seasons=max_seasons)
    
    assert result.shape == (4, max_seasons, rows, cols)
    assert result.chunks == ((4,), (max_seasons,), (5, 5), (5, 5))
    
    res_computed = result.compute()
    
    assert res_computed.shape == (4, max_seasons, rows, cols)
    # SOS is at index 0
    np.testing.assert_allclose(res_computed[0, 0, 0, 0], 100.0)
    # EOS is at index 1
    np.testing.assert_allclose(res_computed[1, 0, 0, 0], 200.0)

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
    assert res.shape == (4, 2, rows, cols)
    
    res_computed = res.compute()
    sos_mean = res_computed.loc[{"metric": "SOS"}].values.mean()
    assert 99.0 <= sos_mean <= 101.0
    eos_mean = res_computed.loc[{"metric": "EOS"}].values.mean()
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
    assert res_computed.shape == (4, max_seasons, rows, cols)
    
    # Try with another extraction method (GU)
    res_gu = ds.cdts.run_phenology(
        dates=dates, 
        curve_type=0, # BECK
        extraction_method=2, # GU
        max_seasons=max_seasons,
        apply_whittaker=True,
        apply_hants=False,
        n_jobs=2
    ).compute()
    
    assert res_gu.shape == (4, max_seasons, rows, cols)

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
    assert res.shape == (4, 3, rows, cols)
