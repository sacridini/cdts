import numpy as np
import xarray as xr
import dask.array as da
import pytest
from unittest.mock import patch

from cdts.phenology import run_phenology_dask
import cdts.xarray_api # Registers the accessor

def mock_fit_phenology_batch(values_array, dates_array, curve_type, max_seasons, whittaker_lambda, apply_whittaker, n_jobs, **kwargs):
    n_pixels = values_array.shape[0]
    sos = np.full((n_pixels, max_seasons), 100.0, dtype=np.float32)
    eos = np.full((n_pixels, max_seasons), 200.0, dtype=np.float32)
    los = np.full((n_pixels, max_seasons), 100.0, dtype=np.float32)
    pop = np.full((n_pixels, max_seasons), 150.0, dtype=np.float32)
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
    assert res.dims == ("metric", "season", "y", "x")
    assert res.shape == (4, 2, rows, cols)
    
    res_computed = res.compute()
    assert res_computed.loc[{"metric": "SOS"}].values.mean() == 100.0
    assert res_computed.loc[{"metric": "EOS"}].values.mean() == 200.0

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
    
    # Check that it extracted valid dates (not all NaNs)
    sos = res_computed.loc[{"metric": "SOS"}].values
    assert not np.all(np.isnan(sos)), "All SOS values are NaN, extraction failed"
    
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
