import numpy as np
import pytest
from cdts import desawtooth, run_landtrendr
from cdts.raster import run_landtrendr_array

def test_desawtooth_removes_spike():
    values = np.array([0.1, 0.12, 0.9, 0.15, 0.11, 0.13])
    filtered = desawtooth(values)
    assert filtered[2] < 0.9
    assert filtered[2] < 0.4
    assert np.isclose(filtered[0], 0.1)
    assert np.isclose(filtered[-1], 0.13)

def test_desawtooth_no_spike():
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    filtered = desawtooth(values)
    np.testing.assert_allclose(values, filtered, atol=0.01)

def test_run_landtrendr_basic():
    years = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
    values = np.array([0.9, 0.9, 0.85, 0.2, 0.3, 0.4, 0.5, 0.85, 0.9, 0.9])
    vertices = run_landtrendr(years, values, max_segments=2)
    assert len(vertices) == 3
    assert vertices[0]['year'] == 2000
    assert vertices[1]['year'] == 2003
    assert vertices[2]['year'] == 2009

def test_run_landtrendr_array():
    years = np.array([2000, 2001, 2002, 2003, 2004])
    stack = np.zeros((5, 2, 2))
    
    # Exactly piecewise-linear V (vertices at 2000/2002/2004) so the F-test is
    # significant even with only 5 observations (2 residual degrees of freedom).
    stack[:, 0, 0] = [0.8, 0.5, 0.2, 0.4, 0.6]
    stack[:, 1, 1] = [0, 0, 0, 0, 0]
    
    output = run_landtrendr_array(years, stack, max_segments=2, n_jobs=2, min_observations_needed=5)
    
    assert output.shape == (6, 2, 2)
    assert output[0, 0, 0] == 2000
    assert output[1, 0, 0] == 2002
    assert output[2, 0, 0] == 2004
    assert np.all(output[:, 1, 1] == 0)

def test_run_landtrendr_batch():
    from cdts.landtrendr import run_landtrendr_batch
    
    # 2 pixels, 5 years
    years = np.array([2000, 2001, 2002, 2003, 2004])
    
    # values shape [Y, X, Time]
    values = np.zeros((1, 2, 5))
    # Pixel 0: Drop in 2002, exactly piecewise-linear so the F-test clears
    # significance even with only 5 observations (2 residual degrees of freedom).
    values[0, 0, :] = [0.8, 0.5, 0.2, 0.4, 0.6]
    # Pixel 1: No data
    values[0, 1, :] = [-9999.0] * 5
    
    max_segments = 2
    verts, counts, rmse = run_landtrendr_batch(years, values, max_segments=max_segments, min_observations_needed=5)

    assert counts.shape == (2,)
    assert verts.shape == (2, max_segments + 1, 2)
    assert rmse.shape == (2,)

    # Pixel 0 should have 3 vertices (start, break, end), fit exactly (SSE=0).
    assert counts[0] == 3
    assert verts[0, 0, 0] == 2000
    assert verts[0, 1, 0] == 2002
    assert verts[0, 2, 0] == 2004
    assert np.isclose(rmse[0], 0.0, atol=1e-6)

    # Pixel 1 should have 0 vertices and no RMSE (fitting was skipped, no-data).
    assert counts[1] == 0
    assert rmse[1] == 0.0

def test_run_landtrendr_min_observations_needed():
    # Below min_observations_needed, fitting is skipped entirely and the raw
    # (unsegmented) trajectory is returned as-is.
    years = np.array([2000, 2001, 2002, 2003])
    values = np.array([0.8, 0.5, 0.2, 0.6])

    vertices = run_landtrendr(years, values, max_segments=2, min_observations_needed=6)
    assert len(vertices) == 4
    assert [v['year'] for v in vertices] == [2000, 2001, 2002, 2003]
    assert [v['value'] for v in vertices] == list(values)

    # With the gate lowered, normal segmentation proceeds instead.
    vertices = run_landtrendr(years, values, max_segments=2, min_observations_needed=2)
    assert len(vertices) <= 3

def test_run_landtrendr_sequential_fit_smooths_noisy_segment():
    # A noisy-but-stable segment (2000-2003, endpoints both 0.50) followed by a
    # clean drop to 2005. The per-segment regression-vs-point-to-point choice
    # (Kennedy et al. 2010 Sec. 2.5.3) should pick the regression line for the
    # first segment, since it has lower SSE against the noisy interior points
    # than the flat point-to-point line through the two (coincidentally equal)
    # endpoint values -- so the fitted endpoint values should NOT just be the
    # raw 0.50/0.50, but a slightly sloped regression line through all 4 points.
    years = np.array([2000, 2001, 2002, 2003, 2004, 2005])
    values = np.array([0.50, 0.54, 0.46, 0.50, 0.30, 0.10])

    vertices = run_landtrendr(years, values, max_segments=3, min_observations_needed=6)
    by_year = {v['year']: v['value'] for v in vertices}

    assert 2000 in by_year and 2003 in by_year
    assert not np.isclose(by_year[2000], 0.50, atol=1e-6)
    assert not np.isclose(by_year[2003], 0.50, atol=1e-6)
    assert np.isclose(by_year[2000], 0.512, atol=1e-6)
    assert np.isclose(by_year[2003], 0.488, atol=1e-6)

def test_run_landtrendr_array_return_rmse():
    years = np.array([2000, 2001, 2002, 2003, 2004])
    stack = np.zeros((5, 2, 2))
    # Exactly piecewise-linear V again -> RMSE of the selected fit should be ~0.
    stack[:, 0, 0] = [0.8, 0.5, 0.2, 0.4, 0.6]
    stack[:, 1, 1] = [0, 0, 0, 0, 0]

    output, rmse_map = run_landtrendr_array(years, stack, max_segments=2, min_observations_needed=5, return_rmse=True)

    assert output.shape == (6, 2, 2)
    assert rmse_map.shape == (2, 2)
    assert np.isclose(rmse_map[0, 0], 0.0, atol=1e-5)
    assert rmse_map[1, 1] == 0.0  # no-data pixel: fitting was skipped

def test_extract_events_dsnr():
    from cdts.metrics import extract_events

    years = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006])
    stack = np.zeros((7, 1, 1))
    # A disturbance (drop) at 2003 with some noise elsewhere, so the fit RMSE
    # is a real, non-zero value the DSNR calculation actually normalizes by.
    stack[:, 0, 0] = [0.82, 0.79, 0.81, 0.30, 0.32, 0.29, 0.31]

    vertices_stack, rmse_map = run_landtrendr_array(
        years, stack, max_segments=2, min_observations_needed=7, return_rmse=True
    )

    with pytest.raises(ValueError):
        extract_events(vertices_stack, sort_by="dsnr")  # no rmse_map given

    events = extract_events(vertices_stack, event_type="loss", sort_by="dsnr", rmse_map=rmse_map)
    assert "dsnr" in events
    assert rmse_map[0, 0] > 0.0
    expected_dsnr = events["magnitude"][0, 0] / rmse_map[0, 0]
    assert np.isclose(events["dsnr"][0, 0], expected_dsnr, atol=1e-4)

def test_run_landtrendr_vertex_count_overshoot():
    # A longer, exactly piecewise-linear 3-segment (4-vertex) trajectory.
    # With vertex_count_overshoot=0, the initial angle-culling has to pick the
    # true vertices directly out of every observation; overshoot just gives it
    # more candidates to choose from before pruning back down. Either way the
    # same true vertices should be recoverable, since they're the only ones
    # that make the fit exact (SSE=0).
    years = np.arange(2000, 2013)  # 13 points
    values = np.array([1.0, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 0.6, 0.2, 0.6, 1.0, 1.0, 1.0])

    for vco in (0, 3, 6):
        vertices = run_landtrendr(years, values, max_segments=6, vertex_count_overshoot=vco, min_observations_needed=6)
        years_out = [v['year'] for v in vertices]
        assert years_out[0] == 2000
        assert years_out[-1] == 2012
