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
    
    stack[:, 0, 0] = [0.8, 0.8, 0.2, 0.4, 0.6]
    stack[:, 1, 1] = [0, 0, 0, 0, 0]
    
    output = run_landtrendr_array(years, stack, max_segments=2, n_jobs=2)
    
    assert output.shape == (6, 2, 2)
    assert output[0, 0, 0] == 2000
    assert output[1, 0, 0] == 2002
    assert output[2, 0, 0] == 2004
    assert np.all(output[:, 1, 1] == 0)
