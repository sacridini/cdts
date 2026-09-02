import os
import numpy as np
import rasterio
import pytest
from cdts.io import save_raster, load_raster

def test_load_raster_basic(tmp_path):
    # Create a dummy array and save it
    array = np.random.randint(0, 10000, size=(10, 50, 50), dtype=np.int16)
    file_path = str(tmp_path / "dummy_stack.tif")
    
    # Save using the existing robust save_raster
    save_raster(array, file_path)
    
    # Load using the new load_raster
    loaded_array, profile = load_raster(file_path)
    
    assert loaded_array.shape == (10, 50, 50)
    assert profile['count'] == 10
    assert profile['dtype'] == 'int16'
    np.testing.assert_array_equal(array, loaded_array)

def test_load_raster_warnings(tmp_path):
    # Unscaled small float array
    array = np.random.rand(2, 50, 50).astype(np.float32)
    file_path = str(tmp_path / "dummy_float_stack.tif")
    save_raster(array, file_path)
    
    with pytest.warns(UserWarning) as record:
        load_raster(file_path, raster_check='landtrendr')
        
    # Should throw warnings for < 3 bands and for not being scaled
    assert len(record) >= 2
    warn_msgs = [str(w.message) for w in record]
    assert any("annual time series" in m for m in warn_msgs)
    assert any("unscaled floats" in m for m in warn_msgs)
    
    with pytest.warns(UserWarning) as record:
        load_raster(file_path, raster_check='ccdc')
        
    assert len(record) >= 2
    warn_msgs = [str(w.message) for w in record]
    assert any("dense time series" in m for m in warn_msgs)
    assert any("unscaled" in m for m in warn_msgs)
