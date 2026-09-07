import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import xarray as xr

from cdts.local import build_local_cube

@pytest.fixture
def dummy_tif_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create some dummy TIFs using rasterio
        transform = from_origin(0, 0, 10, 10)
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'nodata': 0,
            'width': 10,
            'height': 10,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform,
        }
        
        # Simulated file names with dates and bands
        files = [
            ("SENTINEL_20220101_B02.tif", "20220101", "B02"),
            ("SENTINEL_20220101_B03.tif", "20220101", "B03"),
            ("SENTINEL_20220115_B02.tif", "20220115", "B02"),
            ("SENTINEL_20220115_B03.tif", "20220115", "B03"),
        ]
        
        for fname, _, _ in files:
            fpath = tmp_path / fname
            with rasterio.open(fpath, 'w', **profile) as dst:
                dst.write(np.ones((1, 10, 10), dtype='uint16'))
                
        yield tmpdir

def test_build_local_cube(dummy_tif_dir):
    # Regex with named groups for 'date' and 'band'
    regex_pattern = r".*_(?P<date>\d{8})_(?P<band>B\d{2})\.tif"
    
    cube = build_local_cube(
        data_dir=dummy_tif_dir,
        regex_pattern=regex_pattern,
        date_format="%Y%m%d"
    )
    
    assert isinstance(cube, xr.DataArray)
    # Check that dimensions are (time, band, y, x)
    assert cube.dims == ('time', 'band', 'y', 'x')
    
    # Check dimension lengths
    assert len(cube.time) == 2
    assert len(cube.band) == 2
    
    # Check coordinate values
    assert list(cube.band.values) == ["B02", "B03"]
    assert cube.time.values[0] == np.datetime64('2022-01-01')
    assert cube.time.values[1] == np.datetime64('2022-01-15')
    
    # Assert dask array is used (lazily evaluated)
    assert cube.chunks is not None

    cube.close()
