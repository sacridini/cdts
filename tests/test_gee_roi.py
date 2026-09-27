import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from cdts.gee.roi import parse_mgrs, parse_wrs2, resolve_roi, roi_bounds, s2_tile_utm_bounds

gpd = pytest.importorskip('geopandas')
shapely_geometry = pytest.importorskip('shapely.geometry')

RIO_BBOX = (-43.6, -23.1, -43.1, -22.6)


@pytest.mark.parametrize('tile', ['217/076', '217_076', '217-76', '217 076', '217076', ' 217/76 '])
def test_parse_wrs2_formats(tile):
    assert parse_wrs2(tile) == (217, 76)


@pytest.mark.parametrize('tile', ['abc', '217/', '300/076', '217/300', '21776', ''])
def test_parse_wrs2_rejects_invalid(tile):
    with pytest.raises(ValueError):
        parse_wrs2(tile)


def test_bbox():
    assert roi_bounds(list(RIO_BBOX)) == RIO_BBOX


@pytest.mark.parametrize('bad', [[1, 2, 3], [-43.1, -23.1, -43.6, -22.6]])
def test_bad_bbox(bad):
    with pytest.raises(ValueError):
        roi_bounds(bad)


def _rio_polygon_utm():
    # the RIO_BBOX polygon expressed in UTM 23S, to exercise reprojection
    gdf = gpd.GeoDataFrame(geometry=[shapely_geometry.box(*RIO_BBOX)], crs='EPSG:4326')
    return gdf.to_crs('EPSG:32723')


@pytest.mark.parametrize('ext, driver', [('.shp', 'ESRI Shapefile'), ('.gpkg', 'GPKG'), ('.geojson', 'GeoJSON')])
def test_vector_file_reprojected_to_lonlat(tmp_path, ext, driver):
    path = tmp_path / f'area{ext}'
    _rio_polygon_utm().to_file(path, driver=driver)
    np.testing.assert_allclose(roi_bounds(str(path)), RIO_BBOX, atol=1e-6)
    np.testing.assert_allclose(roi_bounds(path), RIO_BBOX, atol=1e-6)  # pathlib.Path too


def test_raster_file_extent(tmp_path):
    path = tmp_path / 'ref.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=100, height=50, count=1, dtype='uint8',
                       crs='EPSG:32723', transform=from_origin(650000, 7500000, 30, 30)) as ds:
        ds.write(np.zeros((1, 50, 100), 'uint8'))
    minx, miny, maxx, maxy = roi_bounds(str(path))
    assert -45 < minx < maxx < -42 and -24 < miny < maxy < -22
    # 3 km x 1.5 km at ~22.6 S
    assert (maxx - minx) == pytest.approx(3000 / 102700, rel=0.05)
    assert (maxy - miny) == pytest.approx(1500 / 110800, rel=0.05)


def test_geodataframe_and_shapely():
    np.testing.assert_allclose(roi_bounds(_rio_polygon_utm()), RIO_BBOX, atol=1e-6)
    assert roi_bounds(shapely_geometry.box(*RIO_BBOX)) == pytest.approx(RIO_BBOX)


def test_vector_without_crs_assumes_lonlat(tmp_path):
    gdf = gpd.GeoDataFrame(geometry=[shapely_geometry.box(*RIO_BBOX)])
    with pytest.warns(UserWarning, match='no CRS'):
        assert roi_bounds(gdf) == pytest.approx(RIO_BBOX)


def test_empty_vector_file(tmp_path):
    path = tmp_path / 'empty.gpkg'
    gpd.GeoDataFrame(geometry=[], crs='EPSG:4326').to_file(path, driver='GPKG')
    with pytest.raises(ValueError, match='no features'):
        roi_bounds(str(path))


def test_non_local_inputs_are_left_to_earth_engine():
    with pytest.raises(TypeError):
        roi_bounds('217/076')  # WRS-2 ids need an EE lookup
    with pytest.raises(TypeError):
        resolve_roi(42)
    with pytest.raises(ValueError, match='existing file'):
        resolve_roi('does/not/exist.shp')


# UTM bounds of real Sentinel-2 tiles; checked against the union of scene
# footprints in COPERNICUS/S2_SR_HARMONIZED (all within 51 m, the footprint
# polygons' own simplification error).
@pytest.mark.parametrize('tile, crs, bounds', [
    ('23KPQ', 'EPSG:32723', (600000, 7390200, 709800, 7500000)),   # Rio de Janeiro
    ('T31TCJ', 'EPSG:32631', (300000, 4790200, 409800, 4900000)),  # S2 'T' prefix
    ('33UUP', 'EPSG:32633', (300000, 5290200, 409800, 5400000)),
    ('19HCD', 'EPSG:32719', (300000, 6290200, 409800, 6400000)),
    ('32VNM', 'EPSG:32632', (500000, 6590200, 609800, 6700000)),
    ('22MHE', 'EPSG:32722', (800000, 9890200, 909800, 10000000)),  # touches the equator
    ('10SEG', 'EPSG:32610', (500000, 4090200, 609800, 4200000)),
    ('55hbu', 'EPSG:32755', (200000, 5790200, 309800, 5900000)),   # lower case
])
def test_s2_tile_utm_bounds(tile, crs, bounds):
    assert s2_tile_utm_bounds(tile) == (crs, bounds)


def test_s2_tile_lonlat_bounds_contain_the_city():
    min_lon, min_lat, max_lon, max_lat = roi_bounds('23KPQ')
    assert min_lon < -43.2 < max_lon and min_lat < -22.9 < max_lat  # Rio de Janeiro centre


@pytest.mark.parametrize('tile', ['23IPQ', '23KPW', '61KPQ', '0KPQ', '23KAQ', 'KPQ'])
def test_parse_mgrs_rejects_invalid(tile):
    # I/O never used; row letters stop at V; zone 1-60; 'A' is not a column in zone 23
    with pytest.raises(ValueError):
        parse_mgrs(tile)


def test_s2_tile_across_antimeridian_is_rejected():
    with pytest.raises(ValueError, match='antimeridian'):
        roi_bounds('60CWT')
