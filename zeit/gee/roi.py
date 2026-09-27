"""Region-of-interest inputs for the GEE download functions.

Users describe the area with local, offline inputs -- a Landsat WRS-2 tile id,
a Sentinel-2 (MGRS) tile id, a lon/lat bounding box, a vector file, a raster file, or an in-memory
GeoDataFrame / shapely geometry -- and only resolve_roi turns that into the
ee.Geometry Earth Engine needs.
"""
import os
import re
import warnings
from typing import Tuple

import ee

# Collections searched (in order) for a scene on the requested path/row. All
# Landsat 4-9 missions share the WRS-2 grid, so any one scene gives the
# tile's footprint; L8 comes first since it covers every land tile.
_FOOTPRINT_SOURCES = (
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LE07/C02/T1_L2",
    "LANDSAT/LT05/C02/T1_L2",
)

# path, optional separator, row -- '217076' splits as 217/076 (row always
# given with 3 digits when there is no separator)
_WRS_PATTERN = re.compile(r"^\s*(\d{1,3})\s*(?:[/_\-\s]\s*(\d{1,3})|(\d{3}))\s*$")

_RASTER_EXTENSIONS = {'.tif', '.tiff', '.vrt', '.img', '.jp2', '.nc'}

# Sentinel-2 tile id: UTM zone, latitude band, 100 km square column and row
# letters (I and O are never used), with the optional 'T' prefix of S2 names.
_MGRS_PATTERN = re.compile(r"^\s*T?(\d{1,2})([C-HJ-NP-X])([A-HJ-NP-Z])([A-HJ-NP-V])\s*$", re.IGNORECASE)
_MGRS_COLUMNS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")  # by (zone - 1) % 3
_MGRS_ROWS = "ABCDEFGHJKLMNPQRSTUV"                   # 20 letters, cycling every 2,000 km
_MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"                  # 8 deg bands from 80 S (X spans 72-84 N)
_S2_TILE_SIZE = 109800                                # m; S2 tiles overlap their neighbours by 9.8 km


def parse_wrs2(tile: str) -> Tuple[int, int]:
    """Parses a WRS-2 tile id like '217/076', '217_076', '217-76' or '217076'
    into (path, row)."""
    m = _WRS_PATTERN.match(tile)
    if not m:
        raise ValueError(f"Not a WRS-2 path/row or an existing file: {tile!r}. Use e.g. '217/076'.")
    path, row = int(m.group(1)), int(m.group(2) or m.group(3))
    if not (1 <= path <= 233 and 1 <= row <= 248):
        raise ValueError(f"WRS-2 path must be 1-233 and row 1-248, got {path}/{row}.")
    return path, row


def wrs2_footprint(path: int, row: int) -> ee.Geometry:
    """Footprint of a Landsat WRS-2 tile, taken from a Collection 2 scene on
    that path/row. Raises ValueError if no scene exists for it."""
    for asset in _FOOTPRINT_SOURCES:
        scenes = (ee.ImageCollection(asset)
                  .filter(ee.Filter.eq("WRS_PATH", path))
                  .filter(ee.Filter.eq("WRS_ROW", row)))
        if scenes.limit(1).size().getInfo():
            return ee.Geometry(scenes.first().geometry())
    raise ValueError(f"No Landsat Collection 2 scene found for WRS-2 path {path}, row {row}.")


def parse_mgrs(tile: str) -> Tuple[int, str, str, str]:
    """Parses a Sentinel-2 / MGRS tile id like '23KPQ' or 'T23KPQ' into
    (zone, band, column letter, row letter)."""
    m = _MGRS_PATTERN.match(tile)
    if not m or not 1 <= int(m.group(1)) <= 60:
        raise ValueError(f"Not a Sentinel-2 (MGRS) tile id: {tile!r}. Use e.g. '23KPQ'.")
    zone, band, col, row = int(m.group(1)), *(g.upper() for g in m.groups()[1:])
    if col not in _MGRS_COLUMNS[(zone - 1) % 3]:
        raise ValueError(f"Column letter {col!r} does not exist in UTM zone {zone} ({tile!r}).")
    return zone, band, col, row


def s2_tile_utm_bounds(tile: str) -> Tuple[str, Tuple[float, float, float, float]]:
    """(UTM CRS, (minx, miny, maxx, maxy)) of a Sentinel-2 tile, computed
    offline from its MGRS id. A tile starts at the upper-left corner of its
    100 km MGRS square and extends 109.8 km east and south."""
    from rasterio.warp import transform

    zone, band, col, row = parse_mgrs(tile)
    south = band < 'N'
    crs = f"EPSG:{(32700 if south else 32600) + zone}"
    easting = (_MGRS_COLUMNS[(zone - 1) % 3].index(col) + 1) * 100000

    # The row letter fixes the northing modulo 2,000 km (letters shift by 5
    # in even zones); the latitude band picks which 2,000 km cycle.
    row_offset = (_MGRS_ROWS.index(row) - (5 if zone % 2 == 0 else 0)) % 20
    band_south = -80 + 8 * _MGRS_BANDS.index(band)
    band_north = 84 if band == 'X' else band_south + 8
    central_lon = -183 + 6 * zone
    _, (n_lo, n_hi) = transform('EPSG:4326', crs, [central_lon] * 2, [band_south, band_north])
    northing = row_offset * 100000
    while northing + 100000 <= n_lo - 100000:
        northing += 2000000
    if northing > n_hi + 100000:
        raise ValueError(f"MGRS square {tile!r} does not intersect latitude band {band}.")
    top = northing + 100000
    return crs, (easting, top - _S2_TILE_SIZE, easting + _S2_TILE_SIZE, top)


def s2_tile_bounds(tile: str) -> Tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) of a Sentinel-2 tile, offline."""
    from rasterio.warp import transform_bounds

    crs, bounds = s2_tile_utm_bounds(tile)
    west, south, east, north = transform_bounds(crs, 'EPSG:4326', *bounds, densify_pts=21)
    if west > east:
        raise ValueError(f"Sentinel-2 tile {tile!r} crosses the antimeridian, which a lon/lat "
                         "bounding box can't represent; pass a bbox for one side instead.")
    return west, south, east, north


def file_bounds(path: str) -> Tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) of a local raster or vector file,
    reprojected to EPSG:4326. Rasters are recognised by extension; anything
    else is read as a vector (Shapefile, GeoPackage, GeoJSON, KML, ...)."""
    if os.path.splitext(path)[1].lower() in _RASTER_EXTENSIONS:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"Raster {path!r} has no CRS; can't place it on the map.")
            return tuple(transform_bounds(src.crs, 'EPSG:4326', *src.bounds, densify_pts=21))

    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Vector file {path!r} has no features.")
    return _gdf_bounds(gdf, path)


def _gdf_bounds(gdf, label: str) -> Tuple[float, float, float, float]:
    if gdf.crs is None:
        warnings.warn(f"{label} has no CRS; assuming EPSG:4326 (lon/lat).", stacklevel=3)
    elif not gdf.crs.equals('EPSG:4326'):
        gdf = gdf.to_crs('EPSG:4326')
    return tuple(float(v) for v in gdf.total_bounds)


def roi_bounds(roi) -> Tuple[float, float, float, float]:
    """Offline part of resolve_roi: the lon/lat bbox of any local ROI input
    (Sentinel-2 tile id, bbox, vector/raster file path, GeoDataFrame/GeoSeries,
    shapely geometry).
    Raises TypeError for inputs that need Earth Engine (WRS-2 ids, ee objects)."""
    if isinstance(roi, (str, os.PathLike)) and os.path.isfile(roi):
        return file_bounds(os.fspath(roi))
    if isinstance(roi, str) and _MGRS_PATTERN.match(roi):
        return s2_tile_bounds(roi)
    if isinstance(roi, (tuple, list)):
        if len(roi) != 4:
            raise ValueError("ROI tuple/list must contain 4 elements (min_lon, min_lat, max_lon, max_lat)")
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in roi)
        if not (min_lon < max_lon and min_lat < max_lat):
            raise ValueError(f"Bounding box must be (min_lon, min_lat, max_lon, max_lat), got {tuple(roi)}.")
        return min_lon, min_lat, max_lon, max_lat
    if hasattr(roi, 'total_bounds') and hasattr(roi, 'crs'):  # GeoDataFrame / GeoSeries
        return _gdf_bounds(roi, 'GeoDataFrame')
    if hasattr(roi, 'bounds') and hasattr(roi, '__geo_interface__'):  # shapely geometry, assumed lon/lat
        return tuple(float(v) for v in roi.bounds)
    raise TypeError(f"Not a local ROI input: {type(roi).__name__}")


def resolve_roi(roi) -> ee.Geometry:
    """ROI as an ee.Geometry from any of:

    - a Landsat WRS-2 tile id: '217/076' (the tile's footprint, looked up
      from a Landsat scene);
    - a Sentinel-2 (MGRS) tile id: '23KPQ' or 'T23KPQ' (computed offline);
    - a (min_lon, min_lat, max_lon, max_lat) bounding box;
    - a path to a local vector file (Shapefile, GeoPackage, GeoJSON, KML...)
      or raster file (GeoTIFF...): its extent, reprojected to lon/lat;
    - a GeoDataFrame / GeoSeries (its extent) or a lon/lat shapely geometry;
    - an ee.Geometry, passed through.

    Local inputs are reduced to their bounding box -- the download covers
    that box anyway, and a box keeps the request small no matter how
    detailed the source polygon is."""
    if isinstance(roi, ee.Geometry):
        return roi
    try:
        return ee.Geometry.Rectangle(list(roi_bounds(roi)))
    except TypeError:
        pass
    if isinstance(roi, str):
        return wrs2_footprint(*parse_wrs2(roi))
    raise TypeError(f"Unsupported ROI type: {type(roi).__name__}. Use a WRS-2 tile id ('217/076'), "
                    "a Sentinel-2 tile id ('23KPQ'), "
                    "a (min_lon, min_lat, max_lon, max_lat) bbox, a vector/raster file path, "
                    "a GeoDataFrame, a shapely geometry or an ee.Geometry.")
