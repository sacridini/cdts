import xarray as xr
from pystac_client import Client
import stackstac
import geopandas as gpd

# Known public STAC catalogs
STAC_CATALOGS = {
    "earth_search": "https://earth-search.aws.element84.com/v1",
    "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "brazil_data_cube": "https://brazildatacube.dpi.inpe.br/stac/"
}

from typing import List, Optional, Union
import xarray as xr

def build_time_series(
    source: str = "earth_search", 
    collection: Union[str, List[str]] = "sentinel-2-l2a", 
    bbox: Optional[List[float]] = None, 
    vector_path: Optional[str] = None,
    start_date: str = "2020-01-01", 
    end_date: str = "2020-12-31", 
    cloud_cover_max: int = 30,
    bands: Optional[List[str]] = None,
    resolution: Optional[int] = None,
    epsg: int = 4326
) -> xr.DataArray:
    """
    Builds a lazy Dask-backed xarray DataCube from a STAC catalog.
    
    source: A string from STAC_CATALOGS or a custom STAC API URL.
    collection: The dataset collection ID (e.g., "sentinel-2-l2a", "landsat-c2-l2").
    bbox: [minx, miny, maxx, maxy] in WGS84 (EPSG:4326).
    vector_path: Path to a shapefile or geojson to derive the bounding box.
    start_date, end_date: YYYY-MM-DD strings.
    cloud_cover_max: Maximum cloud cover percentage for image filtering.
    bands: List of band names to load (e.g., ["red", "green", "blue", "nir"]).
    resolution: Target spatial resolution in meters (if reprojection is needed).
    """
    
    # 1. Resolve Bounding Box
    if vector_path is not None:
        gdf = gpd.read_file(vector_path).to_crs("EPSG:4326")
        bounds = gdf.total_bounds
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
    
    if bbox is None:
        raise ValueError("Must provide either bbox or vector_path")
        
    # 2. Connect to STAC API
    stac_url = STAC_CATALOGS.get(source, source)
    catalog = Client.open(stac_url)
    
    # 3. Search for items
    query_params = {}
    if cloud_cover_max < 100:
        # Different catalogs use different cloud cover properties
        # eo:cloud_cover is the standard
        query_params["eo:cloud_cover"] = {"lt": cloud_cover_max}
        
    search = catalog.search(
        collections=[collection] if isinstance(collection, str) else collection,
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query=query_params
    )
    
    items = search.item_collection()
    
    # 3.5 Microsoft Planetary Computer SAS Token Signing
    if "planetarycomputer" in stac_url:
        try:
            import planetary_computer
            print("Signing items with Planetary Computer SAS tokens...")
            items = [planetary_computer.sign(item) for item in items]
        except ImportError:
            raise ImportError("Please install 'planetary-computer' via pip to use Microsoft Planetary Computer.")
            
    print(f"Found {len(items)} scenes in {source} for {collection}")
    
    if len(items) == 0:
        raise ValueError("No images found for the given criteria.")
        
    # 4. Build DataCube via stackstac
    # stackstac handles the reprojection, resampling, and alignment automatically!
    cube = stackstac.stack(
        items,
        assets=bands,
        bounds_latlon=bbox,
        resolution=resolution,
        epsg=epsg,
        chunksize=512
    )
    
    return cube

