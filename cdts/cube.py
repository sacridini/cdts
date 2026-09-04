import xarray as xr
from pystac_client import Client
import stackstac
import geopandas as gpd

# Known public STAC catalogs
STAC_CATALOGS = {
    "earth_search": "https://earth-search.aws.element84.com/v1",
    "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "brazil_data_cube": "https://data.inpe.br/bdc/stac/v1/"
}

from typing import List, Optional, Union
import xarray as xr

import concurrent.futures
import rasterio

def _validate_stac_item(item, band):
    """Helper function to test if a STAC item URL is physically readable."""
    try:
        url = item.assets[band].href
        with rasterio.open(url) as src:
            pass # Just opening is enough to test if header exists and is valid
        return item
    except Exception:
        return None

def build_time_series(
    source: str = "earth_search", 
    collection: Union[str, List[str]] = "sentinel-2-l2a", 
    bbox: Optional[List[float]] = None, 
    vector_path: Optional[str] = None,
    start_date: str = "2020-01-01", 
    end_date: str = "2020-12-31", 
    cloud_cover_max: int = 30,
    bands: Optional[List[str]] = None,
    resolution: Optional[float] = None,
    epsg: int = 4326,
    validate_items: bool = False,
    access_token: Optional[str] = None
) -> xr.DataArray:
    """
    Builds a lazy Dask-backed xarray DataCube from a STAC catalog.
    
    source: A string from STAC_CATALOGS or a custom STAC API URL.
    collection: The dataset collection ID (e.g., "sentinel-2-l2a", "CBERS4A_WFI_L4_SR").
    bbox: [minx, miny, maxx, maxy] in WGS84 (EPSG:4326).
    vector_path: Path to a shapefile or geojson to derive the bounding box.
    start_date, end_date: YYYY-MM-DD strings.
    cloud_cover_max: Maximum cloud cover percentage for image filtering.
    bands: List of band names to load (e.g., ["red", "green", "blue", "nir"]).
    resolution: Target spatial resolution in meters (if reprojection is needed).
    validate_items: If True, tests each STAC item's URL before stacking to drop corrupted files.
    access_token: API token for restricted catalogs like Brazil Data Cube (BDC).
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
        
    items_list = list(items)
    
    # 3.7 BDC Token injection
    if access_token:
        print("Injecting access token into asset URLs...")
        for item in items_list:
            for asset_key in item.assets:
                asset = item.assets[asset_key]
                if "?" in asset.href:
                    asset.href = f"{asset.href}&access_token={access_token}"
                else:
                    asset.href = f"{asset.href}?access_token={access_token}"
    
    # 3.6 Pre-flight Validation
    if validate_items and bands:
        print(f"Iniciando validação de {len(items_list)} cenas para bloquear arquivos corrompidos...")
        valid_items = []
        band_to_check = bands[0]  # Just check the first band as a proxy for the whole scene
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_validate_stac_item, item, band_to_check): item for item in items_list}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    valid_items.append(res)
                    
        removed = len(items_list) - len(valid_items)
        print(f"Cenas íntegras aprovadas: {len(valid_items)} (Removidas/Corrompidas: {removed})")
        items_list = valid_items
        
        if len(items_list) == 0:
            raise ValueError("All scenes were invalid or corrupted after validation.")
        
    # 4. Build DataCube via stackstac
    # stackstac handles the reprojection, resampling, and alignment automatically!
    cube = stackstac.stack(
        items_list,
        assets=bands,
        bounds_latlon=bbox,
        resolution=resolution,
        epsg=epsg,
        chunksize=512,
        errors_as_nodata=(Exception,)
    )
    
    return cube

