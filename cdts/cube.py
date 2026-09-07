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
    tiles: Optional[List[str]] = None,
    start_date: str = "2020-01-01", 
    end_date: str = "2020-12-31", 
    cloud_cover_max: int = 30,
    bands: Optional[List[str]] = None,
    apply_cloud_mask: bool = False,
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
    tiles: List of specific MGRS/WRS tiles to fetch (e.g., ["20LKP"]).
    start_date, end_date: YYYY-MM-DD strings.
    cloud_cover_max: Maximum cloud cover percentage for image filtering.
    bands: List of band names to load (e.g., ["red", "green", "blue", "nir"]).
    apply_cloud_mask: Automatically identify platform and mask out clouds (requires QA band).
    resolution: Target spatial resolution in meters (if reprojection is needed).
    validate_items: If True, tests each STAC item's URL before stacking to drop corrupted files.
    access_token: API token for restricted catalogs like Brazil Data Cube (BDC).
    """
    
    # 1. Resolve Spatial Boundary
    if vector_path is not None:
        gdf = gpd.read_file(vector_path).to_crs("EPSG:4326")
        bounds = gdf.total_bounds
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
    
    if bbox is None and tiles is None:
        raise ValueError("Must provide either bbox, vector_path, or tiles")
        
    # 2. Connect to STAC API
    stac_url = STAC_CATALOGS.get(source, source)
    catalog = Client.open(stac_url)
    
    # 3. Search for items
    query_params = {}
    if cloud_cover_max < 100:
        query_params["eo:cloud_cover"] = {"lt": cloud_cover_max}
        
    if tiles:
        # Determine the correct STAC property based on collection name heuristically
        col_name = collection[0].lower() if isinstance(collection, list) else collection.lower()
        if "sentinel" in col_name or "s2" in col_name:
            query_params["s2:mgrs_tile"] = {"in": tiles}
        elif "landsat" in col_name:
            # Note: Landsat uses wrs_path and wrs_row, if 'tiles' represents pathrow like "215065"
            pass
            
    search_kwargs = {
        "collections": [collection] if isinstance(collection, str) else collection,
        "datetime": f"{start_date}/{end_date}",
        "query": query_params
    }
    
    if bbox is not None:
        search_kwargs["bbox"] = bbox
        
    search = catalog.search(**search_kwargs)
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
    
    # 3.8 Add QA band if apply_cloud_mask is requested but not in bands
    col_name = collection[0].lower() if isinstance(collection, list) else collection.lower()
    qa_band = None
    if apply_cloud_mask:
        if "sentinel" in col_name or "s2" in col_name:
            qa_band = "scl"
        elif "landsat" in col_name or "l8" in col_name:
            qa_band = "qa_pixel"
            
        if qa_band and bands and qa_band not in bands:
            bands.append(qa_band)
            
    # 3.6 Pre-flight Validation
    if validate_items and bands:
        print(f"Iniciando validação de {len(items_list)} cenas para bloquear arquivos corrompidos...")
        valid_items = []
        band_to_check = bands[0]
        
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
    stack_kwargs = {
        "assets": bands,
        "resolution": resolution,
        "epsg": epsg,
        "chunksize": 512,
        "errors_as_nodata": (Exception,)
    }
    if bbox is not None:
        stack_kwargs["bounds_latlon"] = bbox
        
    cube = stackstac.stack(items_list, **stack_kwargs)
    
    # 5. Apply Cloud Mask semantics natively
    if apply_cloud_mask and qa_band and qa_band in cube.band.values:
        qa = cube.sel(band=qa_band)
        
        if qa_band == "scl":
            # Sentinel-2 SCL: 4 (vegetation), 5 (bare soils), 6 (water), 7 (unclassified), 11 (snow)
            # Masking out clouds (8, 9, 10) and shadows (3)
            valid_mask = qa.isin([4, 5, 6, 7, 11])
            cube = cube.where(valid_mask)
        elif qa_band == "qa_pixel":
            # Landsat QA_PIXEL: simplified clear bits (bit 6 indicates clear)
            # using bitwise AND operation in xarray
            valid_mask = (qa.astype(int) & (1 << 6)) > 0
            cube = cube.where(valid_mask)
            
        print(f"Applied semantic cloud mask using QA band '{qa_band}'")
    
    return cube
