import ee
import os
import io
import zipfile
import requests
import concurrent.futures
import numpy as np
import rasterio
from rasterio.merge import merge
from typing import Optional

def _download_single_tile(image: ee.Image, roi_bounds: ee.Geometry, tile_filename: str, scale: float) -> Optional[str]:
    """
    Downloads a single tile from Earth Engine.
    """
    try:
        url = image.getDownloadURL({
            'scale': scale,
            'crs': 'EPSG:4326',
            'region': roi_bounds,
            'format': 'GEO_TIFF'
        })
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            if zipfile.is_zipfile(io.BytesIO(response.content)):
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    extracted_name = z.namelist()[0]
                    z.extract(extracted_name, os.path.dirname(tile_filename))
                    extracted_path = os.path.join(os.path.dirname(tile_filename), extracted_name)
                    if os.path.exists(tile_filename):
                        os.remove(tile_filename)
                    os.rename(extracted_path, tile_filename)
            else:
                # It's likely a direct TIF file
                with open(tile_filename, 'wb') as f:
                    f.write(response.content)
            return tile_filename
        else:
            print(f"Failed to download tile {tile_filename}: HTTP {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error on tile {tile_filename}: {e}")
    return None

def download_gee_image(image: ee.Image, roi: ee.Geometry, out_filename: str, method: str = 'direct', scale: float = 30, tile_size: float = 0.25) -> Optional[str]:
    """
    Downloads an image from GEE either directly (tiled) or via Google Drive.
    
    Args:
        image (ee.Image): The Earth Engine image to download.
        roi (ee.Geometry): Region of interest geometry.
        out_filename (str): The output file path.
        method (str): 'direct' for local tiled download, 'drive' for batch export.
        scale (float): Resolution in meters.
        tile_size (float): Grid size in degrees for tiled download.
        
    Returns:
        str or None: Task ID if 'drive', output filename if 'direct', None on failure.
    """
    if method == 'drive':
        filename_no_ext = os.path.splitext(os.path.basename(out_filename))[0]
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename_no_ext,
            folder='CDTS_Downloads',
            scale=scale,
            region=roi.bounds(),
            maxPixels=1e13
        )
        task.start()
        print(f"[{filename_no_ext}] Task sent to Google Drive (Task ID: {task.id}).")
        return task.id

    elif method == 'direct':
        print(f"Starting direct download for {out_filename}...")
        
        # 1. Get bounds
        coords = roi.bounds().coordinates().getInfo()[0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # 2. Create Grid
        tiles = []
        for lon in np.arange(min_lon, max_lon, tile_size):
            for lat in np.arange(min_lat, max_lat, tile_size):
                tile_roi = ee.Geometry.Rectangle([
                    lon, lat, 
                    min(lon + tile_size, max_lon), 
                    min(lat + tile_size, max_lat)
                ])
                tiles.append(tile_roi)
                
        print(f"Divided area into {len(tiles)} tiles. Downloading concurrently...")
        
        temp_dir = os.path.join(os.path.dirname(out_filename) or '.', "temp_tiles")
        os.makedirs(temp_dir, exist_ok=True)
        downloaded_files = []

        # 3. Concurrent downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_tile = {}
            for i, tile_geom in enumerate(tiles):
                temp_file = os.path.join(temp_dir, f"tile_{i}.tif")
                future = executor.submit(_download_single_tile, image, tile_geom, temp_file, scale)
                future_to_tile[future] = temp_file
            
            for future in concurrent.futures.as_completed(future_to_tile):
                result = future.result()
                if result:
                    downloaded_files.append(result)

        # 4. Mosaic local tiles
        if downloaded_files:
            print("Mosaicking downloaded tiles...")
            try:
                src_files_to_mosaic = [rasterio.open(fp) for fp in downloaded_files]
                mosaic, out_trans = merge(src_files_to_mosaic)
                out_meta = src_files_to_mosaic[0].meta.copy()
                
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans
                })
                
                with rasterio.open(out_filename, "w", **out_meta) as dest:
                    dest.write(mosaic)
                    
                for src in src_files_to_mosaic:
                    src.close()
                    
                # 5. Cleanup
                for fp in downloaded_files:
                    os.remove(fp)
                os.rmdir(temp_dir)
                
                print(f"Successfully downloaded and mosaicked to: {out_filename}")
                return out_filename
            except Exception as e:
                print(f"Error during mosaicking: {e}")
        else:
            print("Failed to download any tiles.")
            
        return None
    else:
        raise ValueError(f"Unknown download method: {method}. Choose 'direct' or 'drive'.")
