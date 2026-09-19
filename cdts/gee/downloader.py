import ee
import os
import io
import time
import uuid
import zipfile
import requests
import concurrent.futures
import numpy as np
import rasterio
from rasterio.merge import merge
from typing import Optional
from .drive_sync import submit_drive_export, wait_and_download_task

def _download_single_tile(image: ee.Image, roi_bounds: ee.Geometry, tile_filename: str, scale: float,
                           max_retries: int = 5, base_backoff: float = 5.0) -> Optional[str]:
    """
    Downloads a single tile from Earth Engine, retrying with exponential
    backoff on transient failures (HTTP 429 concurrency-limit rejections
    in particular -- see download_gee_image's docstring). Returns None
    only after every retry is exhausted, so callers can tell a genuine
    failure apart from a transient one that self-healed.
    """
    last_error = None
    for attempt in range(max_retries):
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
                    with open(tile_filename, 'wb') as f:
                        f.write(response.content)
                return tile_filename

            last_error = f"HTTP {response.status_code}: {response.text}"
            if response.status_code == 429 or response.status_code >= 500:
                sleep_s = base_backoff * (2 ** attempt)
                print(f"[retry {attempt + 1}/{max_retries}] {tile_filename}: {last_error} -- "
                      f"backing off {sleep_s:.0f}s")
                time.sleep(sleep_s)
                continue
            else:
                print(f"Failed to download tile {tile_filename}: {last_error}")
                return None
        except Exception as e:
            last_error = str(e)
            sleep_s = base_backoff * (2 ** attempt)
            print(f"[retry {attempt + 1}/{max_retries}] {tile_filename}: {last_error} -- "
                  f"backing off {sleep_s:.0f}s")
            time.sleep(sleep_s)

    print(f"Giving up on tile {tile_filename} after {max_retries} attempts: {last_error}")
    return None

def download_gee_image(image: ee.Image, roi: ee.Geometry, out_filename: str, method: str = 'direct',
                        scale: float = 30, tile_size: float = 0.25, sub_tile_workers: int = 4) -> Optional[str]:
    """
    Downloads an image from GEE either directly (tiled) or via Google Drive.

    Args:
        image (ee.Image): The Earth Engine image to download.
        roi (ee.Geometry): Region of interest geometry.
        out_filename (str): The output file path.
        method (str): 'direct' for local tiled download, 'drive' for batch export.
        scale (float): Resolution in meters.
        tile_size (float): Grid size in degrees for tiled download.
        sub_tile_workers (int): Concurrent getDownloadURL requests for this one
            image. Earth Engine enforces a per-account limit on concurrent
            *interactive* requests (distinct from the batch/Export queue) --
            see https://developers.google.com/earth-engine/guides/usage#concurrent_interactive_requests.
            Exceeding it makes some sub-tile requests fail outright (HTTP 429);
            _download_single_tile retries those with backoff, but if a caller
            also parallelizes ACROSS tiles (e.g. one process per tile), the
            *combined* concurrency across all of them counts against the same
            account-wide limit -- keep sub_tile_workers x (number of tiles
            downloading at once) comfortably under that ceiling.

    Returns:
        str or None: Output filename on success. Returns None (and writes
        nothing to out_filename) if even one sub-tile could not be fetched
        after retries -- a partial mosaic is never silently written, since a
        caller relying on file-existence for resumability must never see a
        gappy composite reported as complete.
    """
    if method == 'drive':
        filename_no_ext = os.path.splitext(os.path.basename(out_filename))[0]

        # Use the base name of the output directory as the Drive folder name
        out_dir = os.path.dirname(out_filename)
        drive_folder = os.path.basename(out_dir) if out_dir and os.path.basename(out_dir) else 'CDTS_Downloads'

        task = submit_drive_export(image, filename_no_ext, drive_folder, roi.bounds(), scale=scale)
        print(f"[{filename_no_ext}] Task sent to Google Drive (Task ID: {task.id}). Waiting for completion...")
        return wait_and_download_task(task, drive_folder, filename_no_ext, out_filename)

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

        print(f"Divided area into {len(tiles)} tiles. Downloading with {sub_tile_workers} concurrent workers...")

        # Unique temp dir per call so two processes downloading the same
        # output tile at once (e.g. a resumed run racing a still-live one)
        # can't collide on each other's partial files.
        temp_dir = os.path.join(os.path.dirname(out_filename) or '.', f"temp_tiles_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_dir, exist_ok=True)
        downloaded_files = []

        # 3. Concurrent downloads (bounded -- see sub_tile_workers docstring above)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=sub_tile_workers) as executor:
                future_to_tile = {}
                for i, tile_geom in enumerate(tiles):
                    temp_file = os.path.join(temp_dir, f"tile_{i}.tif")
                    future = executor.submit(_download_single_tile, image, tile_geom, temp_file, scale)
                    future_to_tile[future] = temp_file

                for future in concurrent.futures.as_completed(future_to_tile):
                    result = future.result()
                    if result:
                        downloaded_files.append(result)

            # 4. Refuse to mosaic (and refuse to write out_filename at all) unless
            # every sub-tile succeeded -- a partial mosaic reported as a success
            # is worse than a missing file, since resumability logic can't tell
            # the difference between "done" and "silently incomplete".
            if len(downloaded_files) != len(tiles):
                print(f"Only {len(downloaded_files)}/{len(tiles)} sub-tiles succeeded for {out_filename} "
                      f"-- refusing to write a partial mosaic. Not writing output; caller should retry.")
                return None

            print("Mosaicking downloaded tiles...")
            src_files_to_mosaic = [rasterio.open(fp) for fp in downloaded_files]
            try:
                mosaic, out_trans = merge(src_files_to_mosaic)
                out_meta = src_files_to_mosaic[0].meta.copy()

                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw",
                    "predictor": 3,
                })

                with rasterio.open(out_filename, "w", **out_meta) as dest:
                    dest.write(mosaic)
            finally:
                for src in src_files_to_mosaic:
                    src.close()

            print(f"Successfully downloaded and mosaicked to: {out_filename}")
            return out_filename
        except Exception as e:
            print(f"Error during direct download/mosaicking of {out_filename}: {e}")
            return None
        finally:
            # 5. Cleanup temp files regardless of success/failure
            for fp in downloaded_files:
                try:
                    os.remove(fp)
                except OSError:
                    pass
            try:
                for leftover in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, leftover))
                os.rmdir(temp_dir)
            except OSError:
                pass
    else:
        raise ValueError(f"Unknown download method: {method}. Choose 'direct' or 'drive'.")
