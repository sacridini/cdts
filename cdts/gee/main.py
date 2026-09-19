import ee
import os
import concurrent.futures
from typing import Optional, Union
from .auth import initialize_gee
from .harmonization import get_harmonized_collection
from .composites import create_annual_medoid
from .downloader import download_gee_image
from .drive_sync import submit_drive_export, wait_and_download_task

def _compute_indices(img: "ee.Image", bands: list) -> "ee.Image":
    img_bands = img
    ndvi_img = img.normalizedDifference(['SR_B5', 'SR_B4'])

    if 'NDVI' in bands:
        img_bands = img_bands.addBands(ndvi_img.rename('NDVI').toFloat())
    if 'NBR' in bands:
        nbr = img.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR').toFloat()
        img_bands = img_bands.addBands(nbr)
    if 'NDWI' in bands:
        ndwi = img.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI').toFloat()
        img_bands = img_bands.addBands(ndwi)
    if 'kNDVI' in bands:
        kndvi = ndvi_img.pow(2).tanh().rename('kNDVI').toFloat()
        img_bands = img_bands.addBands(kndvi)
    if 'EVI' in bands:
        evi = img.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': img.select('SR_B5'),
                'RED': img.select('SR_B4'),
                'BLUE': img.select('SR_B2')
            }).rename('EVI').toFloat()
        img_bands = img_bands.addBands(evi)
    return img_bands.select(bands)


def download_gee_timeseries_drive(
    roi: Union[tuple, list, "ee.Geometry"],
    start_date: str,
    end_date: str,
    out_dir: str,
    tile_label: str,
    bands: Optional[list] = None,
    project: Optional[str] = None,
    drive_folder: str = "cdts_exports",
    max_concurrent_tasks: int = 10,
    delete_after: bool = True,
    poll_interval: int = 15,
) -> list:
    """
    Downloads an annual-medoid time series via GEE batch Export.image.toDrive
    instead of the synchronous getDownloadURL tiling in download_gee_timeseries.
    Submits one export task per year up front (so GEE computes them in
    parallel server-side), then polls/downloads finished files with bounded
    local concurrency -- avoiding the per-request compute overhead that makes
    method='direct' slow for a full multi-decade time series.

    Args:
        roi: Region of interest (bbox tuple or ee.Geometry).
        start_date, end_date (str): YYYY-MM-DD.
        out_dir (str): Local directory to save the downloaded GeoTIFFs.
        tile_label (str): Identifier (e.g. "214_064") used to prefix filenames
            and the Drive export description, so concurrent tiles don't collide.
        bands (list, optional): Index bands to compute (e.g. ["NDVI"]). If
            None, keeps the raw harmonized SR_B2-SR_B7 bands.
        project (str, optional): Google Cloud Project ID.
        drive_folder (str): Google Drive folder name used for every export
            in this batch (created automatically by GEE on first export).
        max_concurrent_tasks (int): How many years to have in flight
            (submitted-but-not-yet-downloaded) at once.
        delete_after (bool): Delete each file from Drive once downloaded
            locally, so a multi-tile batch doesn't fill up Drive quota.
        poll_interval (int): Seconds between task-status checks.

    Returns:
        list[str]: Local file paths successfully downloaded.
    """
    initialize_gee(project=project)

    geom = ee.Geometry.Rectangle(roi) if isinstance(roi, (tuple, list)) else roi
    os.makedirs(out_dir, exist_ok=True)

    # Medoid selection runs on the raw harmonized spectral bands (matching
    # LT-GEE's own medoidMosaic: squared distance to the annual per-band
    # median summed across all SR bands); indices are derived only from the
    # single per-year composite pixel that selection picks. Computing an
    # index first would collapse medoid selection to 1D distance-to-median
    # in index space -- a different, non-standard criterion for which
    # candidate scene wins a given year.
    col = get_harmonized_collection(geom, start_date, end_date)

    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])
    years = list(range(start_year, end_year + 1))

    downloaded = []

    def _submit(year):
        medoid_raw = create_annual_medoid(col, year)
        img_medoid = _compute_indices(medoid_raw, bands) if bands else medoid_raw
        prefix = f"{tile_label}_{year}"
        task = submit_drive_export(img_medoid, prefix, drive_folder, geom.bounds())
        return year, prefix, task

    def _wait(year_prefix_task):
        year, prefix, task = year_prefix_task
        out_path = os.path.join(out_dir, f"{prefix}.tif")
        print(f"[{tile_label}] Waiting on year {year} (task {task.id})...")
        result = wait_and_download_task(
            task, drive_folder, prefix, out_path,
            poll_interval=poll_interval, delete_after=delete_after,
        )
        if result:
            print(f"[{tile_label}] Downloaded year {year} -> {result}")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        submitted = [_submit(y) for y in years]
        for result in executor.map(_wait, submitted):
            if result:
                downloaded.append(result)

    print(f"[{tile_label}] Finished: {len(downloaded)}/{len(years)} years downloaded.")
    return downloaded


def download_gee_timeseries(
    roi: Union[tuple, list, "ee.Geometry"], 
    start_date: str, 
    end_date: str, 
    out_dir: str, 
    method: str = 'direct',
    composite_type: str = 'annual',
    bands: Optional[list] = None,
    project: Optional[str] = None
) -> None:
    """
    Downloads time series data from Google Earth Engine.
    
    Args:
        roi (tuple, list, or ee.Geometry): Region of interest (min_lon, min_lat, max_lon, max_lat) or ee.Geometry.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        out_dir (str): Output directory to save the files.
        method (str): Download method ('direct' for local tiling, 'drive' for GDrive export).
        composite_type (str): Type of composition ('annual' for LandTrendr Medoid, etc.).
        project (str, optional): Google Cloud Project ID for authentication.
    """
    initialize_gee(project=project)
    
    # Handle ROI parsing
    if isinstance(roi, (tuple, list)):
        if len(roi) == 4:
            geom = ee.Geometry.Rectangle(roi)
        else:
            raise ValueError("ROI tuple/list must contain 4 elements (min_lon, min_lat, max_lon, max_lat)")
    else:
        geom = roi
        
    os.makedirs(out_dir, exist_ok=True)
    
    print("Preparing harmonized collection...")
    col = get_harmonized_collection(geom, start_date, end_date)

    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])

    if composite_type == 'annual':
        # Medoid selection runs on the raw harmonized spectral bands (matching
        # LT-GEE's own medoidMosaic: squared distance to the annual per-band
        # median summed across all SR bands); indices are derived only from
        # the single per-year composite pixel that selection picks -- computing
        # an index first would collapse medoid selection to 1D distance in
        # index space, a different (non-standard) criterion for which
        # candidate scene wins a given year.
        print(f"Extracting annual composites from {start_year} to {end_year}...")
        for year in range(start_year, end_year + 1):
            print(f"Processing year {year}...")
            medoid_raw = create_annual_medoid(col, year)
            img_medoid = _compute_indices(medoid_raw, bands) if bands else medoid_raw
            filename = os.path.join(out_dir, f"landsat_medoid_{year}.tif")
            download_gee_image(img_medoid, geom, filename, method=method)
    elif composite_type == 'dense':
        # Dense per-observation stack, not a medoid composite -- indices are
        # computed per image before flattening, same as any other band.
        if bands:
            col = col.map(lambda img: _compute_indices(img, bands))
        print("Extracting dense time series dates...")

        # Get dates in milliseconds from GEE
        dates_ms = col.aggregate_array('system:time_start').getInfo()
        
        if not dates_ms:
            print("No images found in the given date range.")
            return
            
        import pandas as pd
        from datetime import datetime
        
        dates_list = []
        ordinal_dates = []
        
        for ms in dates_ms:
            dt = datetime.utcfromtimestamp(ms / 1000.0)
            dates_list.append(dt.strftime('%Y-%m-%d'))
            ordinal_dates.append(dt.toordinal())
            
        # Save to CSV
        csv_path = os.path.join(out_dir, "ccdc_dates.csv")
        df = pd.DataFrame({
            'Date': dates_list,
            'Ordinal_Day': ordinal_dates
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved dates for CCDC to: {csv_path}")
        
        print("Flattening collection for dense stack download...")
        # Convert the ImageCollection to a single multi-band Image
        # The bands will be ordered chronologically, matching the dates array
        dense_image = col.toBands()
        
        filename = os.path.join(out_dir, "landsat_dense_stack.tif")
        download_gee_image(dense_image, geom, filename, method=method)
    else:
        raise NotImplementedError(f"Composite type '{composite_type}' is not currently supported.")
        
    print("Process finished.")
