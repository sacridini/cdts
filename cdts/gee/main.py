import ee
import os
from typing import Optional, Union
from .auth import initialize_gee
from .harmonization import get_harmonized_collection
from .composites import create_annual_medoid
from .downloader import download_gee_image

def download_gee_timeseries(
    roi: Union[tuple, list, "ee.Geometry"], 
    start_date: str, 
    end_date: str, 
    out_dir: str, 
    method: str = 'direct',
    composite_type: str = 'annual',
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
        print(f"Extracting annual composites from {start_year} to {end_year}...")
        for year in range(start_year, end_year + 1):
            print(f"Processing year {year}...")
            img_medoid = create_annual_medoid(col, year)
            filename = os.path.join(out_dir, f"landsat_medoid_{year}.tif")
            download_gee_image(img_medoid, geom, filename, method=method)
    else:
        raise NotImplementedError(f"Composite type '{composite_type}' is not currently supported.")
        
    print("Process finished.")
