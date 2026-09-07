import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import xarray as xr


def build_local_cube(data_dir: str, regex_pattern: str, date_format: str = "%Y%m%d") -> xr.DataArray:
    """
    Build a local data cube from a directory of TIFF files, lazily loaded via dask and rioxarray.
    Replicates local ARD ingestion functionality.
    """
    data_path = Path(data_dir)
    pattern = re.compile(regex_pattern)
    
    file_info = []
    
    # Scan data_dir for .tif files
    for file_path in data_path.rglob("*.tif"):
        match = pattern.search(file_path.name)
        if match:
            match_dict = match.groupdict()
            if 'date' in match_dict and 'band' in match_dict:
                parsed_date = datetime.strptime(match_dict['date'], date_format)
                file_info.append({
                    'path': file_path,
                    'time': parsed_date,
                    'band': match_dict['band']
                })
                
    if not file_info:
        raise ValueError(f"No files matching the pattern '{regex_pattern}' were found in '{data_dir}'")
        
    df = pd.DataFrame(file_info)
    df = df.sort_values(by=['time', 'band'])
    
    time_arrays = []
    times = []
    
    # Group by time to build the bands
    for time_val, time_group in df.groupby('time'):
        band_arrays = []
        bands = []
        
        for _, row in time_group.iterrows():
            # Lazily open file with rioxarray (engine='rasterio') and dask (chunks='auto')
            da = xr.open_dataarray(row['path'], engine='rasterio', chunks='auto')
            
            # rioxarray loads a 'band' dimension of size 1 by default, squeeze it out
            if 'band' in da.dims:
                da = da.squeeze('band').drop_vars('band', errors='ignore')
                
            band_arrays.append(da)
            bands.append(row['band'])
            
        # Concatenate bands for the current timestep
        da_time = xr.concat(band_arrays, dim=pd.Index(bands, name='band'))
        time_arrays.append(da_time)
        times.append(time_val)
        
    # Concatenate all timesteps
    cube = xr.concat(time_arrays, dim=pd.Index(times, name='time'))
    
    # Sort by time to guarantee the final array is time-ordered
    cube = cube.sortby('time')
    
    return cube
