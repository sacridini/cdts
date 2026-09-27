import xarray as xr
import numpy as np

def regularize_time_series(cube: xr.DataArray, freq: str = '16D', method: str = 'median') -> xr.DataArray:
    """
    Regularize a time series cube to a given frequency using median or medoid.
    
    Parameters:
    - cube: xr.DataArray with dims (time, band, y, x)
    - freq: str, the resampling frequency (e.g., '16D', '1M')
    - method: str, 'median' or 'medoid'
    """
    if method == 'median':
        return cube.resample(time=freq).median(dim='time')
    elif method == 'medoid':
        resampled = cube.resample(time=freq)
        
        def _compute_medoid(group):
            # group has dims (time, band, y, x)
            # median across time: dims (band, y, x)
            if group.sizes['time'] == 0:
                return group.isel(time=0, drop=True)
            median_val = group.median(dim='time')
            
            # distance to median: dims (time, y, x)
            dist = np.sqrt(((group - median_val) ** 2).sum(dim='band'))
            
            # find index of min distance along time
            idx = dist.argmin(dim='time')
            
            # select the medoid values using advanced indexing
            return group.isel(time=idx)
            
        # Using map to apply the function to each temporal window
        return resampled.map(_compute_medoid)
    else:
        raise ValueError(f"Unknown method {method}. Use 'median' or 'medoid'.")
