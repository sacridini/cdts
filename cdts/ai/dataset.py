import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr

class STACCubeDataset(Dataset):
    """
    PyTorch DataLoader for xarray/Dask STAC Cubes.
    Slices a large virtual datacube into spatio-temporal PyTorch batches.
    """
    def __init__(self, cube: xr.DataArray, patch_size=256, stride=256):
        self.cube = cube
        self.patch_size = patch_size
        self.stride = stride
        
        # Dask array shape: (time, bands, y, x)
        self.time_len, self.bands, self.h, self.w = cube.shape
        
        self.y_starts = list(range(0, self.h - patch_size + 1, stride))
        self.x_starts = list(range(0, self.w - patch_size + 1, stride))
        self.num_patches_y = len(self.y_starts)
        self.num_patches_x = len(self.x_starts)
        
        # Extract Julian Dates (Day of Year) for Positional Encoding
        if hasattr(cube, "time"):
            dates = cube.time.dt.dayofyear.values
            self.dates = torch.tensor(dates, dtype=torch.float32)
        else:
            self.dates = torch.zeros(self.time_len, dtype=torch.float32)
        
    def __len__(self) -> int:
        return self.num_patches_y * self.num_patches_x
        
    def __getitem__(self, idx: int) -> dict:
        y_idx = idx // self.num_patches_x
        x_idx = idx % self.num_patches_x
        
        y0 = self.y_starts[y_idx]
        x0 = self.x_starts[x_idx]
        
        # .compute() triggers Dask to download/process JUST this 256x256 patch
        patch = self.cube.isel(
            y=slice(y0, y0 + self.patch_size),
            x=slice(x0, x0 + self.patch_size)
        ).compute().values
        
        patch = np.nan_to_num(patch)
        tensor_patch = torch.tensor(patch, dtype=torch.float32)
        
        # Returns the 4D Tensor and the 1D Array of Julian Dates
        return tensor_patch, self.dates
