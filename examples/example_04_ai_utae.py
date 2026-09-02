"""
Example 04: U-TAE (U-Net with Temporal Attention)
"""
import os
import torch
import numpy as np
from cdts import build_time_series, save_raster
from cdts.ai import UTAE

bbox = [-55.01, -11.01, -55.00, -11.00]
cube = build_time_series(bbox=bbox, start_date="2021-01-01", end_date="2021-06-30", source="earth_search", bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22'], resolution=30, epsg=3857, cloud_cover_max=20)

print("Downloading Cube...")
cube_data = np.nan_to_num(cube.compute().values)
dates = cube.time.dt.dayofyear.values

tensor_cube = torch.tensor(cube_data, dtype=torch.float32).unsqueeze(0)
tensor_dates = torch.tensor(dates, dtype=torch.float32)

print("Initializing U-TAE...")
model = UTAE(in_channels=6, num_classes=5)

print("Running Forward Pass with Temporal Attention...")
logits = model(tensor_cube, tensor_dates)
predictions = torch.argmax(logits, dim=1).squeeze().numpy().astype(np.uint8)

out_tif = os.path.join("data", "utae_prediction.tif")
save_raster(predictions, out_tif, reference_cube=cube, nodata=255)

print(f"Done! Output saved to {out_tif}")
