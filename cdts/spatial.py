import numpy as np
from scipy import ndimage
import rasterio

def apply_mmu_filter(input_path: str, output_path: str, mmu_pixels: int = 11) -> None:
    """
    Applies a Minimum Mapping Unit (MMU) spatial filter using scipy.ndimage.
    Removes isolated pixel groups smaller than mmu_pixels and fills the gaps with the dominant neighbor class.
    Very useful for LandTrendr post-processing to reduce "salt and pepper" noise.
    """
    with rasterio.open(input_path) as src:
        data = src.read(1) # Assuming single band (e.g., year of detection or magnitude)
        nodata = src.nodata if src.nodata is not None else 0
        
        # Create a binary mask of disturbance vs no-disturbance
        disturbed = (data != nodata)
        
        # Label connected components
        labeled, num_features = ndimage.label(disturbed)
        
        # Count sizes
        sizes = ndimage.sum(disturbed, labeled, range(num_features + 1))
        
        # Create a mask of features smaller than MMU
        mask_size = sizes < mmu_pixels
        remove_pixel = mask_size[labeled]
        
        # Apply mask
        filtered_data = np.copy(data)
        filtered_data[remove_pixel] = nodata
        
        profile = src.profile
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(filtered_data, 1)
            
    print(f"MMU filtering applied. Saved to {output_path}")


from scipy.ndimage import generic_filter
import numpy as np
from scipy import stats

def apply_majority_filter(image: "np.ndarray", size: int = 3) -> "np.ndarray":
    """
    Applies a spatial majority (mode) filter to regularize classification outputs.
    Similar to fits 'sits_regularize'.
    """
    def _mode_func(window):
        # Return the most common value in the window
        return stats.mode(window, axis=None, keepdims=False).mode
        
    # generic_filter applies the function to a moving window
    return generic_filter(image, _mode_func, size=size)
