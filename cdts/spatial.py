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

from scipy.ndimage import uniform_filter

def apply_bayesian_filter(probs: "np.ndarray", window_size: int = 3) -> "np.ndarray":
    """
    Applies a Bayesian smoothing filter to class probabilities.
    Unlike a simple majority filter, this considers the confidence (probability) of the 
    AI/TWDTW model. It multiplies the local pixel probability by the neighborhood average probability.
    
    Args:
        probs: 3D numpy array [Classes, Y, X] of probabilities or confidence scores.
        window_size: Size of the spatial window.
        
    Returns:
        2D numpy array [Y, X] of the winning class indices after Bayesian smoothing.
    """
    C, Y, X = probs.shape
    smoothed_probs = np.zeros_like(probs)
    
    for c in range(C):
        smoothed_probs[c] = uniform_filter(probs[c], size=window_size)
        
    # Bayesian update: P(class|neighbor) is proportional to P(class) * P_neighbor(class)
    updated_probs = probs * smoothed_probs
    
    return np.argmax(updated_probs, axis=0)
