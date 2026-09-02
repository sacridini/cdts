from .landtrendr import run_landtrendr, desawtooth, apply_vertices
from .raster import run_landtrendr_array, run_landtrendr_image, run_ccdc_array, run_ccdc_image
from .metrics import extract_events
from .ccdc import predict_synthetic_image
from .classify import train_ccdc_classifier, classify_ccdc_stack
from .spatial import apply_mmu_filter, apply_majority_filter
from .smooth import apply_savgol_filter
from .masks import extract_water_mask
from .tmask import run_tmask_pixel, apply_tmask_stack
from .cube import build_time_series
from .io import save_raster, get_georef
import cdts.xarray_api # This registers the xarray accessor automatically
import cdts.ai

__all__ = [
    "run_landtrendr", "desawtooth", "apply_vertices", 
    "run_landtrendr_array", "run_landtrendr_image", 
    "run_ccdc_array", "run_ccdc_image", 
    "extract_events", "predict_synthetic_image",
    "train_ccdc_classifier", "classify_ccdc_stack",
    "apply_mmu_filter", "apply_majority_filter", "apply_savgol_filter", 
    "extract_water_mask",
    "run_tmask_pixel", "apply_tmask_stack",
    "build_time_series",
    "save_raster",
    "ai"
]
