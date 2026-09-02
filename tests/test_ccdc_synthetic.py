import numpy as np
from cdts.ccdc import predict_synthetic_image

def test_predict_synthetic_image():
    # Shape: (max_segments=2, params_per_seg=3 + 6bands*7 = 45, rows=2, cols=2)
    # Actually just test with 1 band to make it simple (num_bands=1)
    # params_per_seg = 3 + 1*7 = 10
    ccdc_coefs_stack = np.zeros((2, 10, 2, 2), dtype=np.float32)
    
    # Segment 1: active from day 1 to 100
    ccdc_coefs_stack[0, 0, :, :] = 1   # t_start
    ccdc_coefs_stack[0, 1, :, :] = 100 # t_end
    
    # Set a simple intercept for band 1, coef 0
    # band 1 starts at index 4 (0=start, 1=end, 2=break, 3=rmse)
    ccdc_coefs_stack[0, 4, :, :] = 500.0 # intercept
    
    # Synthetic image for day 50 (should fall in segment 1)
    synthetic = predict_synthetic_image(ccdc_coefs_stack, target_julian_day=50, num_bands=1)
    
    assert synthetic.shape == (1, 2, 2)
    
    # Because only intercept is set and terms[0] = 1, output should be exactly 500
    assert np.allclose(synthetic, 500.0)

