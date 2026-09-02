import numpy as np
import rasterio
import os
from cdts.classify import train_ccdc_classifier, classify_ccdc_stack

def test_ccdc_classification(tmp_path):
    # Dummy training data
    # 50 samples of class 1, 50 samples of class 2
    # Let say we have 6 bands, 7 params each (RMSE + 6 coeffs) = 42 params
    X_train = np.random.rand(100, 42)
    
    # Make class 1 distinct from class 2
    X_train[:50, 0] += 5.0
    X_train[50:, 0] -= 5.0
    
    y_train = np.array([1]*50 + [2]*50)
    
    clf = train_ccdc_classifier(X_train, y_train, n_estimators=10)
    
    # Create a dummy coef stack GeoTIFF
    coef_path = str(tmp_path / "dummy_coefs.tif")
    out_path = str(tmp_path / "dummy_class.tif")
    
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 42,
        "dtype": "float32",
        "nodata": 0
    }
    
    # Test image where top half is class 1, bottom is class 2
    test_img = np.random.rand(42, 10, 10).astype(np.float32)
    test_img[0, :5, :] += 5.0
    test_img[0, 5:, :] -= 5.0
    
    with rasterio.open(coef_path, "w", **profile) as dst:
        dst.write(test_img)
        
    classify_ccdc_stack(clf, coef_path, out_path, chunk_size=5)
    
    assert os.path.exists(out_path)
    
    with rasterio.open(out_path) as src:
        classified = src.read(1)
        
    assert classified.shape == (10, 10)
    
    # The top half should mostly be 1, bottom half mostly 2
    # Because of random noise it might not be 100% perfect but should be very close
    top_half = classified[:5, :]
    bottom_half = classified[5:, :]
    
    assert np.mean(top_half == 1) > 0.8
    assert np.mean(bottom_half == 2) > 0.8

