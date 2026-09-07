# Self-Organizing Maps (SOM)

CDTS includes a highly optimized, multi-threaded C++ implementation of the **Batch Self-Organizing Map (Batch SOM)** algorithm. It is specifically designed to handle large Earth Observation data cubes efficiently using OpenMP and Eigen (SIMD vectorization).

Self-Organizing Maps are unsupervised neural networks used to reduce the dimensionality of your data, clustering similar temporal profiles into a 2D grid. It is incredibly useful for discovering land cover patterns without prior training data.

## Training a SOM

You can train a SOM directly on a 2D numpy array (Pixels x Features). The underlying C++ engine handles the heavy lifting safely.

```python
import numpy as np
from cdts.ai import train_som_batch, predict_bmus

# Simulate 10,000 pixels with 24 features (e.g., 4 bands x 6 time steps)
X_train = np.random.rand(10000, 24).astype(np.float64)

# Train a 10x10 SOM
# C++ OpenMP automatically uses available cores (leaving 1 free to keep the OS responsive)
som_weights = train_som_batch(
    data=X_train,
    grid_rows=10,
    grid_cols=10,
    num_epochs=100,
    initial_learning_rate=0.5,
    n_jobs=-1 # Automatically uses all cores minus 1
)

print("Trained SOM Weights Shape:", som_weights.shape) # (100, 24)
```

## Predicting Best Matching Units (BMUs)

Once trained, you can classify new pixels by finding their Best Matching Unit (BMU) on the 2D grid. 

```python
# Simulate new data
X_new = np.random.rand(5000, 24).astype(np.float64)

# Predict the BMU index for each pixel (values from 0 to 99)
bmus = predict_bmus(X_new, som_weights, n_jobs=-1)

print("BMUs Shape:", bmus.shape) # (5000,)
```

## Hardware & Threading (n_jobs)

Both `train_som_batch` and `predict_bmus` expose the `n_jobs` parameter to control the level of parallelization via C++ OpenMP.

- `n_jobs = -1` (Default): Uses `max_threads - 1`. Your machine will stay responsive during heavy training.
- `n_jobs = 4`: Forces the use of exactly 4 cores.
- `n_jobs = 1`: Disables OpenMP (runs sequentially, great for debugging).
