# Self-Organizing Maps (SOM)

CDTS includes a multi-threaded C++ implementation of the **Self-Organizing Map** proposed by Kohonen (1990), in both its classic **online** form and the **Batch SOM** form (Kohonen, 2013) (see [References](#references)).

Self-Organizing Maps are unsupervised neural networks used to reduce the dimensionality of your data, clustering similar temporal profiles into a 2D grid. They are useful for discovering land cover patterns without prior training data and for cleaning noisy training samples.

`cdts.ai.SOM` is an operation-by-operation port of Python [`minisom`](https://github.com/JustGlowing/minisom): given the same seed and arguments, it produces **exactly the same codebook** as `MiniSom`, only much faster (see the [benchmarks](../benchmarks/fidelity.md#6-self-organizing-maps-som)).

## Training a SOM

You can train a SOM directly on a 2D numpy array (Pixels x Features).

```python
import numpy as np
from cdts.ai import SOM

# Simulate 10,000 pixels with 24 features (e.g., 4 bands x 6 time steps)
X_train = np.random.rand(10000, 24)

som = SOM(x=10, y=10, input_len=24, sigma=1.5, learning_rate=0.5, random_seed=42)
som.random_weights_init(X_train)   # or som.pca_weights_init(X_train)

# Batch SOM: 20 passes over the whole dataset, parallelized with OpenMP
som.train(X_train, num_iters=20, algorithm="batch", n_jobs=-1)

print("Trained SOM Weights Shape:", som.get_weights().shape)  # (10, 10, 24)
print("Quantization error:", som.quantization_error(X_train))
```

### Online vs. Batch

| `algorithm` | Equivalent `minisom` call | Meaning of `num_iters` | Parallel |
| :--- | :--- | :--- | :---: |
| `'online'` (default) | `MiniSom.train(data, num_iters, random_order, use_epochs)` | Number of single-sample updates (epochs with `use_epochs=True`) | No (sequential by definition) |
| `'batch'` | `MiniSom.train_batch_offline(data, num_iters)` | Number of full passes over the data | Yes (`n_jobs`) |

```python
# Online SOM: 50,000 single-sample updates, samples drawn in random order
som.train(X_train, num_iters=50_000, random_order=True)

# Online SOM for 5 epochs over the data
som.train(X_train, num_iters=5, use_epochs=True)
```

`train` continues from the current weights, like `minisom`. Other options mirror `minisom` too: `neighborhood_function` (`'gaussian'`, `'mexican_hat'`, `'bubble'`, `'triangle'`), `topology` (`'rectangular'`, `'hexagonal'`), `decay_function` and `sigma_decay_function`.

## Predicting Best Matching Units (BMUs)

Once trained, you can classify new pixels by finding their Best Matching Unit (BMU) on the 2D grid.

```python
X_new = np.random.rand(5000, 24)

# Flat BMU index for each pixel: i * y + j (values from 0 to 99)
bmus = som.predict(X_new, n_jobs=-1)
print("BMUs Shape:", bmus.shape)  # (5000,)

# Grid coordinates of a single sample, as MiniSom.winner
print(som.winner(X_new[0]))
```

## Filtering Noisy Training Samples

`filter_noisy_samples` flags samples whose label disagrees with the majority label of their neuron:

```python
clean_mask = som.filter_noisy_samples(X_train, labels)
X_clean, y_clean = X_train[clean_mask], labels[clean_mask]
```

## Hardware & Threading (n_jobs)

`train(..., algorithm="batch")`, `predict` and `quantization_error` expose the `n_jobs` parameter to control the level of parallelization via C++ OpenMP. Batch results are identical for any `n_jobs`.

- `n_jobs = -1` (Default): Uses `max_threads - 1`. Your machine will stay responsive during heavy training.
- `n_jobs = 4`: Forces the use of exactly 4 cores.
- `n_jobs = 1`: Runs sequentially.

---

## References

- Kohonen, T. (1990). The self-organizing map. **Proceedings of the IEEE**, 78(9), 1464–1480. [https://doi.org/10.1109/5.58325](https://doi.org/10.1109/5.58325)
- Kohonen, T. (2013). Essentials of the self-organizing map. **Neural Networks**, 37, 52–65. [https://doi.org/10.1016/j.neunet.2012.09.018](https://doi.org/10.1016/j.neunet.2012.09.018)
- Vettigli, G. (2018). MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map. [https://github.com/JustGlowing/minisom](https://github.com/JustGlowing/minisom)
