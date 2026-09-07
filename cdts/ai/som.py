import numpy as np
from .. import _core

class SOM:
    """
    High-Performance Self-Organizing Map (SOM) accelerated by C++, OpenMP and Eigen.
    Uses the Batch SOM algorithm which is massively faster for large satellite datasets.
    """
    def __init__(self, x: int, y: int, input_len: int, sigma: float = 1.0, random_seed: int = 42):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.random_seed = random_seed
        self.weights = np.zeros((x, y, input_len), dtype=np.float64)
        self._is_trained = False
        
    def train(self, data: np.ndarray, num_iters: int, n_jobs: int = -1):
        data = np.ascontiguousarray(data, dtype=np.float64)
        # Ensure 2D input
        if data.ndim != 2:
            raise ValueError("Data must be a 2D array [Samples, Features]")
            
        self.weights = _core.som.train_som_batch(
            data, self.x, self.y, num_iters, self.sigma, n_jobs, self.random_seed
        )
        self._is_trained = True
        
    def predict(self, data: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("SOM is not trained yet.")
        data = np.ascontiguousarray(data, dtype=np.float64)
        w = np.ascontiguousarray(self.weights, dtype=np.float64)
        return _core.som.predict_bmus(data, w, n_jobs)

    def filter_noisy_samples(self, data: np.ndarray, labels: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        winners_flat = self.predict(data, n_jobs)
        clean_mask = np.ones(len(data), dtype=bool)
        
        for w in np.unique(winners_flat):
            neuron_mask = (winners_flat == w)
            neuron_labels = labels[neuron_mask]
            if len(neuron_labels) == 0:
                continue
            
            vals, counts = np.unique(neuron_labels, return_counts=True)
            majority = vals[np.argmax(counts)]
            
            mismatch = neuron_mask & (labels != majority)
            clean_mask[mismatch] = False
            
        return clean_mask
