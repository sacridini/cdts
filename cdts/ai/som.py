import numpy as np

class SOM:
    """
    Self-Organizing Map (SOM) for clustering and filtering time series samples.
    Similar to sits_som_map for identifying noisy training samples.
    """
    def __init__(self, x: int, y: int, input_len: int, sigma: float = 1.0, learning_rate: float = 0.5, random_seed: int = None):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        if random_seed is not None:
            np.random.seed(random_seed)
        self.weights = np.random.rand(x, y, input_len)
        self._xx, self._yy = np.meshgrid(np.arange(x), np.arange(y), indexing='ij')
    
    def winner(self, x: np.ndarray) -> tuple:
        dists = np.linalg.norm(self.weights - x, axis=-1)
        w = np.unravel_index(np.argmin(dists), dists.shape)
        return w
        
    def update(self, x: np.ndarray, win: tuple, t: int, max_iters: int):
        eta = self.learning_rate * np.exp(-t / max_iters)
        sig = self.sigma * np.exp(-t / max_iters)
        
        d = (self._xx - win[0])**2 + (self._yy - win[1])**2
        h = np.exp(-d / (2 * sig**2 + 1e-8))
        
        self.weights += eta * h[..., np.newaxis] * (x - self.weights)
        
    def train(self, data: np.ndarray, num_iters: int):
        for t in range(num_iters):
            idx = np.random.randint(0, len(data))
            x = data[idx]
            win = self.winner(x)
            self.update(x, win, t, num_iters)
            
    def filter_noisy_samples(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Groups data into neurons and filters out samples whose label doesn't match the neuron's majority label.
        Returns a boolean mask of 'clean' samples.
        """
        winners = np.array([self.winner(d) for d in data])
        winners_flat = winners[:, 0] * self.y + winners[:, 1]
        
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
