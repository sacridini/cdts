"""Regenerate tests/data/som_minisom_parity.npz from Python MiniSom.

Trains MiniSom (the reference cdts.ai.SOM ports) on the cases below and stores
the input data, the configuration and the trained codebook, so the parity
tests run even where minisom is not installed.

    pip install minisom==2.3.6
    python tests/data/make_som_parity_fixtures.py
"""
import json
import os

import numpy as np
from minisom import MiniSom

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (MiniSom kwargs, init, train method, train kwargs)
CASES = {
    "online_sequential": (dict(x=5, y=5, sigma=1.2), "random", "train", dict(num_iteration=500)),
    "online_random_order": (dict(x=6, y=4, sigma=1.5), "none", "train", dict(num_iteration=800, random_order=True)),
    "online_epochs": (dict(x=4, y=4, sigma=1.0), "pca", "train", dict(num_iteration=3, use_epochs=True)),
    "online_hex_mexican": (dict(x=6, y=5, sigma=1.3, topology="hexagonal", neighborhood_function="mexican_hat",
                                decay_function="linear_decay_to_zero", sigma_decay_function="linear_decay_to_one"),
                           "random", "train", dict(num_iteration=400)),
    "online_bubble": (dict(x=5, y=5, sigma=2.0, neighborhood_function="bubble",
                           decay_function="inverse_decay_to_zero", sigma_decay_function="inverse_decay_to_one"),
                      "random", "train", dict(num_iteration=400, random_order=True)),
    "online_triangle_d24": (dict(x=5, y=6, sigma=2.0, neighborhood_function="triangle"),
                            "random", "train", dict(num_iteration=400)),
    "batch_gaussian": (dict(x=6, y=6, sigma=2.0), "random", "train_batch_offline", dict(num_iteration=15)),
    "batch_hex_d24": (dict(x=5, y=7, sigma=1.5, topology="hexagonal"), "random", "train_batch_offline",
                      dict(num_iteration=10)),
}


def make_data(name, n_features):
    rng = np.random.RandomState(sum(map(ord, name)))
    centers = rng.uniform(-4, 4, size=(4, n_features))
    X = np.vstack([c + rng.normal(0, 0.6, size=(75, n_features)) for c in centers])
    return X[rng.permutation(len(X))]


def main():
    out = {}
    for name, (kwargs, init, method, train_kwargs) in CASES.items():
        n_features = 24 if name.endswith("d24") else 6
        X = make_data(name, n_features)
        kwargs = dict(kwargs, input_len=n_features, learning_rate=0.5, random_seed=11)
        som = MiniSom(**kwargs)
        if init == "random":
            som.random_weights_init(X)
        elif init == "pca":
            som.pca_weights_init(X)
        getattr(som, method)(X, **train_kwargs)
        out[name + "__data"] = X
        out[name + "__weights"] = som.get_weights()
        out[name + "__config"] = np.array(json.dumps(
            dict(som=kwargs, init=init, method=method, train=train_kwargs)))
    np.savez_compressed(os.path.join(HERE, "som_minisom_parity.npz"), **out)
    print("wrote", len(CASES), "cases")


if __name__ == "__main__":
    main()
