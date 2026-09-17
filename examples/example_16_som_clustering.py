"""
Example 16: Self-Organizing Map (SOM) End-to-End Unsupervised Clustering

Loads a synthetic 4-band reflectance datacube (no network needed) with 3
spatially distinct land-cover-like clusters (water, vegetation, bare soil),
trains a Batch SOM on the per-pixel spectra with `cdts.ai.SOM`, predicts a
Best-Matching-Unit (BMU) map, and saves both the raw BMU map and a
noise-filtering diagnostic with `cdts.save_raster`.
"""
import os
import numpy as np
from rasterio.transform import from_origin
import cdts
from cdts.ai import SOM


def build_synthetic_scene(rows=40, cols=40, seed=9):
    """
    Three spectrally distinct classes arranged in vertical thirds. The
    REFLECTANCE always follows the true (clean) spatial class - only the
    training LABELS get some salt-and-pepper corruption, simulating
    mislabeled ground-truth samples (e.g. a GPS/digitizing error) whose
    spectra still clearly belong to their real class's SOM neuron. This is
    exactly the case `filter_noisy_samples` is meant to catch: a label that
    disagrees with the majority label among its neuron's other members.
    """
    rng = np.random.RandomState(seed)
    signatures = {
        0: np.array([0.05, 0.06, 0.04, 0.03]),   # water: low reflectance everywhere, esp. NIR/SWIR
        1: np.array([0.04, 0.09, 0.05, 0.45]),   # vegetation: high NIR
        2: np.array([0.25, 0.28, 0.30, 0.35]),   # bare soil: bright, flat spectrum
    }
    third = cols // 3
    labels = np.zeros((rows, cols), dtype=np.int32)
    labels[:, third:2 * third] = 1
    labels[:, 2 * third:] = 2

    data = np.zeros((rows, cols, 4), dtype=np.float64)
    for cls, sig in signatures.items():
        mask = labels == cls
        data[mask] = sig + rng.normal(0, 0.015, (mask.sum(), 4))

    # Corrupt only the LABELS of a few pixels (not their reflectance).
    noisy_labels = labels.copy()
    flip_idx = rng.choice(rows * cols, size=(rows * cols) // 25, replace=False)
    for idx in flip_idx:
        true_cls = labels.flat[idx]
        noisy_labels.flat[idx] = rng.choice([c for c in signatures if c != true_cls])

    return data, labels, noisy_labels


def main():
    print("CDTS Example 16: Self-Organizing Map (SOM) Unsupervised Clustering")

    rows, cols = 40, 40
    print(f"\n[1/3] Generating synthetic 4-band scene ({rows}x{cols} px, 3 land-cover clusters + noise)...")
    data, clean_labels, noisy_labels = build_synthetic_scene(rows=rows, cols=cols)
    samples = data.reshape(-1, 4)

    print("\n[2/3] Training a 4x4 Batch SOM and predicting Best-Matching Units...")
    som = SOM(x=4, y=4, input_len=4, sigma=1.5, random_seed=42)
    som.train(samples, num_iters=100, n_jobs=-1)
    bmus = som.predict(samples, n_jobs=-1)
    bmu_map = bmus.reshape(rows, cols)
    print(f"    {len(np.unique(bmus))} of {4 * 4} neurons were actually activated as a winner.")

    print("    Using filter_noisy_samples() to flag pixels that disagree with their neuron's majority class...")
    clean_mask = som.filter_noisy_samples(samples, noisy_labels.reshape(-1), n_jobs=-1)
    clean_mask_map = clean_mask.reshape(rows, cols)
    true_noise_mask = (noisy_labels != clean_labels)
    recall = (~clean_mask_map[true_noise_mask]).mean() if true_noise_mask.any() else float("nan")
    print(f"    Flagged {np.sum(~clean_mask):d}/{clean_mask.size} pixels as noisy; "
          f"recall on the actually-injected noise: {recall:.1%}.")

    print("\n[3/3] Saving results with cdts.save_raster()...")
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)

    out_bmu = os.path.join("data", "som_bmu_map.tif")
    cdts.save_raster(bmu_map.astype("float32"), out_bmu, crs="EPSG:32721", transform=transform, nodata=-1.0)
    print(f"    BMU map (0-15, one id per SOM neuron) -> {out_bmu}")

    out_clean = os.path.join("data", "som_clean_mask.tif")
    cdts.save_raster(clean_mask_map.astype("uint8"), out_clean, crs="EPSG:32721", transform=transform, nodata=255)
    print(f"    Noise-filter mask (1=kept, 0=flagged as noisy) -> {out_clean}")

    print("\nDone!")


if __name__ == "__main__":
    main()
