# TempCNN

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Classify each pixel's time series into land-cover classes, fast.</span></div>
<div><span class="k">Input</span><span class="v"><code>(batch, bands, time)</code> tensors with a fixed length</span></div>
<div><span class="k">Output</span><span class="v">Class logits per pixel</span></div>
<div><span class="k">Reference</span><span class="v">Pelletier et al. (2019); weight-compatible with R <code>sits</code></span></div>
</div>

<figure markdown>
  ![TempCNN training on four synthetic land-cover classes: samples, accuracy per epoch and confusion matrix](../assets/figures/tempcnn_training.png)
  <figcaption><strong>A complete training run.</strong> 1,000 noisy NDVI series (23 dates, 12% cloud-contaminated observations) of four classes, split 70/30. After 30 epochs the held-out accuracy is 96%. The only confusion is between single and double cropping, whose second season is weak in many samples. It uses the same model and a training loop like the one below; the full script is in <code>docs/scripts/make_figures.py</code>.</figcaption>
</figure>

TempCNN is a 1D convolutional neural network designed for classifying **per-pixel satellite image time series**. It's a simpler, faster-to-train alternative to attention-based models like [LightTAE](ltae.md), and a strong baseline for most pixel time-series classification tasks. It was introduced in:

> Pelletier, C., Webb, G. I., & Petitjean, F. (2019). *Temporal convolutional neural network for the classification of satellite image time series*. **Remote Sensing**, 11(5), 523. [https://doi.org/10.3390/rs11050523](https://doi.org/10.3390/rs11050523)

`zeit.ai.TempCNN` was **ported layer-for-layer from the R package [`sits`](https://github.com/e-sensing/sits)'s `sits_tempcnn()`** (`R/sits_tempcnn.R`, `R/api_torch.R`), so trained weights are directly portable between the two via `state_dict()` — no name translation needed. This was checked by exporting a trained `sits_tempcnn()` model's weights, loading them into `zeit.ai.TempCNN` via `load_state_dict()`, and confirming the predictions match `sits`'s own output within float32 numerical tolerance on identical input.

## How It Works

TempCNN treats the time axis of a pixel's spectral history like the spatial axis of a 1D signal, and applies a stack of 1D convolutions along it:

1. **Three convolutional blocks** (`Conv1d -> BatchNorm1d -> ReLU -> Dropout`, default widths `(64, 64, 64)` and kernel sizes `(3, 3, 3)`), each convolving over the time axis while keeping every spectral band as a separate input channel.
2. **Flatten**: the full `(hidden_dim, n_times)` feature map is flattened into a single vector — **not** global-average-pooled. This is a deliberate architectural choice matching `sits`'s implementation: it means the dense layer's input size is tied to `n_times`, so **a given `TempCNN` instance is fixed to one sequence length** for its lifetime (unlike LightTAE, whose attention mechanism can be more flexible about padding, though `day_offsets` is still fixed per-instance too).
3. **Dense block** (`Linear -> BatchNorm1d -> ReLU -> Dropout`, default `256` nodes) followed by a **final linear classifier** producing `num_classes` logits (softmax applied externally, e.g. via `torch.nn.functional.cross_entropy` or manually at inference time).

## When to Use It

| | TempCNN | LightTAE |
|---|---|---|
| Best for | Fast baselines, shorter/noisier series, limited training data | Longer, well-sampled series where attention over specific timesteps helps |
| Compute cost | Lower (no attention, no positional encoding) | Moderate |
| Cross-validated against | `sits_tempcnn()` (R) | `sits_lighttae()` (R) |

See the [LTAE & LightTAE tutorial](ltae.md) for the attention-based alternative, and the [UTAE tutorial](utae.md) if you need whole-patch spatial segmentation rather than per-pixel classification.

## Preparing Your Data

`TempCNN` expects a tensor of shape `(Batch, Channels, Time)` — spectral bands as channels, observations along the time axis (this is `sits`'s own internal convention; if your data is naturally `(Batch, Time, Bands)`, transpose the last two axes with `.permute(0, 2, 1)` before feeding it in).

```python
import numpy as np
import torch

# X: (n_samples, n_bands, n_times)
X_train = torch.tensor(np.load("pixel_time_series.npy"), dtype=torch.float32)
y_train = torch.tensor(np.load("pixel_labels.npy"), dtype=torch.long)

n_bands = X_train.shape[1]
n_times = X_train.shape[2]
```

## Instantiating the Model

```python
from zeit.ai import TempCNN

model = TempCNN(
    in_channels=n_bands,
    n_times=n_times,               # fixed sequence length this instance is built for
    num_classes=10,
    hidden_dims=(64, 64, 64),       # widths of the 3 conv blocks
    kernel_sizes=(3, 3, 3),
    dropout_rates=(0.2, 0.2, 0.2),
    dense_layer_nodes=256,
    dense_layer_dropout_rate=0.5,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()  # or zeit.ai.losses.FocalLoss for imbalanced classes

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for values, labels in train_loader:
        values, labels = values.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(values)          # (batch, num_classes)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
```

## Inference

```python
model.eval()

with torch.no_grad():
    new_series = torch.rand(1, n_bands, n_times).to(device)  # (1, n_bands, n_times)
    logits = model(new_series)
    predicted_class = torch.argmax(logits, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
```

For inference over a whole raster, extract every pixel's time series into a `(N_pixels, n_bands, n_times)` tensor, run it through the model in batches, and reshape the resulting predictions back to `(H, W)`.

## Validation Against `sits`

`TempCNN` was validated end-to-end against `sits_tempcnn()`: a model trained in R was exported, its weights loaded into `zeit.ai.TempCNN` via `load_state_dict()` (a direct, layer-for-layer match — no key renaming), and run on the same input. Outputs matched `sits`'s predictions within float32 numerical tolerance.

---

## References

- Pelletier, C., Webb, G. I., & Petitjean, F. (2019). Temporal convolutional neural network for the classification of satellite image time series. **Remote Sensing**, 11(5), 523. [https://doi.org/10.3390/rs11050523](https://doi.org/10.3390/rs11050523)
- e-sensing/sits: [https://github.com/e-sensing/sits](https://github.com/e-sensing/sits)
