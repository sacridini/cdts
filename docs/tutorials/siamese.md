# Siamese Change Detector

## 1. Introduction

The Siamese Change Detector is a **bi-temporal** architecture: given two co-registered images of the same area at two different dates, it outputs a per-pixel change probability map. It follows the general design of:

> Daudt, R. C., Le Saux, B., & Boulch, A. (2018). *Fully convolutional siamese networks for change detection*. **2018 25th IEEE International Conference on Image Processing (ICIP)** (pp. 4063–4067). [https://doi.org/10.1109/ICIP.2018.8451652](https://doi.org/10.1109/ICIP.2018.8451652)

Unlike [LightTAE](ltae.md), [TempCNN](tempcnn.md), and [UTAE](utae.md) — which were ported layer-for-layer and rigorously cross-validated against reference implementations (`sits` or the official UTAE repo) — `cdts.ai.SiameseChangeDetector` is a **compact, independent implementation** of the general Siamese/twin-encoder change-detection pattern, not a line-for-line port of a specific published codebase. Treat it as a solid, ready-to-train baseline architecture for two-date change detection rather than a bit-exact reproduction of any one paper's exact numbers.

## 2. How It Works

The core idea of a Siamese network is **weight sharing**: the same encoder is applied independently to both input images, so that the two resulting feature maps live in a comparable representation space. `SiameseChangeDetector`'s forward pass:

1. **Twin encoding** (`forward_once`): each input image (`x_t0`, the "before" image, and `x_t1`, the "after" image) is passed *independently* through the **same** two-stage convolutional encoder (`ConvBlock` at 64 channels, max-pooled, then a second `ConvBlock` at 128 channels) — the weights are shared, so a difference in the two output feature maps reflects a genuine change in content, not a difference in how the two images were processed.
2. **Difference**: the absolute difference between the two encoded feature maps (`|feat_t0 - feat_t1|`) is computed — large values indicate the encoder detected substantially different content at that spatial location between the two dates.
3. **Decoder**: the difference map is passed through a decoding `ConvBlock`, upsampled back to the input resolution (bilinear upsampling), and a final `1x1` convolution (`classifier`) produces per-pixel class logits (by default `num_classes=2`: "no change" vs. "change").

## 3. When to Use It

Use the Siamese Change Detector for classic **bi-temporal change detection**: you have exactly two dates (before/after an event — a wildfire, deforestation, a flood, construction) and want a change map between them. If you have a **longer time series** and want to classify or segment based on the whole trajectory rather than just two snapshots, use [LightTAE](ltae.md) (per-pixel) or [UTAE](utae.md) (whole-patch segmentation) instead.

## 4. Preparing Your Data

The model expects two separate image tensors of identical shape, `(Batch, Channels, Height, Width)` — one per acquisition date — plus a binary (or multi-class) change label map of shape `(Batch, Height, Width)`.

```python
import numpy as np
import torch

# x_t0, x_t1: (n_samples, n_bands, patch_h, patch_w) - "before" and "after" patches
x_t0_train = torch.tensor(np.load("images_before.npy"), dtype=torch.float32)
x_t1_train = torch.tensor(np.load("images_after.npy"), dtype=torch.float32)
# y: (n_samples, patch_h, patch_w) - 0 = no change, 1 = change
y_train = torch.tensor(np.load("change_masks.npy"), dtype=torch.long)
```

`x_t0` and `x_t1` must have the same spatial dimensions and be co-registered — the model assumes pixel `(i, j)` in both images corresponds to the same location on the ground.

## 5. Instantiating the Model

```python
from cdts.ai import SiameseChangeDetector

model = SiameseChangeDetector(
    in_channels=6,     # spectral bands per image
    num_classes=2,     # 2 for binary change/no-change; more for multi-class change type
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## 6. Loss Functions

Two natural options from `cdts.ai.losses`, depending on how you want to train:

- **`FocalLoss`** / **`TverskyLoss`**: apply directly to the classifier's `(B, num_classes, H, W)` output logits vs. the `(B, H, W)` label map — the standard approach if you're training the full pipeline (encoders + decoder + classifier) end-to-end as a segmentation problem. Both handle the severe class imbalance typical of change detection (changed pixels are usually a small minority).
- **`ContrastiveSiameseLoss`**: operates directly on the two *encoder* feature maps (`feat_t0`, `feat_t1` from `forward_once`) rather than the final classifier output — it pulls feature vectors together for unchanged pixels and pushes them apart (up to a margin) for changed pixels. Use this if you want to train the twin encoder as a metric-learning problem (e.g. for downstream thresholding or few-shot change detection), separately from or in addition to the classifier head.

```python
from cdts.ai.losses import FocalLoss, ContrastiveSiameseLoss

criterion = FocalLoss(alpha=0.25, gamma=2.0)
# or, to also supervise the encoder features directly:
contrastive_criterion = ContrastiveSiameseLoss(margin=2.0)
```

## 7. Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(
    TensorDataset(x_t0_train, x_t1_train, y_train), batch_size=16, shuffle=True
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for x_t0, x_t1, labels in train_loader:
        x_t0, x_t1, labels = x_t0.to(device), x_t1.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(x_t0, x_t1)      # (batch, num_classes, H, W)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
```

## 8. Inference

```python
model.eval()

with torch.no_grad():
    new_t0 = torch.rand(1, 6, 256, 256).to(device)
    new_t1 = torch.rand(1, 6, 256, 256).to(device)

    logits = model(new_t0, new_t1)
    change_map = torch.argmax(logits, dim=1)   # (1, H, W), 0 = no change, 1 = change

    print(f"Change map shape: {change_map.shape}")
```

> **Pro Tip:** because the decoder only upsamples once (matching the single `MaxPool2d(2)` in the encoder), the output resolution matches the input resolution exactly — no separate resizing step is needed before comparing the predicted change map to your reference mask.

---

## References

- Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully convolutional siamese networks for change detection. In **2018 25th IEEE International Conference on Image Processing (ICIP)** (pp. 4063–4067). [https://doi.org/10.1109/ICIP.2018.8451652](https://doi.org/10.1109/ICIP.2018.8451652)
