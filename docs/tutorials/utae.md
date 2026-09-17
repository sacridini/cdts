# UTAE (U-Net with Temporal Attention Encoder)

## 1. Introduction

UTAE combines a U-Net with a temporal attention mechanism to perform **spatio-temporal segmentation** of satellite image time series — producing a full class map over a whole image patch (e.g. `256x256`) using its *entire* observation history, not just a per-pixel classification. It was introduced in:

> Garnot, V. S. F., & Landrieu, L. (2021). *Panoptic segmentation of satellite image time series with convolutional temporal attention networks*. ICCV 2021. [doi:10.1109/ICCV48922.2021.00483](https://doi.org/10.1109/ICCV48922.2021.00483)

`cdts.ai.UTAE` was **ported layer-for-layer from the official reference implementation** ([VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps), MIT License). This is the most rigorously validated model in `cdts.ai`: building both implementations with identical weights (loaded via `load_state_dict()`, with `state_dict()` key names matching directly, no translation table) and feeding them the same input reproduces the reference implementation's output **bit-for-bit exactly** (`max abs diff = 0.0`), including the padded-sequence (irregular temporal sampling) code path.

## 2. How It Works

UTAE is a multi-scale U-Net where every stage is applied independently to each timestep (weights shared across time — see `_TemporallySharedBlock`), and temporal fusion happens once, at the bottleneck, via an image-aware L-TAE variant (`_LTAE2d`). The architecture has four parts:

1. **Encoder**: a standard convolutional U-Net encoder (`in_conv` + a stack of `_DownConvBlock`s), applied frame-by-frame with shared weights, producing a pyramid of per-timestep feature maps at decreasing spatial resolution.
2. **Temporal bottleneck (`_LTAE2d`)**: applies a per-pixel L-TAE (the same "learned master query" multi-head attention mechanism as [LTAE](ltae.md), but computed independently at every spatial position of the deepest feature map) to fuse the time dimension into a single feature map, **plus** per-head, per-timestep attention maps.
3. **Temporal aggregator (`_TemporalAggregator`)**: this is UTAE's key idea — the attention maps from step 2 are resampled to each decoder scale and used to weight the temporal aggregation of *every* skip connection, not just the bottleneck. This propagates the learned "when does this pixel matter" signal to every resolution of the decoder.
4. **Decoder**: a standard U-Net decoder (`_UpConvBlock` stack) that upsamples the fused bottleneck feature, concatenating each attention-weighted skip connection along the way, ending in a final `out_conv` producing per-pixel class logits over the whole patch.

UTAE supports **irregular temporal sampling**: pass a `pad_value` (default `0`) and pad shorter sequences in a batch up to a common length with that value — the model automatically builds a `pad_mask` and skips (or masks out) padded frames in both the shared-weight per-timestep convolutions and the attention mechanism, so you don't need every sample in a batch to have the exact same number of valid observations.

## 3. When to Use It

Use UTAE when you need a **class map over a spatial patch** informed by its full time series — e.g. crop-type mapping, burned-area segmentation, or any task where spatial context (not just a single pixel's own spectral history) matters. If you only need a classification of individual pixels' own time series (no spatial context needed), [LightTAE](ltae.md) or [TempCNN](tempcnn.md) are lighter-weight and faster to train.

## 4. Preparing Your Data

UTAE expects a **5D tensor** of shape `(Batch, Time, Bands, Height, Width)`, plus a `(Batch, Time)` tensor of acquisition dates (used for the positional encoding — raw day-of-year or day-offset values, not calendar dates).

```python
import numpy as np
import torch

# X: (n_samples, n_times, n_bands, patch_h, patch_w)
X_train = torch.tensor(np.load("image_patches.npy"), dtype=torch.float32)
# y: (n_samples, patch_h, patch_w) - one class label per pixel
y_train = torch.tensor(np.load("segmentation_masks.npy"), dtype=torch.long)
# dates: (n_samples, n_times) - day offsets/day-of-year per acquisition, per sample
dates_train = torch.tensor(np.load("acquisition_dates.npy"), dtype=torch.float32)
```

For irregular sequence lengths within a batch, pad the shorter sequences (along the `Time` axis) with the same `pad_value` you'll pass to `UTAE` (default `0`) — the model detects fully-padded frames automatically via `(input == pad_value).all(...)`.

## 5. Instantiating the Model

```python
from cdts.ai import UTAE

model = UTAE(
    input_dim=6,                    # number of spectral bands
    encoder_widths=(64, 64, 64, 128),
    decoder_widths=(32, 32, 64, 128),
    out_conv=[32, 10],              # final conv stack -> 10 output classes
    str_conv_k=4, str_conv_s=2, str_conv_p=1,  # strided-conv down/up-sampling geometry
    agg_mode="att_group",           # temporal aggregation mode for skip connections
    encoder_norm="group",
    n_head=16,
    d_model=256,
    d_k=4,
    pad_value=0,                    # value used for padded (missing) timesteps
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

`out_conv`'s last entry sets the number of output classes; `encoder_widths`/`decoder_widths` must have matching lengths and equal final entries (assertions enforce this at construction time).

## 6. Training Loop

Segmentation targets are dense per-pixel class maps, so `FocalLoss` or `TverskyLoss` (both operating on `(B, C, H, W)` logits vs. `(B, H, W)` labels) are natural fits for imbalanced classes such as rare disturbance/change events.

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cdts.ai.losses import FocalLoss

train_loader = DataLoader(TensorDataset(X_train, dates_train, y_train), batch_size=8, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for images, dates, labels in train_loader:
        images, dates, labels = images.to(device), dates.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images, batch_positions=dates)  # (batch, n_classes, H, W)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
```

## 7. Inference

```python
model.eval()

with torch.no_grad():
    # 1 batch, 12 time steps, 6 bands, 256x256 patch
    new_data = torch.rand(1, 12, 6, 256, 256).to(device)
    new_dates = torch.arange(12, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 12)

    logits = model(new_data, batch_positions=new_dates)
    predicted_classes = torch.argmax(logits, dim=1)

    print(f"Prediction shape: {predicted_classes.shape}")
    # Output: Prediction shape: torch.Size([1, 256, 256])
```

You can also request the raw attention maps (`return_att=True`) for interpretability, e.g. to visualize which timesteps the model relied on most for a given pixel:

```python
logits, attn = model(new_data, batch_positions=new_dates, return_att=True)
# attn: (n_head, batch, n_times, H, W)
```

> **Pro Tip:** for inference over massive geographical areas, use `xarray`/`rasterio` windows to chunk the data into `256x256` (or similar) patches, run them through the model, and mosaic the results back together.

## 8. Validation Against the Official Reference

`UTAE` (and its internal `LTAE2d`, `_TemporalAggregator`, etc.) is a line-for-line port of [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps). Validation methodology: the official repo was cloned locally, both implementations were instantiated with the same hyperparameters and the same random weights (copied via `load_state_dict()` — the `state_dict()` key names match with no translation needed), and run forward on identical random input. The outputs were bit-for-bit identical (`max abs diff = 0.0`), across both the regular (unpadded) code path and the padded-sequence (`pad_value`/`pad_mask`) code path used for irregular temporal sampling. This cross-check is not part of the pytest suite, since it requires the reference repo cloned locally rather than a pip dependency — see the source docstring in `cdts/ai/utae.py` for details.

---

## References

- Garnot, V. S. F., & Landrieu, L. (2021). Panoptic segmentation of satellite image time series with convolutional temporal attention networks. In **Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)** (pp. 4852–4861). [https://doi.org/10.1109/ICCV48922.2021.00483](https://doi.org/10.1109/ICCV48922.2021.00483)
- Garnot, V. S. F., & Landrieu, L. (2020). *Lightweight Temporal Self-Attention for Classifying Satellite Image Time Series*. arXiv:2007.00586. [https://arxiv.org/abs/2007.00586](https://arxiv.org/abs/2007.00586)
- Reference implementation: [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps) (MIT License)
