# GeoFoundationViT

## 1. Introduction

`GeoFoundationViT` is a thin wrapper that lets you plug large, pretrained **geospatial foundation models** (Vision Transformers trained on massive satellite imagery corpora) into a `cdts` workflow, and fine-tune a lightweight classification/segmentation head on top for your own downstream task. Rather than training a model from scratch, you're doing **transfer learning** from a model that has already learned general-purpose visual representations of satellite imagery.

By default it loads NASA/IBM's **Prithvi-100M** via HuggingFace `transformers`:

> Jakubik, J., Roy, S., Phillips, C. E., Fraccaro, P., Godwin, D., Zadrozny, B., et al. (2023). *Foundation models for generalist geospatial artificial intelligence*. arXiv:2310.18660. [https://arxiv.org/abs/2310.18660](https://arxiv.org/abs/2310.18660)

but any compatible HuggingFace geospatial ViT (e.g. SatMAE-style models) can be loaded by passing a different `model_id`.

> **Unlike** [LightTAE](ltae.md), [TempCNN](tempcnn.md), and [UTAE](utae.md) — which are `cdts`-native architectures ported and validated against reference implementations — `GeoFoundationViT` is a **wrapper around an external pretrained model**. Its behavior and output quality depend entirely on the backbone you load; there is no `cdts`-side numerical validation to speak of here, since correctness is inherited from the upstream model.

## 2. How It Works

```
input (B, Bands, Time, H, W) -> ViT backbone -> classifier (1x1 Conv2d) -> per-pixel class logits
```

1. **Backbone**: on construction, `GeoFoundationViT` attempts to load the requested `model_id` from the HuggingFace Hub via `transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)`.
2. **Graceful fallback**: if the download fails (no network access, model unavailable, missing `transformers` extras, etc.), the wrapper does **not** raise — it silently falls back to a randomly-initialized `Conv3d` projection (`self.fallback_conv`) standing in for the backbone. This keeps the class usable offline/in CI, but means predictions will be meaningless until you either restore network access or explicitly train the fallback conv from scratch. **Check `model.has_hf` after construction** to know which path you're on.
3. **Reshape + classify**: the backbone's patch-token output (`last_hidden_state`, shape `(B, Seq, Dim)`) is reshaped back into a 2D feature map (assuming a square patch grid, `H = W = sqrt(Seq)`), upsampled by 16x (matching the typical ViT patch size) to recover roughly the original spatial resolution, and passed through a final `1x1 Conv2d` classifier head to produce per-pixel class logits.

## 3. When to Use It

Use `GeoFoundationViT` when you have **limited labeled data** for your specific task but want to benefit from representations learned on a much larger, general-purpose satellite imagery corpus — a classic transfer-learning scenario. If you have ample labeled training data and want an architecture purpose-built and validated for time-series classification/segmentation, prefer [LightTAE](ltae.md), [TempCNN](tempcnn.md), or [UTAE](utae.md) instead.

## 4. Preparing Your Data

The exact expected input shape depends on the specific backbone you load (check that model's HuggingFace card for its required resolution, band count/ordering, and normalization statistics — Prithvi-100M, for instance, expects specific band selections and a fixed patch size). As a general pattern:

```python
import torch

# Example shape convention (verify against your chosen model_id's documentation):
# (Batch, Bands, Time, Height, Width)
x = torch.rand(2, 6, 1, 224, 224)
```

Always check the backbone's model card for required preprocessing (band order, normalization/statistics, expected patch size) before training — mismatched preprocessing is the most common cause of poor fine-tuning results with foundation models.

## 5. Instantiating the Model

```python
from cdts.ai import GeoFoundationViT

model = GeoFoundationViT(
    model_id="ibm-nasa-geospatial/Prithvi-100M",
    num_classes=2,
)

if not model.has_hf:
    print("Warning: backbone failed to load from HuggingFace Hub - using untrained fallback conv.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## 6. Fine-Tuning

Because the backbone carries pretrained weights worth preserving, it's common to **freeze it initially** and only train the lightweight classifier head, then optionally unfreeze the backbone for a lower-learning-rate fine-tuning pass once the head has converged:

```python
import torch.optim as optim
from cdts.ai.losses import FocalLoss

# Stage 1: freeze the backbone, train only the classifier head
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    for images, labels in train_loader:  # your DataLoader
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Head-only] Epoch [{epoch + 1}/10], Loss: {epoch_loss / len(train_loader):.4f}")

# Stage 2 (optional): unfreeze the backbone for full fine-tuning at a lower LR
for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    epoch_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Full fine-tune] Epoch [{epoch + 1}/5], Loss: {epoch_loss / len(train_loader):.4f}")
```

## 7. Inference

```python
model.eval()

with torch.no_grad():
    new_data = torch.rand(1, 6, 1, 224, 224).to(device)
    logits = model(new_data)
    predicted_classes = torch.argmax(logits, dim=1)
    print(f"Prediction shape: {predicted_classes.shape}")
```

## 8. Caveats

- **Network access required for the pretrained path**: the first construction of `GeoFoundationViT` needs to reach the HuggingFace Hub (or a local cache) to download the backbone weights. In offline/air-gapped environments, pre-download the model or explicitly train the fallback path.
- **`trust_remote_code=True`**: loading Prithvi-style models runs custom model code shipped alongside the weights on the Hub. Only point `model_id` at sources you trust.
- **Fallback silently degrades quality**: because construction never raises on a failed download, always check `model.has_hf` in automated pipelines to avoid silently training/evaluating on the untrained fallback path.

---

## References

- Jakubik, J., Roy, S., Phillips, C. E., Fraccaro, P., Godwin, D., Zadrozny, B., et al. (2023). *Foundation models for generalist geospatial artificial intelligence*. arXiv:2310.18660. [https://arxiv.org/abs/2310.18660](https://arxiv.org/abs/2310.18660)
- Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D., & Ermon, S. (2022). SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. In **Advances in Neural Information Processing Systems 35 (NeurIPS 2022)**.
