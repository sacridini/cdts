# Deep Learning

<p class="lead">PyTorch modules in <code>cdts.ai</code>. They are ordinary <code>nn.Module</code>s: move them to any device and train them with your usual loop. Tutorials: <a href="../../tutorials/ai/">Deep Learning</a>.</p>

## Data

### `STACCubeDataset` { .api .cls }

<!-- sig: cdts.ai.STACCubeDataset -->
```python
class cdts.ai.STACCubeDataset(cube, patch_size=256, stride=256)
```

A PyTorch `Dataset` that cuts a lazy xarray cube into square spatial patches. Only the requested patch is computed, so it works on cubes larger than memory.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `xr.DataArray` | required | `(time, band, y, x)` cube, e.g. from `build_time_series`. |
| `patch_size` | `int` | `256` | Patch side in pixels. |
| `stride` | `int` | `256` | Step between patches. Smaller than `patch_size` gives overlapping patches. |

</div>

Each item is `(patch, dates)`: a float32 tensor `(time, band, patch_size, patch_size)` with NaN replaced by 0, and the day of year of each observation. `dataset.dates` holds the same dates.

```python
from cdts.ai import STACCubeDataset

dataset = STACCubeDataset(cube, patch_size=64, stride=64)
patch, dates = dataset[0]
```

## Pixel classifiers

### `TempCNN` { .api .cls }

<!-- sig: cdts.ai.TempCNN -->
```python
class cdts.ai.TempCNN(
    in_channels, n_times, num_classes=5, hidden_dims=(64, 64, 64),
    kernel_sizes=(3, 3, 3), dropout_rates=(0.2, 0.2, 0.2),
    dense_layer_nodes=256, dense_layer_dropout_rate=0.5,
)
```

Temporal convolutional network (Pelletier et al., 2019), ported layer for layer from R `sits`' `sits_tempcnn()`; weights load directly with `load_state_dict`. Input `(batch, bands, time)`, output class logits `(batch, num_classes)`. Tutorial: [TempCNN](../tutorials/tempcnn.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | required | Bands per observation. |
| `n_times` | `int` | required | Series length. Fixed for the life of the model (the flatten layer depends on it). |
| `num_classes` | `int` | `5` | Output classes. |
| `hidden_dims` | `tuple` | `(64, 64, 64)` | Filters of the three convolution blocks. |
| `kernel_sizes` | `tuple` | `(3, 3, 3)` | Kernel sizes. |
| `dropout_rates` | `tuple` | `(0.2, 0.2, 0.2)` | Dropout after each convolution block. |
| `dense_layer_nodes` | `int` | `256` | Width of the dense layer. |
| `dense_layer_dropout_rate` | `float` | `0.5` | Dropout of the dense layer. |

</div>

```python
from cdts.ai import TempCNN

model = TempCNN(in_channels=6, n_times=23, num_classes=4)
logits = model(torch.randn(32, 6, 23))   # (32, 4)
```

### `LightTAE` { .api .cls }

<!-- sig: cdts.ai.LightTAE -->
```python
class cdts.ai.LightTAE(
    n_bands, day_offsets, n_labels,
    layers_spatial_encoder=(32, 64, 128), n_heads=16,
    n_neurons=(256, 128), dropout_rate=0.2, dim_input_decoder=128,
    dim_layers_decoder=(64, 32),
)
```

Lightweight temporal attention classifier: per-observation MLP encoder, L-TAE temporal attention, MLP decoder. Ported from `sits_lighttae()`. Input `(batch, time, bands)`, output logits `(batch, n_labels)`. Tutorial: [LTAE & LightTAE](../tutorials/ltae.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_bands` | `int` | required | Bands per observation. |
| `day_offsets` | `list[float]` | required | Day of each observation, counted from the first. Fixes the series length. |
| `n_labels` | `int` | required | Output classes. |
| `layers_spatial_encoder` | `tuple` | `(32, 64, 128)` | Widths of the per-observation encoder. |
| `n_heads` | `int` | `16` | Attention heads. |
| `n_neurons` | `tuple` | `(256, 128)` | L-TAE widths; `n_neurons[0]` is the attention dimension. |
| `dropout_rate` | `float` | `0.2` | Dropout. |
| `dim_input_decoder` | `int` | `128` | Decoder input width; must equal `n_neurons[-1]`. |
| `dim_layers_decoder` | `tuple` | `(64, 32)` | Decoder hidden widths. |

</div>

```python
from cdts.ai import LightTAE

model = LightTAE(n_bands=6, day_offsets=list(range(0, 36 * 16, 16)), n_labels=5)
logits = model(torch.randn(8, 36, 6))    # (8, 5)
```

### `LTAE` { .api .cls }

<!-- sig: cdts.ai.LTAE -->
```python
class cdts.ai.LTAE(
    in_channels=128, day_offsets=None, n_heads=16,
    n_neurons=(256, 128), dropout_rate=0.2,
)
```

The L-TAE temporal attention block on its own, to build custom models. Input `(batch, time, in_channels)`, output `(batch, n_neurons[-1])`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | `128` | Features per time step. |
| `day_offsets` | `list[float]` | `None` | Day of each observation, counted from the first. Required in practice. |
| `n_heads` | `int` | `16` | Attention heads. |
| `n_neurons` | `tuple` | `(256, 128)` | Internal widths. |
| `dropout_rate` | `float` | `0.2` | Dropout. |

</div>

## Patch segmentation

### `UTAE` { .api .cls }

<!-- sig: cdts.ai.UTAE -->
```python
class cdts.ai.UTAE(
    input_dim, encoder_widths=(64, 64, 64, 128),
    decoder_widths=(32, 32, 64, 128), out_conv=(32, 20), str_conv_k=4,
    str_conv_s=2, str_conv_p=1, agg_mode="att_group",
    encoder_norm="group", n_head=16, d_model=256, d_k=4,
    encoder=False, return_maps=False, pad_value=0,
    padding_mode="reflect",
)
```

U-Net with temporal attention (Garnot & Landrieu, 2021), ported from the official `utae-paps` code and bit-exact with it. Input `(batch, time, bands, H, W)` plus `batch_positions` `(batch, time)` (day of each observation); output class scores `(batch, classes, H, W)`. Tutorial: [UTAE](../tutorials/utae.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | required | Bands. |
| `encoder_widths` | `list[int]` | `(64, 64, 64, 128)` | Encoder widths, finest level first. |
| `decoder_widths` | `list[int]` | `(32, 32, 64, 128)` | Decoder widths. Same length as the encoder; last value equal to the encoder's. |
| `out_conv` | `list[int]` | `(32, 20)` | Output convolution widths; the last is the number of classes. |
| `str_conv_k`, `str_conv_s`, `str_conv_p` | `int` | `4`, `2`, `1` | Kernel, stride and padding of the strided convolutions. |
| `agg_mode` | `str` | `"att_group"` | Temporal aggregation of skip connections: `"att_group"`, `"att_mean"` or `"mean"`. |
| `encoder_norm` | `str` | `"group"` | `"group"`, `"batch"` or `"instance"`. |
| `n_head` | `int` | `16` | Attention heads. |
| `d_model` | `int` | `256` | Attention width (divisible by `n_head`). |
| `d_k` | `int` | `4` | Key size per head. |
| `encoder` | `bool` | `False` | Return features instead of class scores. |
| `return_maps` | `bool` | `False` | Also return the decoder feature maps. |
| `pad_value` | `float` | `0` | Value marking padded time steps (for variable-length batches). |
| `padding_mode` | `str` | `"reflect"` | Spatial padding of the convolutions. |

</div>

```python
from cdts.ai import UTAE

model = UTAE(input_dim=6, out_conv=[32, 10]).eval()
x = torch.randn(2, 12, 6, 128, 128)
days = torch.arange(12.0).expand(2, -1) * 16
scores = model(x, batch_positions=days)   # (2, 10, 128, 128)
```

## Change detection

### `SiameseChangeDetector` { .api .cls }

<!-- sig: cdts.ai.SiameseChangeDetector -->
```python
class cdts.ai.SiameseChangeDetector(in_channels, num_classes=2)
```

Two-date change detection with a shared (Siamese) convolutional encoder: both images are encoded with the same weights, their feature difference is decoded to a per-pixel map. An independent implementation of the design of Daudt et al. (2018). Tutorial: [Siamese Change Detector](../tutorials/siamese.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | required | Bands per image. |
| `num_classes` | `int` | `2` | Output classes (2 for change / no change). |

</div>

```python
from cdts.ai import SiameseChangeDetector

model = SiameseChangeDetector(in_channels=4)
logits = model(torch.randn(8, 4, 256, 256), torch.randn(8, 4, 256, 256))   # (8, 2, 256, 256)
```

## Foundation models

### `GeoFoundationViT` { .api .cls }

<!-- sig: cdts.ai.GeoFoundationViT -->
```python
class cdts.ai.GeoFoundationViT(
    model_id="ibm-nasa-geospatial/Prithvi-100M", num_classes=2,
)
```

Loads a geospatial foundation model from the HuggingFace Hub (Prithvi-100M by default) and adds a 1 × 1 convolution head for segmentation. If the backbone cannot be loaded, it falls back to an untrained 3-D convolution and sets `model.has_hf = False`. Tutorial: [GeoFoundationViT](../tutorials/geo_foundation_vit.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_id` | `str` | `"ibm-nasa-geospatial/Prithvi-100M"` | HuggingFace model ID. |
| `num_classes` | `int` | `2` | Output classes of the head. |

</div>

## Losses

### `FocalLoss` { .api .cls }

<!-- sig: cdts.ai.losses.FocalLoss -->
```python
class cdts.ai.losses.FocalLoss(
    alpha=0.25, gamma=2.0, reduction="mean",
)
```

Cross-entropy that down-weights easy examples, for heavily imbalanced problems such as change detection. Takes logits `(B, C, …)` and integer targets.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `alpha` | `float` | `0.25` | Overall weight. |
| `gamma` | `float` | `2.0` | Focusing strength. `0` gives plain cross-entropy. |
| `reduction` | `str` | `"mean"` | `"mean"` or `"sum"`. |

</div>

### `TverskyLoss` { .api .cls }

<!-- sig: cdts.ai.losses.TverskyLoss -->
```python
class cdts.ai.losses.TverskyLoss(alpha=0.3, beta=0.7, smooth=1.0)
```

Overlap-based loss for the positive (change) class, with separate weights for false positives and false negatives.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `alpha` | `float` | `0.3` | Weight of false positives. |
| `beta` | `float` | `0.7` | Weight of false negatives. `beta > alpha` penalises missed changes more. |
| `smooth` | `float` | `1.0` | Smoothing term. |

</div>

### `ContrastiveSiameseLoss` { .api .cls }

<!-- sig: cdts.ai.losses.ContrastiveSiameseLoss -->
```python
class cdts.ai.losses.ContrastiveSiameseLoss(margin=2.0)
```

Contrastive loss on two feature maps: pulls features together where nothing changed (label 0) and pushes them at least `margin` apart where something changed (label 1).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `margin` | `float` | `2.0` | Minimum distance for changed pixels. |

</div>
