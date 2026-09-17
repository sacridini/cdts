import math
import torch
import torch.nn as nn


class _LinearBNReLU(nn.Module):
    """Linear -> BatchNorm1d -> ReLU (no dropout), matching sits's
    .torch_linear_batch_norm_relu (R/api_torch.R)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x)


class _MultiLinearBNReLU(nn.Module):
    """Chain of _LinearBNReLU blocks, matching sits's
    .torch_multi_linear_batch_norm_relu (R/api_torch.R). Used for the pixel
    spatial encoder, LTAE's internal MLP, and the LightTAE decoder."""

    def __init__(self, input_dim: int, hidden_dims: "list[int]") -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        self.model = nn.Sequential(*[
            _LinearBNReLU(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.model(x)


class _PixelSpatialEncoder(nn.Module):
    """Per-pixel MLP spatial encoder, matching sits's
    .torch_pixel_spatial_encoder (R/api_torch_psetae.R). Runs independently
    on every (pixel, time) observation - a stand-in for the full Pixel-Set
    Encoder of Garnot et al. (2020) when only single-pixel time series (not
    parcel-level sets) are available, per sits's own docs."""

    def __init__(self, n_bands: int, layers_spatial_encoder: "tuple[int, ...]" = (32, 64, 128)) -> None:
        super().__init__()
        self.layers_spatial_encoder = layers_spatial_encoder
        self.spatial_encoder = _MultiLinearBNReLU(n_bands, list(layers_spatial_encoder))

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        # values: (batch, n_times, n_bands) -> (batch, n_times, dim_enc)
        batch_size, n_times, n_bands = values.shape
        dim_enc = self.layers_spatial_encoder[-1]
        values = values.reshape(batch_size * n_times, n_bands)
        values = self.spatial_encoder(values)
        return values.reshape(batch_size, n_times, dim_enc)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017 style) over day
    offsets from the first observation, matching sits's
    .torch_positional_encoding (R/api_torch_psetae.R)."""

    def __init__(self, day_offsets: "list[float]", dim_encoder: int = 128) -> None:
        super().__init__()
        day_offsets_t = torch.as_tensor(day_offsets, dtype=torch.float32)
        len_max = day_offsets_t.shape[0]
        days_t = day_offsets_t.unsqueeze(1)  # (len_max, 1)

        p = torch.zeros(len_max, dim_encoder)
        div_term = torch.exp(
            torch.arange(0, dim_encoder, 2, dtype=torch.float32) * (-math.log(1000.0) / dim_encoder)
        ).unsqueeze(0)
        p[:, 0::2] = torch.sin(days_t * div_term)
        p[:, 1::2] = torch.cos(days_t * div_term)
        p = p.unsqueeze(0)  # (1, len_max, dim_encoder)
        self.register_buffer("p", p)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return x + self.p


class _ScaledDotProductAttention(nn.Module):
    """Matches sits's .torch_scaled_dot_product_attention (R/api_torch_psetae.R)."""

    def __init__(self, temperature: float, attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query: "torch.Tensor", keys: "torch.Tensor", values: "torch.Tensor") -> "torch.Tensor":
        # query: (n_heads*batch, d_k) -> (n_heads*batch, 1, d_k)
        # keys: (n_heads*batch, seq_len, d_k); values: (n_heads*batch, seq_len, split_value)
        attn = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2))  # (n_heads*batch, 1, seq_len)
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        return torch.matmul(attn, values)  # (n_heads*batch, 1, split_value)


class _MultiHeadAttention(nn.Module):
    """Learned "master query" multi-head attention - the core L-TAE trick
    (Garnot & Landrieu, 2020): the query is a single learned per-head vector
    shared across all inputs (not computed from the input, unlike standard
    self-attention), which is what makes this lightweight. Matches sits's
    .torch_multi_head_attention (R/api_torch_psetae.R)."""

    def __init__(self, n_heads: int, d_k: int, d_in: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros(n_heads, d_k))
        nn.init.normal_(self.Q, mean=0.0, std=math.sqrt(2.0 / d_k))

        self.fc_k = nn.Linear(d_in, n_heads * d_k)
        nn.init.normal_(self.fc_k.weight, mean=0.0, std=math.sqrt(2.0 / d_k))

        self.attention = _ScaledDotProductAttention(temperature=math.sqrt(d_k))

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        # values: (batch, seq_len, d_in)
        batch_size, seq_len, _ = values.shape
        n_heads, d_k, d_in = self.n_heads, self.d_k, self.d_in

        # master query, broadcast across the batch: (n_heads, batch, d_k) -> (n_heads*batch, d_k)
        query = self.Q.unsqueeze(1).expand(n_heads, batch_size, d_k).reshape(n_heads * batch_size, d_k)

        keys = self.fc_k(values)  # (batch, seq_len, n_heads*d_k)
        keys = keys.view(batch_size, seq_len, n_heads, d_k)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(n_heads * batch_size, seq_len, d_k)

        split_value = d_in // n_heads
        v = torch.stack(values.split(split_value, dim=-1), dim=0)  # (n_heads, batch, seq_len, split_value)
        v = v.view(n_heads * batch_size, seq_len, split_value)

        out = self.attention(query, keys, v)  # (n_heads*batch, 1, split_value)
        out = out.view(n_heads, batch_size, 1, d_in // n_heads).squeeze(2)  # (n_heads, batch, d_in // n_heads)
        return out


class LTAE(nn.Module):
    """
    Lightweight Temporal Attention Encoder (L-TAE), ported layer-for-layer
    from sits's `.torch_light_temporal_attention_encoder`
    (R/api_torch_psetae.R), itself an implementation of:

    Garnot, V.S.F. & Landrieu, L. (2020). "Lightweight Temporal
    Self-Attention for Classifying Satellite Image Time Series".
    arXiv:2007.00586.

    Takes a (batch, seq_len, in_channels) sequence and returns a
    (batch, n_neurons[-1]) fused embedding. `day_offsets` (day counts from
    the first observation, for the sinusoidal positional encoding) is fixed
    at construction time - like sits's `timeline` parameter, a given LTAE
    instance is tied to one sequence length/temporal sampling for its
    lifetime (the internal Conv1d+LayerNorm and positional encoding buffers
    are sized from it).
    """

    def __init__(
        self,
        in_channels: int = 128,
        day_offsets: "list[float]" = None,
        n_heads: int = 16,
        n_neurons: "tuple[int, ...]" = (256, 128),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if day_offsets is None:
            raise ValueError(
                "day_offsets is required: the fixed timeline (day counts from the "
                "first observation) this LTAE instance is built for, matching sits's "
                "sits_lighttae() `timeline` parameter."
            )
        seq_len = len(day_offsets)

        self.in_channels = in_channels
        self.n_heads = n_heads
        self.d_key_query = in_channels // n_heads
        self.d_model = n_neurons[0]

        self.in_layer_norm = nn.LayerNorm(in_channels)
        self.inconv = nn.Sequential(
            nn.Conv1d(in_channels, self.d_model, kernel_size=1),
            nn.LayerNorm((self.d_model, seq_len)),
        )
        self.pos_encoding = _PositionalEncoding(day_offsets, dim_encoder=self.d_model)

        self.attention_heads = _MultiHeadAttention(n_heads=n_heads, d_k=self.d_key_query, d_in=self.d_model)
        self.mlp = _MultiLinearBNReLU(n_neurons[0], list(n_neurons[1:]))
        self.dropout = nn.Dropout(dropout_rate)
        self.out_layer_norm = nn.LayerNorm(n_neurons[-1])

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        # values: (batch, seq_len, in_channels)
        batch_size = values.shape[0]

        values = self.in_layer_norm(values)
        values = self.inconv(values.permute(0, 2, 1))
        values = values.permute(0, 2, 1)
        values = self.pos_encoding(values)

        values = self.attention_heads(values)  # (n_heads, batch, d_model // n_heads)
        values = values.permute(1, 0, 2).contiguous().view(batch_size, -1)  # (batch, d_model)

        values = self.mlp(values)
        values = self.dropout(values)
        values = self.out_layer_norm(values)
        return values


class LightTAE(nn.Module):
    """
    Full Light Temporal Attention Encoder classifier for pixel-level
    satellite image time series, ported layer-for-layer from sits's
    `sits_lighttae()` (R/sits_lighttae.R): a per-pixel MLP spatial encoder
    -> LTAE temporal fusion -> MLP decoder to class logits (softmax applied
    externally, matching sits). This is the model directly comparable to
    `sits_lighttae()`'s trained output - `LTAE` above is just its reusable
    temporal-fusion block.
    """

    def __init__(
        self,
        n_bands: int,
        day_offsets: "list[float]",
        n_labels: int,
        layers_spatial_encoder: "tuple[int, ...]" = (32, 64, 128),
        n_heads: int = 16,
        n_neurons: "tuple[int, ...]" = (256, 128),
        dropout_rate: float = 0.2,
        dim_input_decoder: int = 128,
        dim_layers_decoder: "tuple[int, ...]" = (64, 32),
    ) -> None:
        super().__init__()
        self.spatial_encoder = _PixelSpatialEncoder(n_bands, layers_spatial_encoder)
        self.temporal_encoder = LTAE(
            in_channels=layers_spatial_encoder[-1],
            day_offsets=day_offsets,
            n_heads=n_heads,
            n_neurons=n_neurons,
            dropout_rate=dropout_rate,
        )
        self.decoder = _MultiLinearBNReLU(dim_input_decoder, [*dim_layers_decoder, n_labels])

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, n_times, n_bands)
        x = self.spatial_encoder(x)
        x = self.temporal_encoder(x)
        return self.decoder(x)


class _NaiveTemporalFusion(nn.Module):
    """Simple per-pixel single-head temporal attention with a date MLP for
    positional encoding. NOT ported from any published architecture or from
    sits - a placeholder fusion block used internally by UTAE below while
    UTAE itself is being rewritten against the U-TAE paper (Garnot &
    Landrieu, 2021) in a follow-up change. Do not use directly."""

    def __init__(self, in_channels: int, d_model: int = 256) -> None:
        super().__init__()
        self.query = nn.Linear(in_channels, d_model)
        self.key = nn.Linear(in_channels, d_model)
        self.value = nn.Linear(in_channels, d_model)
        self.date_mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.out_linear = nn.Linear(d_model, in_channels)

    def forward(self, x: "torch.Tensor", dates: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, C, H, W) -> flatten spatial dims
        B, T, C, H, W = x.shape
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        dates_flat = dates.unsqueeze(-1)
        if dates_flat.dim() == 2:
            dates_flat = dates_flat.unsqueeze(0).expand(B * H * W, -1, -1)

        pe = self.date_mlp(dates_flat)
        q = self.query(x_flat) + pe
        k = self.key(x_flat) + pe
        v = self.value(x_flat)

        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5), dim=-1)
        context = torch.bmm(attn, v)

        out = self.out_linear(self.norm(context + q))
        out = out.mean(dim=1)  # collapse time
        return out.view(B, H, W, C).permute(0, 3, 1, 2)


class UTAE(nn.Module):
    """
    U-Net + Temporal Attention Encoder for spatio-temporal segmentation.

    NOTE: this is currently a simplified stand-in, not yet a faithful port of
    Garnot & Landrieu (2021) "Panoptic Segmentation of Satellite Image Time
    Series with Convolutional Temporal Attention Networks" (the U-TAE
    paper) - it uses a single conv encoder/decoder pair and the internal
    `_NaiveTemporalFusion` block rather than U-TAE's multi-scale U-Net
    encoder with temporally-pooled skip connections. A paper-faithful
    rewrite is planned as a follow-up; there is no `sits` equivalent to
    cross-validate against (sits's temporal attention models - TAE,
    LightTAE - operate per-pixel, not on full spatial feature maps).
    """

    def __init__(self, in_channels: int, num_classes: int = 5) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.ltae = _NaiveTemporalFusion(64)
        self.decoder = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: "torch.Tensor", dates: "torch.Tensor") -> "torch.Tensor":
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feats = self.encoder(x_reshaped)
        feats = feats.view(B, T, 64, H, W)

        fused = self.ltae(feats, dates)
        return self.decoder(fused)
