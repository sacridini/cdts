import torch
import torch.nn as nn


class _ConvBNReLUDropout(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU -> Dropout, matching sits's
    .torch_conv1D_batch_norm_relu_dropout (R/api_torch.R)."""

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, padding: int, dropout_rate: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x)


class _LinearBNReLUDropout(nn.Module):
    """Linear -> BatchNorm1d -> ReLU -> Dropout, matching sits's
    .torch_linear_batch_norm_relu_dropout (R/api_torch.R)."""

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x)


class TempCNN(nn.Module):
    """
    TempCNN: 1D Convolutional Neural Network for Satellite Image Time Series.
    Reference: Pelletier et al. (2019), "Temporal Convolutional Neural Network
    for the Classification of Satellite Image Time Series", Remote Sensing,
    11, 523. doi:10.3390/rs11050523.

    Architecture ported layer-for-layer from the R package `sits`'s
    `sits_tempcnn()` (R/sits_tempcnn.R, R/api_torch.R) so trained weights are
    directly portable between the two implementations for cross-validation:
    3x (Conv1d -> BatchNorm1d -> ReLU -> Dropout), flattened over the full
    time axis (NOT global-average-pooled - the flatten step means `n_times`
    is baked into the dense layer's input size, so a model instance is tied
    to one fixed sequence length, matching sits's behavior), then
    (Linear -> BatchNorm1d -> ReLU -> Dropout) and a final Linear classifier.
    """

    def __init__(
        self,
        in_channels: int,
        n_times: int,
        num_classes: int = 5,
        hidden_dims: "tuple[int, int, int]" = (64, 64, 64),
        kernel_sizes: "tuple[int, int, int]" = (3, 3, 3),
        dropout_rates: "tuple[float, float, float]" = (0.2, 0.2, 0.2),
        dense_layer_nodes: int = 256,
        dense_layer_dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.conv_bn_relu1 = _ConvBNReLUDropout(
            in_channels, hidden_dims[0], kernel_sizes[0], kernel_sizes[0] // 2, dropout_rates[0]
        )
        self.conv_bn_relu2 = _ConvBNReLUDropout(
            hidden_dims[0], hidden_dims[1], kernel_sizes[1], kernel_sizes[1] // 2, dropout_rates[1]
        )
        self.conv_bn_relu3 = _ConvBNReLUDropout(
            hidden_dims[1], hidden_dims[2], kernel_sizes[2], kernel_sizes[2] // 2, dropout_rates[2]
        )

        self.flatten = nn.Flatten()
        self.dense = _LinearBNReLUDropout(
            hidden_dims[2] * n_times, dense_layer_nodes, dense_layer_dropout_rate
        )
        self.nn_linear = nn.Sequential(nn.Linear(dense_layer_nodes, num_classes))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x shape: (Batch, Channels, Time) - same input convention as before.
        # (sits's forward() takes (Batch, Time, Bands) and transposes internally
        # to this same (Batch, Bands, Time) layout before the first conv block.)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.nn_linear(x)
