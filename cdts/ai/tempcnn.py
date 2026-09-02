import torch
import torch.nn as nn
import torch.nn.functional as F

class TempCNN(nn.Module):
    """
    TempCNN: 1D Convolutional Neural Network for Satellite Image Time Series.
    Reference: Pelletier et al. (2019).
    Unlike U-TAE which looks at the 3D cube, TempCNN looks at a single pixel's time series.
    """
    def __init__(self, in_channels: int, num_classes: int = 5, num_filters: int = 64, kernel_size: int = 5, dropout_rate: float = 0.5) -> None:
        super().__init__()
        
        # 1D Convolutions apply along the Time dimension.
        # Input shape expected: (Batch, Channels, Time)
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(num_filters)
        
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Classifier
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x shape: (Batch, Channels, Time)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global Average Pooling over Time
        x = x.mean(dim=2) # Shape: (Batch, num_filters)
        
        x = self.dropout(x)
        out = self.fc(x)
        return out
