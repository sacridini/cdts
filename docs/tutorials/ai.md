# AI & Deep Learning in CDTS

While traditional algorithms like LandTrendr and CCDC rely on pixel-based statistical modeling, modern Remote Sensing increasingly leverages Deep Learning for spatial-temporal representation. The `cdts.ai` module provides native PyTorch implementations of state-of-the-art neural network architectures specifically designed for Earth Observation and Change Detection.

## 1. Available Architectures

The module exposes several advanced architectures ready to be trained or fine-tuned:

*   **UTAE & LTAE**: U-Net with Temporal Attention Encoder (UTAE) and Lightweight Temporal Attention Encoder (LTAE). Excellent for processing irregularly sampled time-series (handling cloud gaps inherently) while maintaining spatial context.
*   **TempCNN**: Temporal Convolutional Neural Networks, a highly efficient 1D CNN for pixel-based time-series classification.
*   **Siamese Change Detector**: A bi-temporal architecture designed to take two images (pre and post-event) and output a change probability map. Uses contrastive representation learning.
*   **GeoFoundationViT**: A Vision Transformer wrapper designed to load weights from large geospatial foundation models (like IBM/NASA Prithvi or similar Masked Auto-Encoders) for downstream tasks.

## 2. Preparing the Dataset

To feed multi-temporal, multi-spectral satellite imagery into these models, we provide the `STACCubeDataset` wrapper. This dataset class is designed to lazily load patches from data cubes.

```python
import torch
from torch.utils.data import DataLoader
from cdts.ai import STACCubeDataset

# Define the dataset using directories of GeoTIFF patches
# X_dir contains the time-series patches of shape (Time, Bands, Height, Width)
# y_dir contains the corresponding ground-truth masks
dataset = STACCubeDataset(
    X_dir="./data/train/images",
    y_dir="./data/train/labels",
    transform=None # Add torchvision or albumentations transforms here
)

# Create a PyTorch DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4
)
```

## 3. Instantiating a Model

Let's instantiate the **UTAE** (U-Net with Temporal Attention Encoder). This model expects a 5D tensor of shape `(Batch, Time, Bands, Height, Width)` and an optional tensor of acquisition dates to calculate temporal positional encoding.

```python
from cdts.ai import UTAE

# Initialize UTAE for a 10-class segmentation problem with 6 input bands
model = UTAE(
    input_dim=6,
    encoder_widths=[64, 64, 64, 128],
    decoder_widths=[32, 32, 64, 128],
    out_conv=[32, 10], # Final output layer for 10 classes
    str_conv_k=4,
    str_conv_s=2,
    str_conv_p=1,
    agg_mode="att_group", 
    encoder_norm="group",
    n_head=16, 
    d_model=256, 
    d_k=4
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## 4. Loss Functions

Imbalanced classes are very common in change detection (where "change" is a rare event compared to "no-change"). `cdts.ai.losses` provides specialized functions to handle this.

```python
from cdts.ai.losses import FocalLoss, TverskyLoss

# Focal Loss heavily penalizes hard-to-classify examples (like rare change pixels)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Tversky Loss allows tuning the penalty for False Positives vs False Negatives
# criterion = TverskyLoss(alpha=0.7, beta=0.3)
```

## 5. Training Loop Example

A standard PyTorch training loop seamlessly integrates with our models. 

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Depending on the model, dates might be required. 
        # If your dataset provides dates, pass them: dates=dates
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
```

## 6. Model Inference

After training, you can run inference on a new time-series stack.

```python
model.eval()

# Example new data: 1 batch, 12 time steps, 6 bands, 256x256 patch
new_data = torch.rand(1, 12, 6, 256, 256).to(device)

with torch.no_grad():
    predictions = model(new_data)
    
    # Get the predicted class for each pixel
    predicted_classes = torch.argmax(predictions, dim=1)
    
    print(f"Prediction shape: {predicted_classes.shape}") 
    # Output: Prediction shape: torch.Size([1, 256, 256])
```

> **Pro Tip:** When running inference over massive geographical areas, use `xarray` or `rasterio` windows to chunk the data into manageable sizes (e.g., `256x256`), run them through the model, and mosaic the results back together.
