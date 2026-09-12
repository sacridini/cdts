# AI & Deep Learning in CDTS

While traditional algorithms like LandTrendr and CCDC rely on pixel-based statistical modeling, modern Remote Sensing increasingly leverages Deep Learning for spatial-temporal representation. The `cdts.ai` module provides native PyTorch implementations of state-of-the-art neural network architectures specifically designed for Earth Observation and Change Detection.

## 1. Available Architectures

The module exposes several advanced architectures ready to be trained or fine-tuned (see [References](#references) for the original papers behind each):

*   **UTAE & LTAE**: U-Net with Temporal Attention Encoder (UTAE) and Lightweight Temporal Attention Encoder (LTAE), from Garnot & Landrieu. Excellent for processing irregularly sampled time-series (handling cloud gaps inherently) while maintaining spatial context.
*   **TempCNN**: Temporal Convolutional Neural Networks (Pelletier *et al.*), a highly efficient 1D CNN for pixel-based time-series classification.
*   **Siamese Change Detector**: A bi-temporal architecture (Daudt *et al.*) designed to take two images (pre and post-event) and output a change probability map. Uses contrastive representation learning.
*   **GeoFoundationViT**: A Vision Transformer wrapper designed to load weights from large geospatial foundation models — such as IBM/NASA's Prithvi (Jakubik *et al.*) or SatMAE-style Masked Auto-Encoders (Cong *et al.*) — for downstream tasks.

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

# Focal Loss (Lin et al., 2017) heavily penalizes hard-to-classify examples (like rare change pixels)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Tversky Loss (Salehi et al., 2017) allows tuning the penalty for False Positives vs False Negatives
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

---

## References

- Garnot, V. S. F., & Landrieu, L. (2021). Panoptic segmentation of satellite image time series with convolutional temporal attention networks. In **Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)** (pp. 4852–4861). [https://doi.org/10.1109/ICCV48922.2021.00483](https://doi.org/10.1109/ICCV48922.2021.00483)
- Garnot, V. S. F., Landrieu, L., Giordano, S., & Chehata, N. (2020). Satellite image time series classification with pixel-set encoders and temporal self-attention. In **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)** (pp. 12322–12331). [https://doi.org/10.1109/CVPR42600.2020.01234](https://doi.org/10.1109/CVPR42600.2020.01234)
- Pelletier, C., Webb, G. I., & Petitjean, F. (2019). Temporal convolutional neural network for the classification of satellite image time series. **Remote Sensing**, 11(5), 523. [https://doi.org/10.3390/rs11050523](https://doi.org/10.3390/rs11050523)
- Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully convolutional siamese networks for change detection. In **2018 25th IEEE International Conference on Image Processing (ICIP)** (pp. 4063–4067). [https://doi.org/10.1109/ICIP.2018.8451652](https://doi.org/10.1109/ICIP.2018.8451652)
- Jakubik, J., Roy, S., Phillips, C. E., Fraccaro, P., Godwin, D., Zadrozny, B., et al. (2023). *Foundation models for generalist geospatial artificial intelligence*. arXiv:2310.18660. [https://arxiv.org/abs/2310.18660](https://arxiv.org/abs/2310.18660)
- Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D., & Ermon, S. (2022). SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. In **Advances in Neural Information Processing Systems 35 (NeurIPS 2022)**.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In **2017 IEEE International Conference on Computer Vision (ICCV)** (pp. 2999–3007). [https://doi.org/10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324)
- Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. In **Machine Learning in Medical Imaging (MLMI 2017)** (pp. 379–387). [https://doi.org/10.1007/978-3-319-67389-9_44](https://doi.org/10.1007/978-3-319-67389-9_44)
