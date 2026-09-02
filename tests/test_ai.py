import torch
from cdts.ai.tempcnn import TempCNN

def test_tempcnn():
    # Batch size 4, 6 bands, 24 timesteps
    x = torch.randn(4, 6, 24)
    model = TempCNN(in_channels=6, num_classes=5)
    
    out = model(x)
    assert out.shape == (4, 5) # 4 pixels, 5 classes
