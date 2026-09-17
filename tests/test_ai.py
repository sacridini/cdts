import torch
from cdts.ai.tempcnn import TempCNN
from cdts.ai.utae import LTAE, LightTAE, UTAE

def test_tempcnn():
    # Batch size 4, 6 bands, 24 timesteps
    x = torch.randn(4, 6, 24)
    model = TempCNN(in_channels=6, n_times=24, num_classes=5)

    out = model(x)
    assert out.shape == (4, 5) # 4 pixels, 5 classes


def test_ltae():
    day_offsets = list(range(0, 24 * 16, 16))  # 24 steps, 16-day composites
    model = LTAE(in_channels=128, day_offsets=day_offsets)

    x = torch.randn(4, 24, 128)  # (batch, seq_len, in_channels)
    out = model(x)
    assert out.shape == (4, 128)  # n_neurons[-1] default


def test_lighttae():
    day_offsets = list(range(0, 24 * 16, 16))
    model = LightTAE(n_bands=6, day_offsets=day_offsets, n_labels=5)

    x = torch.randn(4, 24, 6)  # (batch, n_times, n_bands)
    out = model(x)
    assert out.shape == (4, 5)


def test_utae():
    model = UTAE(in_channels=6, num_classes=3)

    x = torch.randn(2, 5, 6, 8, 8)  # (B, T, C, H, W)
    dates = torch.arange(5, dtype=torch.float32)
    out = model(x, dates)
    assert out.shape == (2, 3, 8, 8)
