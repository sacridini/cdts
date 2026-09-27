import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class SiameseChangeDetector(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(64, 128)
        
        self.decoder = ConvBlock(128, 64)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward_once(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        return e2

    def forward(self, x_t0, x_t1):
        feat_t0 = self.forward_once(x_t0)
        feat_t1 = self.forward_once(x_t1)
        diff = torch.abs(feat_t0 - feat_t1)
        d = self.decoder(diff)
        d = self.up(d)
        return self.classifier(d)
