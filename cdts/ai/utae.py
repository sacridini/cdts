import torch
import torch.nn as nn

class LTAE(nn.Module):
    """ Lightweight Temporal Attention Encoder """
    def __init__(self, in_channels, d_model=256, n_head=8):
        super().__init__()
        self.query = nn.Linear(in_channels, d_model)
        self.key = nn.Linear(in_channels, d_model)
        self.value = nn.Linear(in_channels, d_model)
        
        # Positional encoding for Julian Dates
        self.date_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_linear = nn.Linear(d_model, in_channels)

    def forward(self, x, dates):
        # x: (B, T, C, H, W) -> flatten spatial dims
        B, T, C, H, W = x.shape
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        
        # Date PE
        dates_flat = dates.unsqueeze(-1)
        if dates_flat.dim() == 2:
            dates_flat = dates_flat.unsqueeze(0).expand(B*H*W, -1, -1)
        
        pe = self.date_mlp(dates_flat)
        
        q = self.query(x_flat) + pe
        k = self.key(x_flat) + pe
        v = self.value(x_flat)
        
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.size(-1)**0.5), dim=-1)
        context = torch.bmm(attn, v)
        
        out = self.out_linear(self.norm(context + q))
        # Collapse time dimension
        out = out.mean(dim=1)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return out

class UTAE(nn.Module):
    """ Fully Functional U-TAE Mock (Encoder + L-TAE) """
    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.ltae = LTAE(64)
        self.decoder = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, dates):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B*T, C, H, W)
        feats = self.encoder(x_reshaped)
        feats = feats.view(B, T, 64, H, W)
        
        fused = self.ltae(feats, dates)
        return self.decoder(fused)
