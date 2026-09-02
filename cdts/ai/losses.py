import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Change Detection (handles extreme class imbalance).
    Down-weights easy examples (e.g., the 99% of unchanged pixels) and focuses on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        # inputs: logits (B, C, H, W)
        # targets: labels (B, H, W)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class TverskyLoss(nn.Module):
    """
    Tversky Loss for Change Detection.
    A generalization of Dice Loss. 
    alpha = weight of False Positives
    beta = weight of False Negatives
    Setting beta > alpha penalizes missing change more than false alarms.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        # Assuming class 1 is the 'change' class
        p_pos = probs[:, 1, ...]
        t_pos = targets.float()

        tp = (p_pos * t_pos).sum()
        fp = (p_pos * (1 - t_pos)).sum()
        fn = ((1 - p_pos) * t_pos).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index

class ContrastiveSiameseLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    Minimizes feature distance for unchanged pixels (label 0).
    Maximizes feature distance (up to a margin) for changed pixels (label 1).
    """
    def __init__(self, margin: float = 2.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        # feat1, feat2: (B, C, H, W)
        # label: (B, H, W) -> 0 for unchanged, 1 for changed
        euclidean_distance = F.pairwise_distance(feat1, feat2, keepdim=True) # (B, 1, H, W)
        label = label.unsqueeze(1).float()
        
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive
