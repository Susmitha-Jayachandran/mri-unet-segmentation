"""Loss function and evaluation metrics for binary segmentation."""
import torch
import torch.nn as nn


def dice_score(logits, target, eps=1e-6):
    """Mean Dice coefficient over the batch (threshold 0.5 on sigmoid)."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def iou_score(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


class DiceBCELoss(nn.Module):
    """BCE (pixel-wise) + soft Dice (overlap) - robust to class imbalance."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.bce, self.eps = nn.BCEWithLogitsLoss(), eps

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean()
        return self.bce(logits, target) + dice_loss
