"""Loss functions for building segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """Boundary-weighted BCE loss.

    Applies higher loss weight near building boundaries using a distance-transform
    derived weight map.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, boundary_weights: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted = bce * (1.0 + boundary_weights * 5.0)  # 5x weight at boundaries
        return weighted.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice + Boundary loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.3, boundary_weight: float = 0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits, targets, boundary_weights):
        loss = (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
            + self.boundary_weight * self.boundary(logits, targets, boundary_weights)
        )
        return loss
