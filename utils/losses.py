from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _default_criterion(
    binary: bool,
    class_weights: Optional[torch.Tensor],
    device: torch.device,
) -> nn.Module:
    if binary:
        return nn.BCELoss()
    w = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=w)


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Dice loss for binary segmentation.

    Total loss = alpha * BCE + (1 - alpha) * Dice

    Parameters
    ----------
    alpha : float
        Weight of the BCE term (default 0.5).
    smooth : float
        Laplace smoothing for the Dice term (default 1.0).
    pos_weight : torch.Tensor or None
        Passed to BCEWithLogitsLoss. When set the model must output raw
        logits (no Sigmoid); leave None if Sigmoid is already applied.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.smooth = smooth
        self.bce = (
            nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if pos_weight is not None
            else nn.BCELoss()
        )

    def _dice(self, preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        p = preds.reshape(preds.size(0), -1)
        m = masks.reshape(masks.size(0), -1)
        intersection = (p * m).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            p.sum(dim=1) + m.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()

    def forward(self, preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.bce(preds, masks) + (1.0 - self.alpha) * self._dice(preds, masks)


def compute_class_weights(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from training masks.

    Returns a 1-D FloatTensor of length num_classes, normalised so its
    sum equals num_classes.
    """
    print("Computing class weights...")
    counts = torch.zeros(num_classes)
    for _, masks in tqdm(train_loader):
        for c in range(num_classes):
            counts[c] += (masks == c).sum()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    print("Class weights:", {i: f"{w:.4f}" for i, w in enumerate(weights.tolist())})
    return weights
