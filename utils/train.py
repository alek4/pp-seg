from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import _default_criterion


def _prepare_loss(
    preds: torch.Tensor,
    masks: torch.Tensor,
    binary: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if binary:
        masks = masks.float().unsqueeze(1)
    return preds, masks


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    binary: bool = False,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    class_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = "best_model.pth",
    device: Optional[torch.device] = None,
) -> dict:
    """Train a segmentation model with validation and checkpointing.

    Parameters
    ----------
    model : nn.Module
    train_loader, valid_loader : DataLoader
        Yield (images, masks) batches.
    epochs : int
    lr : float
        Used only when optimizer is None.
    binary : bool
        True for [B,1,H,W] + Sigmoid output; False for [B,C,H,W] logits.
    criterion : nn.Module or None
        Defaults to BCELoss (binary) or CrossEntropyLoss (multi-class).
    optimizer : Optimizer or None
        Defaults to Adam.
    class_weights : torch.Tensor or None
        For CrossEntropyLoss; ignored in binary mode.
    save_path : str or None
        Checkpoint path for best validation loss; None to skip.
    device : torch.device or None
        Auto-detected when None.

    Returns
    -------
    dict with "train_losses" and "valid_losses" (per-epoch averages).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if criterion is None:
        criterion = _default_criterion(binary, class_weights, device)
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=lr)

    train_losses, valid_losses = [], []
    best_valid_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            preds_loss, masks_loss = _prepare_loss(preds, masks, binary)
            loss = criterion(preds_loss, masks_loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [valid]"):
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                preds_loss, masks_loss = _prepare_loss(preds, masks, binary)
                loss = criterion(preds_loss, masks_loss)
                running_loss += loss.item()
        valid_loss = running_loss / len(valid_loader)
        valid_losses.append(valid_loss)

        if save_path is not None and valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model -> {save_path}  (valid loss: {valid_loss:.4f})")

        print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Valid: {valid_loss:.4f}")

    return {"train_losses": train_losses, "valid_losses": valid_losses}
