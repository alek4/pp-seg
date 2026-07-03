import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib.colors import ListedColormap
from PIL import Image
from torch.utils.data import DataLoader

from .evaluate import _binarise


def _make_overlay_cmap(num_colors: int = 2, alpha: float = 0.5) -> ListedColormap:
    colors = np.zeros((num_colors, 4))
    colors[0] = [0, 0, 0, 0]
    colors[1:] = [1, 0, 0, alpha]
    return ListedColormap(colors)


def plot_losses(
    train_losses: Sequence[float],
    valid_losses: Sequence[float],
    *,
    title: str = "Training and Validation Loss",
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(valid_losses, label="Valid Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Loss plot saved -> {save_path}")
    plt.show()


def plot_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    binary: bool = False,
    num_classes: int = 2,
    threshold: float = 0.5,
    n_samples: int = 4,
    device: Optional[torch.device] = None,
    title: str = "Predictions",
    overlay_alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> None:
    """Visualise predictions vs ground-truth for a batch from loader.

    Shows three columns per sample: raw image | ground-truth overlay |
    prediction overlay. Background (class 0) is transparent.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if binary:
        num_classes = 2

    model = model.to(device).eval()

    images, masks = next(iter(loader))
    images = images.to(device)
    n = min(n_samples, images.shape[0])

    with torch.no_grad():
        logits = model(images)
        preds = _binarise(logits, threshold) if binary else torch.argmax(logits, dim=1)

    cmap = _make_overlay_cmap(num_classes, alpha=overlay_alpha)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(title, fontsize=14)
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, ct in enumerate(["Image", "Ground truth", "Prediction"]):
        axes[0, col].set_title(ct, fontsize=12)

    for i in range(n):
        img  = images[i].cpu().permute(1, 2, 0).numpy()
        gt   = masks[i].cpu().numpy()
        pred = preds[i].cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(gt,   cmap=cmap, vmin=0, vmax=num_classes)
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Prediction plot saved -> {save_path}")
    plt.show()


def plot_metrics(
    metrics: dict,
    *,
    title: str = "Model Metrics",
    save_path: Optional[str] = None,
) -> None:
    """Plot per-class and mean IoU, Precision, Recall, F1 as bar charts."""
    class_labels = metrics["class_labels"]
    labels_ext   = class_labels + ["mean"]

    iou  = metrics["per_class_iou"]       + [metrics["mean_iou"]]
    prec = metrics["per_class_precision"] + [metrics["mean_precision"]]
    rec  = metrics["per_class_recall"]    + [metrics["mean_recall"]]
    f1   = metrics["per_class_f1"]        + [metrics["mean_f1"]]

    metric_names  = ["IoU", "Precision", "Recall", "F1"]
    metric_values = [iou, prec, rec, f1]
    colors        = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle(title, fontsize=14, y=1.02)

    x = np.arange(len(labels_ext))
    for ax, name, values, color in zip(axes, metric_names, metric_values, colors):
        bars = ax.bar(x, values, width=0.6, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(1.5)
        ax.set_title(name, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_ext, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        if ax == axes[0]:
            ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Metrics plot saved -> {save_path}")
    plt.show()


def predict_on_images(
    model: torch.nn.Module,
    image_paths: List[str],
    *,
    binary: bool = False,
    num_classes: int = 2,
    threshold: float = 0.5,
    target_size: Tuple[int, int] = (512, 512),
    device: Optional[torch.device] = None,
    overlay_alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> None:
    """Run inference on a list of image files and display overlays.

    Predictions are resized back to each image's original resolution.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if binary:
        num_classes = 2

    model = model.to(device).eval()
    cmap = _make_overlay_cmap(num_classes, alpha=overlay_alpha)

    fig, axes = plt.subplots(len(image_paths), 1, figsize=(8, 6 * len(image_paths)))
    if len(image_paths) == 1:
        axes = [axes]

    for ax, path in zip(axes, image_paths):
        image = Image.open(path).convert("RGB")
        orig_size = image.size

        tensor = (
            TF.to_tensor(TF.resize(image, list(target_size)))
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            logits = model(tensor)
            pred = (
                _binarise(logits, threshold).squeeze(0)
                if binary
                else torch.argmax(logits, dim=1).squeeze(0)
            )

        pred_pil = Image.fromarray(pred.cpu().numpy().astype(np.uint8))
        pred_pil = pred_pil.resize(orig_size, resample=Image.NEAREST)

        ax.imshow(image)
        ax.imshow(np.array(pred_pil), cmap=cmap, vmin=0, vmax=num_classes)
        ax.set_title(os.path.basename(path))
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Image prediction plot saved -> {save_path}")
    plt.show()
