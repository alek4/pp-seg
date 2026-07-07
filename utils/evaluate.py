import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import _default_criterion
from .train import _prepare_loss


def _binarise(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """[B,1,H,W] sigmoid output -> [B,H,W] long mask."""
    return (logits.squeeze(1) >= threshold).long()


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    binary: bool = False,
    num_classes: int = 2,
    threshold: float = 0.5,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluate a segmentation model on a held-out test set.

    Computes per-class and mean IoU, Precision, Recall, F1, and pixel
    accuracy by accumulating TP/FP/FN/TN counts over the full test set.

    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
    binary : bool
        Must match the value used during training.
    num_classes : int
        Ignored in binary mode (always 2).
    threshold : float
        Decision threshold for binary predictions.
    criterion : nn.Module or None
    device : torch.device or None

    Returns
    -------
    dict with test_loss, pixel_accuracy, per_class_iou, mean_iou,
    per_class_precision, mean_precision, per_class_recall, mean_recall,
    per_class_f1, mean_f1, class_labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if binary:
        num_classes = 2

    model = model.to(device).eval()

    if criterion is None:
        criterion = _default_criterion(binary, None, device)

    total_loss = 0.0
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)
    TN = torch.zeros(num_classes)

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)

            logits_loss, masks_loss = _prepare_loss(logits, masks, binary)
            total_loss += criterion(logits_loss, masks_loss).item()

            preds = _binarise(logits, threshold) if binary else torch.argmax(logits, dim=1)

            for c in range(num_classes):
                pred_c  = (preds == c)
                truth_c = (masks == c)
                TP[c] += (pred_c &  truth_c).sum().item()
                FP[c] += (pred_c & ~truth_c).sum().item()
                FN[c] += (~pred_c &  truth_c).sum().item()
                TN[c] += (~pred_c & ~truth_c).sum().item()

    eps = 1e-6
    per_class_iou       = (TP / (TP + FP + FN + eps)).tolist()
    per_class_precision = (TP / (TP + FP + eps)).tolist()
    per_class_recall    = (TP / (TP + FN + eps)).tolist()
    per_class_f1        = (2 * TP / (2 * TP + FP + FN + eps)).tolist()

    total_pix      = (TP + FP + FN + TN).sum().item()
    correct_pix    = (TP + TN).sum().item()
    pixel_accuracy = correct_pix / (total_pix + eps)
    test_loss      = total_loss / len(test_loader)
    mean_iou       = float(np.mean(per_class_iou))
    mean_precision = float(np.mean(per_class_precision))
    mean_recall    = float(np.mean(per_class_recall))
    mean_f1        = float(np.mean(per_class_f1))

    class_labels = (
        ["background", "marestail"]
        if binary
        else ["background"] + [f"class {c}" for c in range(1, num_classes)]
    )

    col = 14
    print("\n-- Test Results --------------------------------------------------")
    print(f"  Loss           : {test_loss:.4f}")
    print(f"  Pixel accuracy : {pixel_accuracy * 100:.2f}%")
    print(f"  {'Class':<{col}}  {'IoU':>6}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print(f"  {'-'*col}  {'------'}  {'---------'}  {'------'}  {'------'}")
    for label, iou, prec, rec, f1 in zip(
        class_labels, per_class_iou, per_class_precision, per_class_recall, per_class_f1
    ):
        print(f"  {label:<{col}}  {iou:>6.4f}  {prec:>9.4f}  {rec:>6.4f}  {f1:>6.4f}")
    print(f"  {'mean':<{col}}  {mean_iou:>6.4f}  {mean_precision:>9.4f}  {mean_recall:>6.4f}  {mean_f1:>6.4f}")
    print("------------------------------------------------------------------\n")

    return {
        "test_loss":           test_loss,
        "pixel_accuracy":      pixel_accuracy,
        "per_class_iou":       per_class_iou,
        "mean_iou":            mean_iou,
        "per_class_precision": per_class_precision,
        "mean_precision":      mean_precision,
        "per_class_recall":    per_class_recall,
        "mean_recall":         mean_recall,
        "per_class_f1":        per_class_f1,
        "mean_f1":             mean_f1,
        "class_labels":        class_labels,
    }


def find_optimal_threshold(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    n_thresholds: int = 50,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    all_probs = []
    all_masks = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Collecting predictions"):
            images = images.to(device)
            probs = torch.sigmoid(model(images).squeeze(1))
            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    foreground = all_masks == 1

    best_iou, best_t = -1.0, 0.5
    for t in torch.linspace(0.1, 0.9, n_thresholds):
        pred_fg = all_probs >= t
        tp = ( pred_fg &  foreground).sum().item()
        fp = ( pred_fg & ~foreground).sum().item()
        fn = (~pred_fg &  foreground).sum().item()
        iou = tp / (tp + fp + fn + 1e-6)
        if iou > best_iou:
            best_iou, best_t = iou, t.item()

    print(f"Optimal threshold: {best_t:.3f}  (foreground IoU: {best_iou:.4f})")
    return {"threshold": best_t, "foreground_iou": best_iou}


def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    n_runs: int = 100,
    n_warmup: int = 10,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    # grab one batch — data loading is not part of the measurement
    images, _ = next(iter(loader))
    images = images.to(device)
    batch_size = images.shape[0]

    # warmup: burns off CUDA kernel compilation on first use
    with torch.no_grad():
        for _ in range(n_warmup):
            model(images)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            # synchronize before and after so we time GPU work, not kernel launch
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    mean_batch = float(np.mean(times_ms))
    std_batch  = float(np.std(times_ms))
    mean_image = mean_batch / batch_size

    print(f"Inference time ({n_runs} runs, batch_size={batch_size}, device={device})")
    print(f"  per batch : {mean_batch:.2f} ± {std_batch:.2f} ms")
    print(f"  per image : {mean_image:.2f} ms")

    return {
        "mean_ms_per_batch": mean_batch,
        "std_ms_per_batch":  std_batch,
        "mean_ms_per_image": mean_image,
        "batch_size":        batch_size,
    }


def evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    *,
    title: str = "Model",
    n_samples: int = 4,
    device: Optional[torch.device] = None,
) -> dict:
    """Standard evaluation protocol for a trained binary checkpoint.

    Loads the weights, tunes the decision threshold on valid, computes
    test metrics at that threshold, measures parameter count and
    inference time, and plots predictions (at the tuned threshold) and
    per-class metrics.

    Returns the test_model dict extended with threshold, n_params and
    the measure_inference_time entries.
    """
    from .viz import plot_metrics, plot_predictions  # deferred: viz imports from this module

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    threshold = find_optimal_threshold(model, valid_loader, device=device)["threshold"]
    metrics = test_model(model, test_loader, binary=True, threshold=threshold, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    timing = measure_inference_time(model, test_loader, device=device)

    plot_predictions(
        model, test_loader,
        binary=True,
        threshold=threshold,
        n_samples=n_samples,
        device=device,
        title=f"{title} – Predictions vs Ground Truth",
    )
    plot_metrics(metrics, title=f"{title} – Test Metrics")

    return {**metrics, "threshold": threshold, "n_params": n_params, **timing}
