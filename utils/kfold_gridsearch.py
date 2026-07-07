"""
kfold_gridsearch.py
===================
3-fold cross-validation grid search over:
    lr, batch_size, base_filters, depth

Selection metric: plant (marestail) IoU on the fold's validation split,
averaged across folds. Results are written to CSV incrementally after
each config and sorted at the end.

Does NOT retrain the final model -- use the best config manually.

Usage
-----
from utils.data import GreenhouseDataset, norm_transform
from utils.kfold_gridsearch import run_grid_search

train_ds = GreenhouseDataset("train", augment=False)
valid_ds = GreenhouseDataset("valid", augment=False)
results_df = run_grid_search([train_ds, valid_ds], transform=norm_transform())
"""

import itertools
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset

from models.BinaryUNet import BinaryUNet
from utils import BCEDiceLoss, train, test_model


PARAM_GRID = {
    "lr":           [1e-3, 1e-4],
    "batch_size":   [4, 8],
    "base_filters": [16, 32],
    "depth":        [2, 3],
}

N_FOLDS      = 3
FOLD_EPOCHS  = 10
RESULTS_PATH = "outputs/results/grid_search_results.csv"


def run_grid_search(
    datasets,
    transform=None,
    param_grid: dict = PARAM_GRID,
    n_folds: int = N_FOLDS,
    fold_epochs: int = FOLD_EPOCHS,
    seed: int = 42,
    results_path: str = RESULTS_PATH,
) -> pd.DataFrame:
    """Run k-fold CV grid search and print a results table.

    Parameters
    ----------
    datasets : sequence of Dataset
        Un-augmented, un-normalized datasets merged into one pool and
        re-split by KFold (pass train + valid; test is never touched).
        Augmentation is re-applied per fold on the training portion.
    transform : callable or None
        Applied to the image after augmentation (e.g. normalization),
        matching the order used in GreenhouseDataset. Must be None on
        the input datasets themselves, or it would run before
        augmentation and break the PIL round-trip.
    param_grid : dict
        Hyperparameter grid.
    n_folds : int
        Number of CV folds.
    fold_epochs : int
        Training epochs per fold (keep low for speed).
    seed : int
    results_path : str
        CSV written incrementally after each config, sorted at the end.

    Returns
    -------
    pd.DataFrame sorted by mean_plant_iou descending.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    full_ds = ConcatDataset(datasets)
    indices = list(range(len(full_ds)))

    combos = [
        dict(zip(param_grid.keys(), vals))
        for vals in itertools.product(*param_grid.values())
    ]
    print(f"Grid: {len(combos)} configs x {n_folds} folds = {len(combos) * n_folds} runs\n")

    kfold   = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []

    for cfg_idx, cfg in enumerate(combos):
        print(f"{'='*55}")
        print(f"Config {cfg_idx+1}/{len(combos)}: {cfg}")

        fold_ious = []
        t0 = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            print(f"  Fold {fold_idx+1}/{n_folds} ", end="", flush=True)

            # Wrap fold indices -- training fold gets augmentation via
            # a thin Dataset that toggles it on the underlying samples
            fold_train = _FoldDataset(full_ds, train_idx, augment=True,  transform=transform)
            fold_valid = _FoldDataset(full_ds, val_idx,   augment=False, transform=transform)

            train_loader = DataLoader(fold_train, batch_size=cfg["batch_size"],
                                      shuffle=True,  num_workers=2, pin_memory=True)
            valid_loader = DataLoader(fold_valid, batch_size=cfg["batch_size"],
                                      shuffle=False, num_workers=2, pin_memory=True)

            model = BinaryUNet(
                base_filters=cfg["base_filters"],
                depth=cfg["depth"],
            )

            train(
                model, train_loader, valid_loader,
                epochs=fold_epochs,
                lr=cfg["lr"],
                binary=True,
                criterion=BCEDiceLoss(alpha=0.5),
                save_path=None,
                device=device,
            )

            metrics   = test_model(model, valid_loader, binary=True, device=device)
            iou       = metrics["per_class_iou"][1]   # plant (marestail) IoU
            fold_ious.append(iou)
            print(f"-> plant IoU: {iou:.4f}")

        mean_iou = float(np.mean(fold_ious))
        std_iou  = float(np.std(fold_ious))
        print(f"  Mean: {mean_iou:.4f} +/- {std_iou:.4f}  ({time.time()-t0:.0f}s)\n")

        results.append({
            **cfg,
            "mean_plant_iou": mean_iou,
            "std_plant_iou":  std_iou,
        })

        pd.DataFrame(results).sort_values("mean_plant_iou", ascending=False).to_csv(results_path, index=False)
        print(f"  Checkpoint saved ({len(results)}/{len(combos)} configs done)")

    results_df = (
        pd.DataFrame(results)
        .sort_values("mean_plant_iou", ascending=False)
        .reset_index(drop=True)
    )
    results_df.to_csv(results_path, index=False)

    print("\n" + "="*55)
    print("Results (sorted by mean plant IoU across folds):\n")
    print(results_df.to_string(index=False))
    print(f"\nBest config:")
    best = results_df.iloc[0]
    for k in param_grid.keys():
        print(f"  {k}: {best[k]}")
    print(f"  mean_plant_iou: {best['mean_plant_iou']:.4f} +/- {best['std_plant_iou']:.4f}")

    return results_df


# ---------------------------------------------------------------------------
# Thin wrapper that re-applies augmentation on a subset of a ConcatDataset.
# We can't use GreenhouseDataset directly because KFold gives us indices
# into the merged pool, not split-level indices. The augmentation logic
# is kept identical to GreenhouseDataset._augment, and `transform`
# (normalization) runs after it, matching GreenhouseDataset's order.
# ---------------------------------------------------------------------------

class _FoldDataset(torch.utils.data.Dataset):
    """Subset with augmentation, matching GreenhouseDataset._augment exactly."""

    def __init__(self, dataset, indices, augment: bool = False, transform=None):
        self.dataset   = dataset
        self.indices   = indices
        self.augment   = augment
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, mask = self.dataset[self.indices[idx]]

        if self.augment:
            # Convert tensors back to PIL for TF ops (matches GreenhouseDataset)
            image_pil = T.ToPILImage()(image)
            mask_pil  = T.ToPILImage()(mask.unsqueeze(0).byte())

            if random.random() > 0.5:
                image_pil = TF.hflip(image_pil)
                mask_pil  = TF.hflip(mask_pil)
            if random.random() > 0.5:
                image_pil = TF.vflip(image_pil)
                mask_pil  = TF.vflip(mask_pil)

            angle     = random.uniform(-15, 15)
            image_pil = TF.rotate(image_pil, angle, interpolation=T.InterpolationMode.BILINEAR)
            mask_pil  = TF.rotate(mask_pil,  angle, interpolation=T.InterpolationMode.NEAREST)

            image_pil = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(image_pil)

            image = TF.to_tensor(image_pil)
            mask  = torch.from_numpy(np.array(mask_pil)).long().squeeze()

        if self.transform:
            image = self.transform(image)

        return image, mask
