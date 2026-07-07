import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader, Dataset

DATA_DIR    = "greenhouse-3-1"
TARGET_SIZE = (512, 512)
STATS_PATH  = "outputs/norm_stats.json"


class GreenhouseDataset(Dataset):

    def __init__(self, split, dataset_location=DATA_DIR, target_size=TARGET_SIZE, augment=None, transform=None):
        self.split_path  = os.path.join(dataset_location, split)
        self.target_size = target_size
        self.augment     = augment if augment is not None else (split == "train")
        self.transform   = transform

        with open(os.path.join(self.split_path, "_annotations.coco.json")) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = {}
        for ann in coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

    def _augment(self, image, mask):
        # Horizontal flip
        if random.random() > 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        # Vertical flip
        if random.random() > 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        # Rotation -- same angle applied to both
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask,  angle, interpolation=T.InterpolationMode.NEAREST)
        # Color jitter on image only (mask holds class IDs, not colour)
        image = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(image)
        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image    = Image.open(os.path.join(self.split_path, img_info["file_name"])).convert("RGB")
        orig_w, orig_h = image.size

        # Build mask from COCO annotations
        mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in self.annotations.get(img_info["id"], []):
            rle = coco_mask.frPyObjects(ann["segmentation"], orig_h, orig_w)
            m   = coco_mask.decode(rle)
            if m.ndim == 3:
                m = m[:, :, 0]
            mask_np[m > 0] = ann["category_id"]

        # Resize both to target size
        image = TF.resize(image, self.target_size)
        mask  = TF.resize(Image.fromarray(mask_np), self.target_size,
                          interpolation=T.InterpolationMode.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = TF.to_tensor(image)
        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array(mask)).long()


def make_loaders(dataset_location=DATA_DIR, batch_size=8, num_workers=2, transform=None):
    loaders = {}
    for split in ["train", "valid", "test"]:
        ds = GreenhouseDataset(split, dataset_location, transform=transform)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        tag = "[augmented]" if split == "train" else ""
        print(f"{split:>5}: {len(ds):>4} images | {len(loaders[split]):>3} batches {tag}")

    return loaders["train"], loaders["valid"], loaders["test"]


def norm_transform(dataset_location=DATA_DIR, stats_path=STATS_PATH, batch_size=16, num_workers=2):
    """T.Normalize with per-channel mean/std of the train split.

    Stats are computed once (two passes: mean, then std) and cached at
    stats_path; later calls load the JSON instead of sweeping the data.
    """
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        ds = GreenhouseDataset("train", dataset_location, augment=False)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

        pixel_sum = torch.zeros(3)
        n_pixels = 0
        for images, _ in loader:
            pixel_sum += images.sum(dim=[0, 2, 3])
            n_pixels  += images.shape[0] * images.shape[2] * images.shape[3]
        mean = pixel_sum / n_pixels

        sq_diff_sum = torch.zeros(3)
        for images, _ in loader:
            sq_diff_sum += ((images - mean[None, :, None, None]) ** 2).sum(dim=[0, 2, 3])
        std = (sq_diff_sum / n_pixels).sqrt()

        stats = {"mean": mean.tolist(), "std": std.tolist()}
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f)

    print(f"norm stats | mean: {stats['mean']} | std: {stats['std']}")
    return T.Normalize(mean=stats["mean"], std=stats["std"])
