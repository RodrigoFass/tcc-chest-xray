"""PyTorch Dataset over the split CSVs, with the train and evaluation transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(cfg: dict, train: bool) -> T.Compose:
    """Augmentation for training (if enabled), plain resize otherwise.

    Evaluation resizes the whole image to 224x224 with no CenterCrop: cropping 256 -> 224
    would cut ~6% of each border, where the costophrenic angles (pleural effusion) are.
    The grayscale image is replicated into 3 channels for the ImageNet-pretrained network.
    """
    size = cfg["data"]["image_size"]
    aug = cfg["augmentation"]
    if train and aug["enabled"]:
        steps = [
            T.RandomResizedCrop(size, scale=tuple(aug["crop_scale"]), ratio=tuple(aug["crop_ratio"])),
            T.RandomHorizontalFlip(aug["hflip_p"]),
            T.RandomRotation(aug["rotation_degrees"]),
            T.ColorJitter(brightness=aug["brightness"]),
        ]
    else:
        steps = [T.Resize((size, size))]
    return T.Compose([
        *steps,
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    """Items are ``(image tensor 3xHxW, float label vector, image name)``."""

    def __init__(self, table: pd.DataFrame | str | Path, images_dir: str | Path,
                 classes: list[str], transform=None):
        table = table if isinstance(table, pd.DataFrame) else pd.read_csv(table)
        missing = [c for c in ["image", *classes] if c not in table.columns]
        if missing:
            raise ValueError(f"Columns missing from the split table: {missing}")
        self.images = table["image"].tolist()
        self.labels = torch.from_numpy(table[classes].to_numpy(dtype=np.float32))
        self.images_dir = Path(images_dir)
        self.classes = list(classes)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        with Image.open(self.images_dir / self.images[index]) as img:
            image = img.convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index], self.images[index]


def build_dataset(cfg: dict, split: str) -> ChestXrayDataset:
    """Dataset for ``split`` (train/val/test), capped at ``data.max_images`` if set."""
    table = pd.read_csv(Path(cfg["paths"]["splits_dir"]) / f"{split}.csv")
    max_images = cfg["data"].get("max_images")
    if max_images and len(table) > max_images:
        table = table.sample(n=max_images, random_state=cfg["split"]["seed"]).sort_index()
    return ChestXrayDataset(
        table,
        Path(cfg["paths"]["data_dir"]) / "images",
        cfg["data"]["classes"],
        build_transforms(cfg, train=split == "train"),
    )


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """Shuffled (with a seeded generator) for train, in order for val/test."""
    workers = cfg["train"]["num_workers"]
    return DataLoader(
        build_dataset(cfg, split),
        batch_size=cfg["train"]["batch_size"],
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
