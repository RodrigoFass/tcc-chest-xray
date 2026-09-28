import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from chestxray.config import load_config
from chestxray.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ChestXrayDataset,
    build_dataloader,
    build_dataset,
    build_transforms,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
CLASSES = BASE["data"]["classes"]
N_IMAGES = 10


@pytest.fixture
def cfg(tmp_path):
    """Base config pointing at a tiny synthetic dataset of 256x256 grayscale images."""
    rng = np.random.default_rng(0)
    (tmp_path / "images").mkdir()
    rows = []
    for i in range(N_IMAGES):
        name = f"{i:08d}_000.png"
        Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8), "L").save(tmp_path / "images" / name)
        rows.append({"image": name, "patient_id": i, **{c: int(j == i % len(CLASSES)) for j, c in enumerate(CLASSES)}})
    for split in ("train", "val", "test"):
        pd.DataFrame(rows).to_csv(tmp_path / f"{split}.csv", index=False)

    cfg = copy.deepcopy(BASE)
    cfg["paths"] = {"splits_dir": str(tmp_path), "data_dir": str(tmp_path)}
    cfg["train"].update(batch_size=4, num_workers=0)
    return cfg


def test_item_shapes_and_labels(cfg):
    ds = build_dataset(cfg, "val")
    image, labels, name = ds[3]
    assert image.shape == (3, 224, 224) and image.dtype == torch.float32
    assert labels.shape == (14,) and labels.dtype == torch.float32
    assert name == "00000003_000.png"
    assert labels.tolist() == [float(j == 3) for j in range(14)]
    # The gray image is replicated into 3 channels before per-channel normalization
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    raw = image * std + mean
    assert torch.allclose(raw[0], raw[1], atol=1e-6) and torch.allclose(raw[1], raw[2], atol=1e-6)


def test_eval_transform_is_deterministic_and_normalized(cfg):
    transform = build_transforms(cfg, train=False)
    black = Image.new("L", (256, 256), 0)
    out = transform(black)
    assert torch.equal(out, transform(black))
    expected = torch.tensor([-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
    assert torch.allclose(out.mean(dim=(1, 2)), expected, atol=1e-5)


def test_train_transform_is_random_unless_augmentation_disabled(cfg):
    image = Image.open(Path(cfg["paths"]["data_dir"]) / "images" / "00000000_000.png")
    train = build_transforms(cfg, train=True)
    torch.manual_seed(0)
    assert train(image).shape == (3, 224, 224)
    assert not torch.equal(train(image), train(image))

    cfg["augmentation"]["enabled"] = False
    no_aug = build_transforms(cfg, train=True)
    assert torch.equal(no_aug(image), build_transforms(cfg, train=False)(image))


def test_dataloader_batch(cfg):
    images, labels, names = next(iter(build_dataloader(cfg, "train")))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4, 14) and labels.dtype == torch.float32
    assert len(names) == 4


def test_train_order_is_seeded(cfg):
    first = [n for _, _, batch in build_dataloader(cfg, "train") for n in batch]
    second = [n for _, _, batch in build_dataloader(cfg, "train") for n in batch]
    assert first == second  # same seed, same order
    assert first != sorted(first)  # but shuffled


def test_max_images_caps_each_split(cfg):
    cfg["data"]["max_images"] = 4
    assert len(build_dataset(cfg, "train")) == 4


def test_missing_class_column_raises(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        ChestXrayDataset(pd.DataFrame({"image": ["a.png"]}), tmp_path, CLASSES)
