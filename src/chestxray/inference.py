"""Predictions from a trained model: the one place where an image is prepared and scored.

    python -m chestxray.inference --config configs/experiments/e1_baseline.yaml --splits val
    python -m chestxray.inference --config configs/experiments/e1_baseline.yaml --splits test

Loads ``<checkpoint_dir>/<experiment>/best.pt`` and writes
``<runs_dir>/<experiment>/preds_<split>.csv``: one row per image with the patient, age, sex
and view (for the patient bootstrap and the subgroup analysis), the true labels, and the
logit and sigmoid score of every class. Evaluation then runs from these files alone, on CPU.
Inference is always fp32 (no mixed precision). Grad-CAM and the demo app score images
through :func:`prepare_image` and :func:`predict_image`, so they match these files.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from chestxray.config import load_config
from chestxray.data.dataset import build_dataset, build_transforms
from chestxray.data.split import META_COLUMNS
from chestxray.models.densenet import build_model
from chestxray.train import comparable
from chestxray.utils import get_device, setup_logging

logger = logging.getLogger(__name__)

FLOAT_FORMAT = "%.6g"


def load_trained_model(checkpoint: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Model in eval mode with the checkpoint's weights, and the config it was trained with."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(state["config"])
    cfg["model"]["pretrained"] = False  # the weights come from the checkpoint, not from ImageNet
    model = build_model(cfg)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state["config"]


def prepare_image(image: Image.Image, cfg: dict) -> torch.Tensor:
    """Any chest X-ray image -> network input (3xHxW), exactly as in evaluation: grayscale,
    reduced to the stored size with Lanczos as in preprocess.py, then the eval transform."""
    size = cfg["data"]["stored_size"]
    image = image.convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return build_transforms(cfg, train=False)(image)


@torch.no_grad()
def predict_image(model: torch.nn.Module, cfg: dict, image: Image.Image) -> np.ndarray:
    """Sigmoid scores of all classes for one image (fp32)."""
    device = next(model.parameters()).device
    logits = model(prepare_image(image, cfg).unsqueeze(0).to(device)).float()
    return torch.sigmoid(logits)[0].cpu().numpy()


@torch.no_grad()
def predict_split(model: torch.nn.Module, cfg: dict, split: str, batch_size: int = 64,
                  num_workers: int = 0) -> pd.DataFrame:
    """Logits and scores for every image of a split, with its metadata and true labels."""
    classes = cfg["data"]["classes"]
    dataset = build_dataset(cfg, split)
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    logits = []
    for i, (images, _, _) in enumerate(loader, start=1):
        logits.append(model(images.to(device, non_blocking=True)).float().cpu())
        if i % 50 == 0:
            logger.info("%s: %d/%d images", split, min(i * batch_size, len(dataset)), len(dataset))
    logits = torch.cat(logits).numpy()

    table = pd.read_csv(Path(cfg["paths"]["splits_dir"]) / f"{split}.csv").set_index("image")
    preds = table.loc[dataset.images, [*META_COLUMNS[1:], *classes]].reset_index()
    scores = 1.0 / (1.0 + np.exp(-logits))
    preds = pd.concat([
        preds,
        pd.DataFrame(logits, columns=[f"logit_{c}" for c in classes]),
        pd.DataFrame(scores, columns=[f"score_{c}" for c in classes]),
    ], axis=1)
    return preds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment config used for training")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = load_config(args.config, args.paths)
    run_dir = Path(cfg["paths"]["runs_dir"]) / cfg["experiment"]
    trained = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if comparable(trained) != comparable(cfg):
        raise SystemExit(f"{args.config} differs from the config {run_dir} was trained with")

    device = get_device(args.device)
    model, _ = load_trained_model(Path(cfg["paths"]["checkpoint_dir"]) / cfg["experiment"] / "best.pt", device)
    for split in args.splits:
        preds = predict_split(model, cfg, split, num_workers=cfg["train"]["num_workers"])
        path = run_dir / f"preds_{split}.csv"
        preds.to_csv(path, index=False, float_format=FLOAT_FORMAT)
        logger.info("Wrote %s (%d images)", path, len(preds))


if __name__ == "__main__":
    main()
