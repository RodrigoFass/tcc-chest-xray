"""Grad-CAM heatmaps (Phase 4): gallery of hits and errors, radiologist boxes, pointing game.

    python -m chestxray.gradcam --config configs/experiments/e1_baseline.yaml

Needs the trained ``best.pt``, the preprocessed images and, from the test evaluation,
``preds_test.csv`` and ``calibration.json`` (the per-class threshold chosen on validation).
Writes, for the experiment:

- ``results/figures/<exp>/gradcam/galeria_<classe>``: for each focus class, true positives,
  false positives and false negatives at the validation threshold. The images are picked by
  a fixed rule (:func:`select_gallery`), not by hand: the highest scores among true and false
  positives and the lowest scores among false negatives, one image per patient;
- ``.../camadas_foco``: the same images with the two candidate target layers side by side;
- ``.../caixas_<classe>``: every test image that has a radiologist box for the class, with
  the heatmap, the box and the heatmap's peak;
- ``results/tables/<exp>/pointing_game``: share of boxes that contain the heatmap's peak, per
  class and target layer, with a 95% Wilson interval and the share hit by the image centre
  (a trivial baseline);
- ``results/tables/<exp>/gradcam_selecao.csv``: the images of the gallery and why they were chosen.

Target layers (plan 3.11): ``relu`` is the ReLU after ``norm5``, the last feature map before the
global pooling. With pooling followed by a single linear layer, Grad-CAM on that map equals the
CAM used by CheXNet (Zhou et al., 2016) up to a positive factor. ``denseblock4`` is the same
map before the final batch norm and ReLU; it is the alternative the plan asks to compare, and
the pointing game made it the default (``DEFAULT_LAYER``), used by the gallery, the box figures
and the demo app.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T

from chestxray.config import load_config
from chestxray.inference import load_trained_model, predict_image, prepare_image
from chestxray.plotting import br_number, pt, save_figure, save_table, setup_style
from chestxray.utils import get_device, setup_logging

logger = logging.getLogger(__name__)

TARGET_LAYERS = {
    "relu": lambda model: model.relu,
    "denseblock4": lambda model: model.features.denseblock4,
}
# Chosen by the pointing game on the E1 test boxes (plan, section 10): denseblock4 hit 15 of 62
# images of the focus classes against 13 for relu; the difference is not significant and the
# maps look alike, so relu (the CAM of CheXNet) stays in every comparison
DEFAULT_LAYER = "denseblock4"

# BBox_List_2017.csv: coordinates refer to the 1024x1024 images of the dataset, and the file
# calls Infiltration "Infiltrate" (plan, Phase 4)
BBOX_FILE = "BBox_List_2017.csv"
BBOX_IMAGE_SIZE = 1024
BBOX_LABELS = {"Infiltrate": "Infiltration"}

GROUPS = {"tp": "Verdadeiros positivos", "fp": "Falsos positivos", "fn": "Falsos negativos"}
HEATMAP_ALPHA = 0.4
HEATMAP_CMAP = "jet"


@dataclass
class Explanation:
    """One class's score for one image and its heatmap, on the network's input grid."""

    class_name: str
    score: float
    cam: np.ndarray       # HxW in [0, 1]
    base: np.ndarray      # HxW grayscale in [0, 1]: the image the network saw, before normalization
    layer: str

    @property
    def overlay(self) -> np.ndarray:
        return overlay_heatmap(self.base, self.cam)

    @property
    def peak(self) -> tuple[float, float]:
        """(x, y) of the heatmap's maximum, at the centre of the pixel."""
        row, col = np.unravel_index(np.argmax(self.cam), self.cam.shape)
        return col + 0.5, row + 0.5


def overlay_heatmap(base: np.ndarray, cam: np.ndarray, alpha: float = HEATMAP_ALPHA) -> np.ndarray:
    """RGB uint8 image: the grayscale X-ray with the heatmap blended on top."""
    colors = plt.get_cmap(HEATMAP_CMAP)(cam)[..., :3]
    gray = np.repeat(base[..., None], 3, axis=2)
    return np.uint8(255 * np.clip((1 - alpha) * gray + alpha * colors, 0, 1))


def network_view(image: Image.Image, cfg: dict) -> np.ndarray:
    """The grayscale image exactly as resized for the network (without the normalization)."""
    size, stored = cfg["data"]["image_size"], cfg["data"]["stored_size"]
    image = image.convert("L")
    if image.size != (stored, stored):
        image = image.resize((stored, stored), Image.Resampling.LANCZOS)
    return np.asarray(T.Resize((size, size))(image), dtype=np.float32) / 255.0


class Explainer:
    """Scores and Grad-CAM heatmaps from a trained model, one image at a time.

    Scores come from :func:`chestxray.inference.predict_image`, the same function behind
    ``preds_*.csv`` and the demo app, so the numbers shown next to a heatmap are the
    evaluated ones.
    """

    def __init__(self, model: torch.nn.Module, cfg: dict, layers: list[str] | None = None):
        self.model, self.cfg = model.eval(), cfg
        self.classes = cfg["data"]["classes"]
        self.cams = {}
        for layer in layers or [DEFAULT_LAYER]:
            if layer not in TARGET_LAYERS:
                raise ValueError(f"Unknown target layer {layer!r}; choose from {sorted(TARGET_LAYERS)}")
            self.cams[layer] = GradCAM(model=model, target_layers=[TARGET_LAYERS[layer](model)])

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: torch.device | str = "cpu",
                        layers: list[str] | None = None) -> "Explainer":
        model, cfg = load_trained_model(checkpoint, torch.device(device))
        return cls(model, cfg, layers)

    def scores(self, image: Image.Image) -> np.ndarray:
        return predict_image(self.model, self.cfg, image)

    def heatmap(self, image: Image.Image, class_name: str, layer: str = DEFAULT_LAYER) -> np.ndarray:
        device = next(self.model.parameters()).device
        x = prepare_image(image, self.cfg).unsqueeze(0).to(device)
        target = [ClassifierOutputTarget(self.classes.index(class_name))]
        cam = self.cams[layer](input_tensor=x, targets=target)[0]
        return np.nan_to_num(cam.astype(np.float32))

    def explain(self, image: Image.Image | str | Path, class_name: str, layer: str = DEFAULT_LAYER,
                scores: np.ndarray | None = None) -> Explanation:
        if not isinstance(image, Image.Image):
            with Image.open(image) as img:
                image = img.convert("L")
        if scores is None:
            scores = self.scores(image)
        return Explanation(class_name, float(scores[self.classes.index(class_name)]),
                           self.heatmap(image, class_name, layer), network_view(image, self.cfg), layer)

    def close(self) -> None:
        for cam in self.cams.values():
            cam.activations_and_grads.release()


# ----------------------------------------------------------------------------- selection

def select_gallery(preds: pd.DataFrame, class_name: str, threshold: float, n: int = 3) -> pd.DataFrame:
    """Gallery images of one class, by a fixed rule (so the figures are not hand-picked).

    At the validation threshold: true positives and false positives with the highest scores,
    false negatives with the lowest scores; at most one image per patient; ties broken by
    image name. Returns the rows with a ``group`` column (tp / fp / fn) and the rank.
    """
    score, label = preds[f"score_{class_name}"], preds[class_name].astype(bool)
    above = score >= threshold
    groups = {"tp": (label & above, False), "fp": (~label & above, False), "fn": (label & ~above, True)}
    chosen = []
    for group, (mask, ascending) in groups.items():
        rows = preds[mask].sort_values([f"score_{class_name}", "image"], ascending=[ascending, True])
        rows = rows.drop_duplicates("patient_id").head(n).copy()
        rows["group"], rows["rank"] = group, range(1, len(rows) + 1)
        chosen.append(rows)
    columns = ["group", "rank", "image", "patient_id", class_name, f"score_{class_name}"]
    return pd.concat(chosen, ignore_index=True)[columns]


def load_boxes(data_dir: str | Path, image_size: int, classes: list[str]) -> pd.DataFrame:
    """Radiologist boxes scaled to the network's input grid: image, class, x, y, w, h.

    The file has a malformed header ("Bbox [x", "y", "w", "h]" and empty trailing columns),
    so columns are taken by position.
    """
    raw = pd.read_csv(Path(data_dir) / BBOX_FILE)
    boxes = raw.iloc[:, :6].copy()
    boxes.columns = ["image", "class", "x", "y", "w", "h"]
    boxes["class"] = boxes["class"].str.strip().replace(BBOX_LABELS)
    unknown = set(boxes["class"]) - set(classes)
    if unknown:
        raise ValueError(f"{BBOX_FILE}: labels not among the classes: {sorted(unknown)}")
    scale = image_size / BBOX_IMAGE_SIZE
    boxes[["x", "y", "w", "h"]] = boxes[["x", "y", "w", "h"]].astype(float) * scale
    return boxes


def point_in_box(point: tuple[float, float], box) -> bool:
    x, y = point
    return bool(box.x <= x <= box.x + box.w and box.y <= y <= box.y + box.h)


def wilson_interval(hits: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval of a proportion (well behaved with few cases)."""
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def pointing_game(results: pd.DataFrame, image_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hit rate per class and layer from one row per (box, layer) with a ``hit`` column.

    An image with several boxes of the same class counts once, as a hit if the peak falls in
    any of them. The centre baseline asks the same of the image centre.
    """
    centre = (image_size / 2, image_size / 2)
    per_image = (results.assign(centre_hit=[point_in_box(centre, b) for b in results.itertuples()])
                 .groupby(["class", "layer", "image"], as_index=False)[["hit", "centre_hit"]].max())
    rows, display = [], []
    for (class_name, layer), g in per_image.groupby(["class", "layer"], sort=False):
        n, hits, centre_hits = len(g), int(g["hit"].sum()), int(g["centre_hit"].sum())
        lo, hi = wilson_interval(hits, n)
        rows.append({"classe": class_name, "camada": layer, "n_imagens": n, "acertos": hits,
                     "taxa": hits / n, "ic_inf": lo, "ic_sup": hi, "taxa_centro": centre_hits / n})
        display.append({"Doença": pt(class_name), "Camada": layer, "Imagens": n,
                        "Acertos": hits,
                        "Taxa (IC95%)": f"{br_number(100 * hits / n, 1)}% "
                                        f"({br_number(100 * lo, 1)}–{br_number(100 * hi, 1)}%)",
                        "Centro da imagem": f"{br_number(100 * centre_hits / n, 1)}%"})
    return pd.DataFrame(rows), pd.DataFrame(display)


# ----------------------------------------------------------------------------- figures

def _show(ax, explanation: Explanation, title: str) -> None:
    ax.imshow(explanation.overlay)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_gallery(class_name: str, items: list[tuple[str, Explanation, str]], n: int) -> plt.Figure:
    """Rows: true positives, false positives, false negatives; ``n`` columns each."""
    fig, axes = plt.subplots(len(GROUPS), n, figsize=(2.9 * n, 3.2 * len(GROUPS)), squeeze=False)
    for i, group in enumerate(GROUPS):
        row = [(e, img) for g, e, img in items if g == group]
        for j in range(n):
            ax = axes[i, j]
            if j < len(row):
                explanation, image = row[j]
                _show(ax, explanation, f"{image}\nescore {br_number(explanation.score, 3)}")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "sem casos", ha="center", va="center", transform=ax.transAxes)
        axes[i, 0].set_ylabel(GROUPS[group], fontsize=10)
        axes[i, 0].axis("on")
    fig.suptitle(f"Grad-CAM: {pt(class_name)} (limiar de Youden da validação)")
    fig.tight_layout()
    return fig


def plot_layers(rows: list[tuple[str, str, dict[str, Explanation]]]) -> plt.Figure:
    """One row per (class, image): the input image, then one heatmap per target layer."""
    layers = list(rows[0][2])
    fig, axes = plt.subplots(len(rows), 1 + len(layers), figsize=(3 * (1 + len(layers)), 3.2 * len(rows)),
                             squeeze=False)
    for i, (class_name, image, by_layer) in enumerate(rows):
        base = next(iter(by_layer.values())).base
        axes[i, 0].imshow(base, cmap="gray")
        axes[i, 0].set_title(f"{pt(class_name)}\n{image}", fontsize=8)
        for j, layer in enumerate(layers, start=1):
            _show(axes[i, j], by_layer[layer], f"camada {layer}")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Grad-CAM nas duas camadas-alvo candidatas")
    fig.tight_layout()
    return fig


def plot_boxes(class_name: str, items: list[tuple[str, Explanation, list]], columns: int = 5) -> plt.Figure:
    """Heatmap, radiologist box(es) and the heatmap's peak (green: inside a box; red: outside)."""
    n_rows = max(1, math.ceil(len(items) / columns))
    fig, axes = plt.subplots(n_rows, columns, figsize=(2.7 * columns, 2.9 * n_rows), squeeze=False)
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    for ax, (image, explanation, boxes) in zip(axes.flat, items):
        hit = any(point_in_box(explanation.peak, b) for b in boxes)
        _show(ax, explanation, f"{image}\nescore {br_number(explanation.score, 3)}")
        for b in boxes:
            ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, edgecolor="white", lw=1.5))
        ax.plot(*explanation.peak, marker="x", ms=9, mew=2.5, color="lime" if hit else "red")
    fig.suptitle(f"{pt(class_name)}: caixas do radiologista (branco) e pico do Grad-CAM\n"
                 f"(verde: dentro da caixa; vermelho: fora)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- run

def run(cfg: dict, layers: list[str], device: torch.device, n_per_group: int = 3,
        with_boxes: bool = True) -> dict:
    name, focus = cfg["experiment"], cfg["data"]["focus_classes"]
    run_dir = Path(cfg["paths"]["runs_dir"]) / name
    results_dir = Path(cfg["paths"]["results_dir"])
    figures_dir, tables_dir = results_dir / "figures" / name / "gradcam", results_dir / "tables" / name
    images_dir = Path(cfg["paths"]["data_dir"]) / "images"
    calibration_path = run_dir / "calibration.json"
    if not calibration_path.exists():
        raise SystemExit(f"{calibration_path} not found: run the test evaluation first "
                         f"(python -m chestxray.evaluate --run {run_dir} --split test)")
    thresholds = {c: v["threshold"] for c, v in json.loads(calibration_path.read_text(encoding="utf-8"))["classes"].items()}
    preds = pd.read_csv(run_dir / "preds_test.csv")

    explainer = Explainer.from_checkpoint(Path(cfg["paths"]["checkpoint_dir"]) / name / "best.pt", device, layers)
    trained_classes = explainer.classes
    if trained_classes != cfg["data"]["classes"]:
        raise SystemExit("The checkpoint was trained with another class list than the config")
    setup_style()
    summary = {"experiment": name, "layers": layers, "gallery": {}, "pointing_game": None}

    def load(image: str) -> Image.Image:
        with Image.open(images_dir / image) as img:
            return img.convert("L")

    selections, layer_rows = [], []
    for class_name in focus:
        selection = select_gallery(preds, class_name, thresholds[class_name], n_per_group)
        selection.insert(0, "classe", class_name)
        selections.append(selection)
        items = []
        for row in selection.itertuples():
            image = load(row.image)
            explanation = explainer.explain(image, class_name, layers[0])
            items.append((row.group, explanation, row.image))
            if row.group == "tp" and row.rank == 1 and len(layers) > 1:
                layer_rows.append((class_name, row.image,
                                   {layer: explainer.explain(image, class_name, layer) for layer in layers}))
        fig = plot_gallery(class_name, items, n_per_group)
        save_figure(fig, figures_dir / f"galeria_{class_name.lower()}")
        plt.close(fig)
        summary["gallery"][class_name] = selection["group"].value_counts().to_dict()
        logger.info("%s: gallery with %s", class_name, summary["gallery"][class_name])
    if layer_rows:
        fig = plot_layers(layer_rows)
        save_figure(fig, figures_dir / "camadas_foco")
        plt.close(fig)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(selections, ignore_index=True).to_csv(tables_dir / "gradcam_selecao.csv", index=False)

    if with_boxes:
        boxes = load_boxes(cfg["paths"]["data_dir"], cfg["data"]["image_size"], trained_classes)
        boxes = boxes[boxes["image"].isin(set(preds["image"]))]
        logger.info("%d boxes on %d test images", len(boxes), boxes["image"].nunique())
        records = []
        for class_name, class_boxes in boxes.groupby("class", sort=False):
            figure_items = []
            for image_name, image_boxes in class_boxes.groupby("image", sort=True):
                image = load(image_name)
                scores = explainer.scores(image)
                for layer in layers:
                    explanation = explainer.explain(image, class_name, layer, scores)
                    for b in image_boxes.itertuples():
                        records.append({"image": image_name, "class": class_name, "layer": layer,
                                        "x": b.x, "y": b.y, "w": b.w, "h": b.h,
                                        "hit": point_in_box(explanation.peak, b)})
                    if layer == layers[0] and class_name in focus:
                        figure_items.append((image_name, explanation, list(image_boxes.itertuples())))
            if figure_items:
                fig = plot_boxes(class_name, figure_items)
                save_figure(fig, figures_dir / f"caixas_{class_name.lower()}")
                plt.close(fig)
        if records:
            results = pd.DataFrame(records)
            results.to_csv(tables_dir / "pointing_game_caixas.csv", index=False)
            numeric, display = pointing_game(results, cfg["data"]["image_size"])
            save_table(numeric, display, tables_dir / "pointing_game")
            summary["pointing_game"] = numeric.to_dict(orient="records")
            logger.info("Pointing game:\n%s", display.to_string(index=False))
    explainer.close()
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment config used for training")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    parser.add_argument("--layers", nargs="+", choices=list(TARGET_LAYERS),
                        default=[DEFAULT_LAYER] + [layer for layer in TARGET_LAYERS if layer != DEFAULT_LAYER],
                        help="target layers; the first one is used for the gallery and box figures")
    parser.add_argument("--per-group", type=int, default=3, help="images per group (TP, FP, FN) in the gallery")
    parser.add_argument("--no-boxes", action="store_true", help="skip the radiologist boxes and pointing game")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.paths)
    run(cfg, args.layers, get_device(args.device), args.per_group, not args.no_boxes)


if __name__ == "__main__":
    main()
