"""External validation on CheXpert (Phase 7): the NIH-trained model, unchanged, on another hospital.

    python -m chestxray.external --config configs/experiments/e1_baseline.yaml --chexpert-root E:/datasets/chexpert
    python -m chestxray.external --config configs/experiments/e1_baseline.yaml --evaluate-only

Uses only CheXpert's **validation set**: its labels are the majority vote of three radiologists
and have no "uncertain" value, and nothing is trained on CheXpert (plan, Phase 7). Only frontal
images are scored. ``--chexpert-root`` may hold either layout:

- the original download (``CheXpert-v1.0`` or ``CheXpert-v1.0-small``, with ``valid.csv``), in
  the folder itself or one level below;
- the one Stanford AIMI distributes today (``python -m chestxray.data.download_chexpert``): the
  validation images of CheXpert Plus in ``PNG_valid/`` and the radiologist labels rebuilt from
  CheXlocalize's ground-truth annotations (``CheXlocalize/gt_annotations_val.json``), which list
  each image's positive observations. CheXlocalize does not annotate Pneumonia, so that class
  is left out.

Steps:

1. Predictions (needs the checkpoint and the CheXpert images, GPU optional):
   ``results/runs/<exp>/preds_chexpert.csv``, in the same format as ``preds_test.csv``, with the
   CheXpert labels renamed to the NIH classes they correspond to (:data:`CLASS_MAP`).
2. Evaluation (CPU, from the CSV): AUC with patient-bootstrap CI for the classes both datasets
   have, next to the same classes on the NIH test set, in
   ``results/tables/<exp>/validacao_externa_chexpert`` and
   ``results/figures/<exp>/roc_foco_chexpert``.

CheXpert images are not square (the NIH ones are 1024x1024). By default they are resized to a
square like any other input (``--fit resize``, the same thing the demo app does); ``--fit pad``
pads the short side with black instead, keeping the proportions. The validation set is small
(a few hundred images) and has very few pneumonia cases: its interval is reported, but no
conclusion should rest on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import roc_curve

from chestxray.config import load_config
from chestxray.evaluate import FOCUS_COLORS, Predictions, bootstrap_auc_ap, fmt, fmt_ci
from chestxray.inference import FLOAT_FORMAT, load_trained_model, prepare_image
from chestxray.plotting import br_number, pt, save_figure, save_table, setup_style, use_decimal_comma
from chestxray.utils import get_device, setup_logging

logger = logging.getLogger(__name__)

# NIH class -> CheXpert column (the observations both datasets label)
CLASS_MAP = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
}
PREDS_NAME = "preds_chexpert.csv"
CSV_NAME = "valid.csv"
MIN_POSITIVES_FOR_CONCLUSIONS = 30

# CheXpert as Stanford AIMI distributes it on Redivis today (plan, section 10), without valid.csv:
# the validation images come from CheXpert Plus (PNG_valid/) and the radiologist labels from
# CheXlocalize, whose ground-truth annotations list, for each validation image, the observations
# with a positive label (the majority vote of three radiologists); an observation absent from an
# image is negative. CheXlocalize covers 10 observations, and Pneumonia is not one of them.
CHEXPLUS_IMAGES = "PNG_valid"
CHEXLOCALIZE_ANNOTATIONS = "CheXlocalize/gt_annotations_val.json"
CHEXLOCALIZE_OBSERVATIONS = {"Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
                             "Pneumothorax", "Enlarged Cardiomediastinum", "Lung Lesion", "Airspace Opacity",
                             "Support Devices"}


def find_valid_csv(root: Path) -> Path:
    for candidate in (root / CSV_NAME, *sorted(root.glob(f"*/{CSV_NAME}"))):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{CSV_NAME} not found in {root} or in its subfolders")


def resolve_image(root: Path, relative: str) -> Path:
    """``Path`` in valid.csv starts with the dataset folder (``CheXpert-v1.0-small/valid/...``);
    accept a root above or at that folder."""
    parts = Path(relative).parts
    for candidate in (root / relative, root / Path(*parts[1:])):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image {relative} not found under {root}")


def load_chexpert(csv_path: Path) -> pd.DataFrame:
    """Frontal images of valid.csv in the prediction-table format: image, patient_id, age, sex,
    view and one 0/1 column per mapped NIH class."""
    raw = pd.read_csv(csv_path)
    missing = [c for c in ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", *CLASS_MAP.values()] if c not in raw]
    if missing:
        raise ValueError(f"{csv_path}: columns missing: {missing}")
    raw = raw[raw["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    labels = raw[list(CLASS_MAP.values())].fillna(0)
    bad = sorted(set(np.unique(labels.to_numpy())) - {0, 1})
    if bad:
        raise ValueError(f"{csv_path}: labels other than 0/1 found ({bad}); use the validation set, "
                         f"whose labels have no uncertain value")
    table = pd.DataFrame({
        "image": raw["Path"],
        "patient_id": raw["Path"].map(lambda p: re.search(r"patient(\d+)", p).group(1)).astype(int),
        "age": raw["Age"],
        "sex": raw["Sex"].map({"Male": "M", "Female": "F"}).fillna("?"),
        "view": raw["AP/PA"].fillna("?"),
    })
    for nih, chexpert in CLASS_MAP.items():
        table[nih] = labels[chexpert].astype(int)
    return table


def load_chexlocalize(root: Path) -> pd.DataFrame:
    """Frontal validation images of CheXpert Plus with the CheXlocalize radiologist labels, in the
    same format as :func:`load_chexpert`; only the classes CheXlocalize annotates get a column.
    Age, sex and view are not in these files and are left unknown."""
    annotations = json.loads((root / CHEXLOCALIZE_ANNOTATIONS).read_text(encoding="utf-8"))
    images_dir = root / CHEXPLUS_IMAGES
    keys = {"_".join(p.relative_to(images_dir).with_suffix("").parts): p for p in images_dir.rglob("*.png")}
    orphans = sorted(set(annotations) - set(keys))
    if orphans:
        raise ValueError(f"{len(orphans)} annotated images missing from {images_dir}, e.g. {orphans[0]}")
    classes = [nih for nih, chexpert in CLASS_MAP.items() if chexpert in CHEXLOCALIZE_OBSERVATIONS]
    rows = []
    for key, path in sorted(keys.items()):
        if not key.endswith("_frontal"):
            continue
        positives = set(annotations.get(key, {})) - {"img_size"}
        unknown = positives - CHEXLOCALIZE_OBSERVATIONS
        if unknown:
            raise ValueError(f"{key}: unexpected observations {sorted(unknown)}")
        relative = path.relative_to(root).as_posix()
        rows.append({"image": relative, "patient_id": int(re.search(r"patient(\d+)", relative).group(1)),
                     "age": np.nan, "sex": "?", "view": "?",
                     **{nih: int(CLASS_MAP[nih] in positives) for nih in classes}})
    return pd.DataFrame(rows)


def load_table(root: Path) -> tuple[pd.DataFrame, str]:
    """The CheXpert validation set in whichever layout ``root`` has: the original download
    (``valid.csv``) or the current AIMI one (CheXpert Plus images + CheXlocalize labels)."""
    try:
        return load_chexpert(find_valid_csv(root)), "valid.csv"
    except FileNotFoundError:
        if (root / CHEXLOCALIZE_ANNOTATIONS).is_file() and (root / CHEXPLUS_IMAGES).is_dir():
            return load_chexlocalize(root), "CheXlocalize"
        raise FileNotFoundError(f"Neither {CSV_NAME} nor {CHEXPLUS_IMAGES}/ with {CHEXLOCALIZE_ANNOTATIONS} "
                                f"found under {root}") from None


def fit_square(image: Image.Image, mode: str) -> Image.Image:
    image = image.convert("L")
    if mode == "resize" or image.width == image.height:
        return image  # prepare_image resizes to a square
    if mode == "pad":
        side = max(image.size)
        return ImageOps.pad(image, (side, side), color=0)
    raise ValueError(f"Unknown fit mode {mode!r}")


@torch.no_grad()
def predict_chexpert(model: torch.nn.Module, cfg: dict, root: Path, fit: str = "resize",
                     batch_size: int = 32) -> pd.DataFrame:
    table, source = load_table(root)
    logger.info("CheXpert labels from %s", source)
    classes = cfg["data"]["classes"]
    device = next(model.parameters()).device
    logits = []
    for start in range(0, len(table), batch_size):
        batch = []
        for relative in table["image"].iloc[start:start + batch_size]:
            with Image.open(resolve_image(root, relative)) as img:
                batch.append(prepare_image(fit_square(img, fit), cfg))
        logits.append(model(torch.stack(batch).to(device)).float().cpu())
    logits = torch.cat(logits).numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    logger.info("CheXpert: %d frontal images from %d patients", len(table), table["patient_id"].nunique())
    return pd.concat([table,
                      pd.DataFrame(logits, columns=[f"logit_{c}" for c in classes]),
                      pd.DataFrame(scores, columns=[f"score_{c}" for c in classes])], axis=1)


def evaluate_external(run_dir: Path, results_dir: Path, n_boot: int, seed: int,
                      focus: list[str]) -> tuple[dict, pd.DataFrame]:
    """AUC per shared class on CheXpert next to the NIH test AUC of the same classes."""
    name = run_dir.name
    columns = pd.read_csv(run_dir / PREDS_NAME, nrows=0).columns
    classes = [c for c in CLASS_MAP if c in columns]  # without valid.csv, no Pneumonia labels
    preds = Predictions(run_dir / PREDS_NAME, classes)
    external = bootstrap_auc_ap(preds, n_boot, seed)
    nih = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))["classes"]
    nih_mean = float(np.mean([nih[c]["auc"] for c in classes]))

    rows, display = [], []
    for c in classes:
        e = external["classes"][c]
        few = e["n_positive"] < MIN_POSITIVES_FOR_CONCLUSIONS
        rows.append({"classe": c, "classe_chexpert": CLASS_MAP[c], "n_positivos_chexpert": e["n_positive"],
                     "auc_chexpert": e["auc"], "ic_inf": e["auc_ci"][0], "ic_sup": e["auc_ci"][1],
                     "auc_nih_teste": nih[c]["auc"], "diferenca": e["auc"] - nih[c]["auc"], "poucos_casos": few})
        display.append({"Doença": pt(c) + (" *" if c in focus else ""), "Casos (CheXpert)": e["n_positive"],
                        "AUC CheXpert (IC95%)": fmt_ci(e["auc"], e["auc_ci"]) + (" †" if few else ""),
                        "AUC NIH teste": fmt(nih[c]["auc"]),
                        "Diferença": fmt(e["auc"] - nih[c]["auc"])})
    mean = external["mean_auc"]
    rows.append({"classe": f"Média ({len(classes)} classes)", "auc_chexpert": mean["value"], "ic_inf": mean["ci"][0],
                 "ic_sup": mean["ci"][1], "auc_nih_teste": nih_mean, "diferenca": mean["value"] - nih_mean})
    display.append({"Doença": f"Média ({len(classes)} classes)", "Casos (CheXpert)": "",
                    "AUC CheXpert (IC95%)": fmt_ci(mean["value"], mean["ci"]), "AUC NIH teste": fmt(nih_mean),
                    "Diferença": fmt(mean["value"] - nih_mean)})
    numeric, shown = pd.DataFrame(rows), pd.DataFrame(display)
    save_table(numeric, shown, results_dir / "tables" / name / "validacao_externa_chexpert")

    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for color, c in zip(FOCUS_COLORS, focus):
        if c not in classes:
            continue
        k = classes.index(c)
        fpr, tpr, _ = roc_curve(preds.y[:, k], preds.scores[:, k])
        e = external["classes"][c]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{pt(c)} (n={br_number(e['n_positive'])}): AUC {fmt_ci(e['auc'], e['auc_ci'])}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("1 − especificidade (taxa de falsos positivos)")
    ax.set_ylabel("Sensibilidade")
    ax.set_title("Validação externa: curvas ROC no CheXpert (validação)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    use_decimal_comma(ax.xaxis, ax.yaxis)
    save_figure(fig, results_dir / "figures" / name / "roc_foco_chexpert")
    plt.close(fig)

    metrics = {"experiment": name, "dataset": "CheXpert (validação)", "n_images": len(preds.y),
               "n_patients": int(len(np.unique(preds.patients))),
               "bootstrap": {"samples": n_boot, "seed": seed, "unit": "patient", "level": 0.95},
               "class_map": {c: CLASS_MAP[c] for c in classes}, **external}
    (run_dir / "metrics_chexpert.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, shown


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment config used for training")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    parser.add_argument("--chexpert-root", type=Path, help="folder with CheXpert's valid.csv (or its parent)")
    parser.add_argument("--fit", choices=["resize", "pad"], default="resize")
    parser.add_argument("--evaluate-only", action="store_true", help="reuse preds_chexpert.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, help="bootstrap samples (default: eval.bootstrap_samples)")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = load_config(args.config, args.paths)
    run_dir = Path(cfg["paths"]["runs_dir"]) / cfg["experiment"]
    if not args.evaluate_only:
        if args.chexpert_root is None:
            parser.error("--chexpert-root is required unless --evaluate-only")
        model, _ = load_trained_model(Path(cfg["paths"]["checkpoint_dir"]) / cfg["experiment"] / "best.pt",
                                      get_device(args.device))
        preds = predict_chexpert(model, cfg, args.chexpert_root, args.fit)
        preds.to_csv(run_dir / PREDS_NAME, index=False, float_format=FLOAT_FORMAT)
    _, table = evaluate_external(run_dir, Path(cfg["paths"]["results_dir"]),
                                 args.bootstrap or cfg["eval"]["bootstrap_samples"], cfg["eval"]["bootstrap_seed"],
                                 cfg["data"]["focus_classes"])
    logger.info("External validation:\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
