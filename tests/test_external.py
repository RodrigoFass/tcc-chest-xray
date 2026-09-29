import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from chestxray import external as ex
from chestxray.config import load_config
from chestxray.models.densenet import build_model

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
CLASSES = BASE["data"]["classes"]
CHEXPERT_COLUMNS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                    "Pleural Other", "Fracture", "Support Devices"]


def make_chexpert(root: Path, n_patients: int = 12, seed: int = 0) -> None:
    """valid.csv and JPGs laid out like CheXpert-v1.0-small, with one lateral view per patient."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, n_patients + 1):
        for view, fl in (("view1_frontal", "Frontal"), ("view2_lateral", "Lateral")):
            rel = f"CheXpert-v1.0-small/valid/patient{64540 + p:05d}/study1/{view}.jpg"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (120, 100), dtype=np.uint8), "L").save(root / rel)
            labels = {c: float(rng.random() < 0.4) for c in CHEXPERT_COLUMNS}
            labels.update({"Pneumonia": float(p <= 3), "Pleural Effusion": float(p % 2)})
            rows.append({"Path": rel, "Sex": "Male" if p % 2 else "Female", "Age": 40 + p,
                         "Frontal/Lateral": fl, "AP/PA": "AP" if fl == "Frontal" else np.nan, **labels})
    (root / "CheXpert-v1.0-small").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "CheXpert-v1.0-small" / "valid.csv", index=False)


def test_load_chexpert_keeps_frontal_and_maps_classes(tmp_path):
    make_chexpert(tmp_path)
    table = ex.load_chexpert(ex.find_valid_csv(tmp_path))
    assert len(table) == 12 and table["patient_id"].nunique() == 12
    assert list(table.columns[5:]) == list(ex.CLASS_MAP)
    assert table["Pneumonia"].sum() == 3 and set(table["sex"]) == {"M", "F"}
    assert ex.resolve_image(tmp_path, table["image"][0]).exists()
    assert ex.resolve_image(tmp_path / "CheXpert-v1.0-small", table["image"][0]).exists()


def test_uncertain_labels_are_refused(tmp_path):
    make_chexpert(tmp_path)
    csv = tmp_path / "CheXpert-v1.0-small" / "valid.csv"
    raw = pd.read_csv(csv)
    raw.loc[0, "Edema"] = -1.0
    raw.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="uncertain"):
        ex.load_chexpert(csv)


def test_fit_square_modes():
    image = Image.new("L", (100, 120), 255)
    assert ex.fit_square(image, "resize").size == (100, 120)
    padded = ex.fit_square(image, "pad")
    assert padded.size == (120, 120) and np.asarray(padded)[:, 0].max() == 0
    with pytest.raises(ValueError):
        ex.fit_square(image, "crop")


def test_predict_and_evaluate_external(tmp_path):
    make_chexpert(tmp_path / "chexpert")
    cfg = copy.deepcopy(BASE)
    cfg["data"]["image_size"] = 64
    cfg["model"]["pretrained"] = False
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    preds = ex.predict_chexpert(model, cfg, tmp_path / "chexpert", "pad", batch_size=5)
    assert len(preds) == 12 and {f"score_{c}" for c in CLASSES} <= set(preds.columns)

    run_dir = tmp_path / "results" / "runs" / "tiny"
    run_dir.mkdir(parents=True)
    preds.to_csv(run_dir / ex.PREDS_NAME, index=False)
    nih = {"classes": {c: {"auc": 0.8} for c in CLASSES}}
    (run_dir / "metrics_test.json").write_text(json.dumps(nih))
    metrics, table = ex.evaluate_external(run_dir, tmp_path / "results", n_boot=30, seed=0,
                                          focus=BASE["data"]["focus_classes"])
    assert metrics["n_images"] == 12 and set(metrics["classes"]) == set(ex.CLASS_MAP)
    assert "†" in table.loc[table["Doença"].str.startswith("Pneumonia"), "AUC CheXpert (IC95%)"].iloc[0]
    assert (tmp_path / "results" / "tables" / "tiny" / "validacao_externa_chexpert.tex").exists()
    assert (tmp_path / "results" / "figures" / "tiny" / "roc_foco_chexpert.pdf").exists()
    assert (run_dir / "metrics_chexpert.json").exists()


def make_chexlocalize(root: Path, n_patients: int = 12, seed: int = 0) -> dict:
    """CheXpert Plus PNGs (PNG_valid/) and CheXlocalize ground-truth annotations, as Stanford AIMI
    distributes the validation set today. Returns the expected Effusion label per patient."""
    rng = np.random.default_rng(seed)
    annotations, effusion = {}, {}
    for p in range(1, n_patients + 1):
        for view in ("view1_frontal", "view2_lateral"):
            rel = Path("PNG_valid") / f"patient{64540 + p:05d}" / "study1" / f"{view}.png"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (120, 100), dtype=np.uint8), "L").save(root / rel)
        positives = {"Pleural Effusion": [[1, 2]]} if p % 2 else {}
        if p <= 4:
            positives["Atelectasis"] = [[3, 4]]
        effusion[64540 + p] = int(p % 2)
        if positives:  # like CheXlocalize: images without positive labels are not listed
            annotations[f"patient{64540 + p:05d}_study1_view1_frontal"] = {"img_size": [120, 100], **positives}
    (root / "CheXlocalize").mkdir(parents=True)
    (root / "CheXlocalize" / "gt_annotations_val.json").write_text(json.dumps(annotations))
    return effusion


def test_load_chexlocalize_rebuilds_the_radiologist_labels(tmp_path):
    effusion = make_chexlocalize(tmp_path)
    table, source = ex.load_table(tmp_path)
    assert source == "CheXlocalize" and len(table) == 12  # frontal images only
    assert "Pneumonia" not in table.columns  # CheXlocalize does not annotate it
    assert dict(zip(table["patient_id"], table["Effusion"])) == effusion
    assert table["Atelectasis"].sum() == 4 and table[["Cardiomegaly", "Edema"]].to_numpy().sum() == 0


def test_chexlocalize_annotation_without_image_is_refused(tmp_path):
    make_chexlocalize(tmp_path)
    path = tmp_path / "CheXlocalize" / "gt_annotations_val.json"
    annotations = json.loads(path.read_text())
    annotations["patient99999_study1_view1_frontal"] = {"Edema": []}
    path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError, match="missing"):
        ex.load_table(tmp_path)


def test_predict_and_evaluate_on_chexlocalize_layout(tmp_path):
    make_chexlocalize(tmp_path / "chexpert")
    cfg = copy.deepcopy(BASE)
    cfg["data"]["image_size"] = 64
    cfg["model"]["pretrained"] = False
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    preds = ex.predict_chexpert(model, cfg, tmp_path / "chexpert", "resize", batch_size=5)
    run_dir = tmp_path / "results" / "runs" / "tiny"
    run_dir.mkdir(parents=True)
    preds.to_csv(run_dir / ex.PREDS_NAME, index=False)
    (run_dir / "metrics_test.json").write_text(json.dumps({"classes": {c: {"auc": 0.8} for c in CLASSES}}))
    metrics, table = ex.evaluate_external(run_dir, tmp_path / "results", n_boot=30, seed=0,
                                          focus=BASE["data"]["focus_classes"])
    assert set(metrics["classes"]) == set(ex.CLASS_MAP) - {"Pneumonia"}
    assert not table["Doença"].str.startswith("Pneumonia").any()
