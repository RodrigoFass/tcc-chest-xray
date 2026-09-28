import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import scale_cam_image

from chestxray import gradcam as gc
from chestxray import inference as inf
from chestxray.config import load_config
from chestxray.models.densenet import build_model

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
CLASSES = BASE["data"]["classes"]


def make_data(folder: Path, n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    (folder / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        name = f"{i:08d}_000.png"
        Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8), "L").save(folder / "images" / name)
        rows.append({"image": name, "patient_id": i // 2 + 1, "age": 50, "sex": "M", "view": "PA",
                     **{c: int(rng.random() < 0.4) for c in CLASSES}})
    table = pd.DataFrame(rows)
    table.to_csv(folder / "test.csv", index=False)
    return table


@pytest.fixture
def setup(tmp_path):
    table = make_data(tmp_path / "data")
    cfg = copy.deepcopy(BASE)
    cfg["experiment"] = "tiny"
    cfg["paths"] = {"data_dir": str(tmp_path / "data"), "splits_dir": str(tmp_path / "data"),
                    "runs_dir": str(tmp_path / "runs"), "checkpoint_dir": str(tmp_path / "ckpt"),
                    "results_dir": str(tmp_path / "results")}
    cfg["data"]["image_size"] = 64
    cfg["model"]["pretrained"] = False
    torch.manual_seed(0)
    model = build_model(cfg)
    ckpt = tmp_path / "ckpt" / "tiny" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"config": cfg, "model": model.state_dict()}, ckpt)
    return cfg, table, ckpt


def test_explanation_matches_inference_scores(setup):
    cfg, table, ckpt = setup
    explainer = gc.Explainer.from_checkpoint(ckpt, "cpu", ["relu", "denseblock4"])
    image = Path(cfg["paths"]["data_dir"]) / "images" / table["image"][0]
    model, _ = inf.load_trained_model(ckpt, torch.device("cpu"))
    with Image.open(image) as img:
        expected = inf.predict_image(model, cfg, img)
    for layer in ("relu", "denseblock4"):
        e = explainer.explain(image, "Effusion", layer)
        assert e.score == pytest.approx(expected[CLASSES.index("Effusion")], abs=1e-6)
        assert e.cam.shape == e.base.shape == (64, 64)
        assert 0 <= e.cam.min() and e.cam.max() <= 1
        assert e.overlay.shape == (64, 64, 3) and e.overlay.dtype == np.uint8
        x, y = e.peak
        assert 0 < x < 64 and 0 < y < 64
    explainer.close()


def test_gradcam_on_last_relu_equals_cam(setup):
    """Plan 3.11: with global pooling and one linear layer, Grad-CAM on the last feature map is
    CAM (Zhou et al., 2016) up to a positive factor, so both give the same normalized heatmap."""
    cfg, table, ckpt = setup
    explainer = gc.Explainer.from_checkpoint(ckpt, "cpu", ["relu"])
    with Image.open(Path(cfg["paths"]["data_dir"]) / "images" / table["image"][3]) as img:
        image = img.convert("L")
    k = CLASSES.index("Atelectasis")
    with torch.no_grad():
        features = explainer.model.relu(explainer.model.features(inf.prepare_image(image, cfg).unsqueeze(0)))[0]
        cam = torch.einsum("c,chw->hw", explainer.model.classifier.weight[k], features).clamp(min=0).numpy()
    expected = scale_cam_image(cam[None], (64, 64))[0]
    np.testing.assert_allclose(explainer.heatmap(image, "Atelectasis"), expected, atol=1e-4)
    explainer.close()


def test_select_gallery_follows_the_fixed_rule():
    preds = pd.DataFrame({
        "image": [f"img{i}" for i in range(8)],
        "patient_id": [1, 1, 2, 3, 4, 5, 6, 7],
        "Pneumonia": [1, 1, 1, 0, 0, 1, 1, 0],
        "score_Pneumonia": [0.9, 0.95, 0.6, 0.8, 0.7, 0.1, 0.3, 0.2],
    })
    sel = gc.select_gallery(preds, "Pneumonia", threshold=0.5, n=2)
    by_group = {g: list(s["image"]) for g, s in sel.groupby("group")}
    assert by_group["tp"] == ["img1", "img2"]  # img0 is the same patient as img1 (higher score kept)
    assert by_group["fp"] == ["img3", "img4"]
    assert by_group["fn"] == ["img5", "img6"]  # lowest scores first
    assert list(sel.loc[sel["group"] == "tp", "rank"]) == [1, 2]


def test_load_boxes_fixes_header_labels_and_scale(tmp_path):
    (tmp_path / gc.BBOX_FILE).write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        "a.png,Infiltrate,512,256,128,64,,,\n"
        "b.png,Effusion,0,0,1024,1024,,,\n", encoding="utf-8")
    boxes = gc.load_boxes(tmp_path, 224, CLASSES)
    assert list(boxes["class"]) == ["Infiltration", "Effusion"]
    assert boxes.iloc[0][["x", "y", "w", "h"]].tolist() == pytest.approx([112, 56, 28, 14])
    (tmp_path / gc.BBOX_FILE).write_text("Image Index,Finding Label,x,y,w,h\na.png,Tumor,1,1,1,1\n")
    with pytest.raises(ValueError, match="Tumor"):
        gc.load_boxes(tmp_path, 224, CLASSES)


def test_wilson_interval_known_value():
    lo, hi = gc.wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert all(np.isnan(gc.wilson_interval(0, 0)))


def test_pointing_game_counts_each_image_once():
    results = pd.DataFrame([
        # two boxes of the same class on one image: one hit is enough
        {"image": "a", "class": "Effusion", "layer": "relu", "x": 0, "y": 0, "w": 10, "h": 10, "hit": False},
        {"image": "a", "class": "Effusion", "layer": "relu", "x": 50, "y": 50, "w": 10, "h": 10, "hit": True},
        {"image": "b", "class": "Effusion", "layer": "relu", "x": 100, "y": 100, "w": 30, "h": 30, "hit": False},
    ])
    numeric, display = gc.pointing_game(results, image_size=224)
    row = numeric.iloc[0]
    assert (row["n_imagens"], row["acertos"], row["taxa"]) == (2, 1, 0.5)
    assert row["taxa_centro"] == 0.5  # the centre (112, 112) is inside image b's box only
    assert "50,0%" in display.iloc[0]["Taxa (IC95%)"]


def test_run_writes_gallery_boxes_and_pointing_game(setup):
    cfg, table, ckpt = setup
    run_dir = Path(cfg["paths"]["runs_dir"]) / "tiny"
    run_dir.mkdir(parents=True)
    model, _ = inf.load_trained_model(ckpt, torch.device("cpu"))
    preds = inf.predict_split(model, cfg, "test")
    preds.to_csv(run_dir / "preds_test.csv", index=False)
    median = {c: float(np.median(preds[f"score_{c}"])) for c in CLASSES}
    (run_dir / "calibration.json").write_text(json.dumps({"classes": {c: {"threshold": t} for c, t in median.items()}}))
    with_box = table.loc[table["Effusion"] == 1, "image"].iloc[0]
    (Path(cfg["paths"]["data_dir"]) / gc.BBOX_FILE).write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        f"{with_box},Effusion,100,100,600,600,,,\n"
        "not_in_test.png,Effusion,0,0,10,10,,,\n", encoding="utf-8")

    summary = gc.run(cfg, ["relu", "denseblock4"], torch.device("cpu"), n_per_group=2)

    figures = Path(cfg["paths"]["results_dir"]) / "figures" / "tiny" / "gradcam"
    for stem in ("galeria_pneumonia", "galeria_atelectasis", "galeria_effusion", "camadas_foco", "caixas_effusion"):
        assert (figures / f"{stem}.png").exists() and (figures / f"{stem}.pdf").exists(), stem
    tables = Path(cfg["paths"]["results_dir"]) / "tables" / "tiny"
    selection = pd.read_csv(tables / "gradcam_selecao.csv")
    assert set(selection["classe"]) == {"Pneumonia", "Atelectasis", "Effusion"}
    game = pd.read_csv(tables / "pointing_game.csv")
    assert set(game["camada"]) == {"relu", "denseblock4"} and set(game["n_imagens"]) == {1}
    assert summary["pointing_game"] is not None
