import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from chestxray import demo as dm
from chestxray import inference as inf
from chestxray.config import load_config
from chestxray.metrics import apply_platt
from chestxray.models.densenet import build_model

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
CLASSES = BASE["data"]["classes"]
FOCUS = BASE["data"]["focus_classes"]


@pytest.fixture
def trained(tmp_path):
    """A tiny untrained model with everything the export needs: best.pt, preds_test.csv,
    calibration.json and the images."""
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    rows = []
    for i in range(10):
        name = f"{i:08d}_000.png"
        Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8), "L").save(data / "images" / name)
        rows.append({"image": name, "patient_id": i, "age": 40, "sex": "F", "view": "AP",
                     **{c: int(i % 3 == 0) for c in CLASSES}})
    pd.DataFrame(rows).to_csv(data / "test.csv", index=False)
    cfg = copy.deepcopy(BASE)
    cfg["experiment"] = "tiny"
    cfg["paths"] = {"data_dir": str(data), "splits_dir": str(data), "runs_dir": str(tmp_path / "runs"),
                    "checkpoint_dir": str(tmp_path / "ckpt"), "results_dir": str(tmp_path / "results")}
    cfg["data"]["image_size"] = 64
    cfg["model"]["pretrained"] = False
    torch.manual_seed(0)
    model = build_model(cfg)
    (tmp_path / "ckpt" / "tiny").mkdir(parents=True)
    torch.save({"config": cfg, "model": model.state_dict(), "epoch": 4, "optimizer": {}},
               tmp_path / "ckpt" / "tiny" / "best.pt")
    run_dir = tmp_path / "runs" / "tiny"
    run_dir.mkdir(parents=True)
    preds = inf.predict_split(model.eval(), cfg, "test")
    preds.to_csv(run_dir / "preds_test.csv", index=False)
    calibration = {"classes": {c: {"threshold": float(np.median(preds[f"score_{c}"])), "platt_a": 0.8,
                                   "platt_b": -0.3} for c in CLASSES}}
    for c, t in calibration["classes"].items():
        t["threshold_calibrated"] = float(apply_platt(dm.to_logit(t["threshold"]), 0.8, -0.3))
    (run_dir / "calibration.json").write_text(json.dumps(calibration))
    return cfg, preds


def test_summary_sentence_never_says_normal():
    assert dm.summary_sentence([]) == "Nenhum achado acima do limiar entre as 14 doenças avaliadas."
    assert dm.summary_sentence(["Effusion", "Atelectasis"]) == "Achados acima do limiar: Efusão pleural, Atelectasia."
    for text in (dm.summary_sentence([]), dm.summary_sentence(["Hernia"])):
        assert "normal" not in text.lower() and "saudável" not in text.lower()


def test_build_result_orders_focus_first_and_flags_thresholds():
    values = np.linspace(0.05, 0.7, len(CLASSES))
    thresholds = {c: 0.5 for c in CLASSES}
    result = dm.build_result(CLASSES, FOCUS, values, thresholds, "Escore do modelo")
    assert list(result.table["Grupo"][:3]) == [dm.FOCUS_GROUP] * 3
    focus_values = [result.values[c] for c in FOCUS]
    shown = [dm.pt(c) for c in sorted(FOCUS, key=lambda c: -result.values[c])]
    assert list(result.table["Doença"][:3]) == shown and len(set(focus_values)) == 3
    above = [c for k, c in enumerate(CLASSES) if values[k] >= 0.5]
    assert result.summary.startswith("Achados acima do limiar:")
    assert all(dm.pt(c) in result.summary for c in above)
    assert (result.table["Situação"] == dm.ABOVE).sum() == len(above)
    assert result.table["Escore do modelo"].iloc[0].endswith("%") and "," in result.table["Limiar"].iloc[0]


def test_export_reproduces_the_evaluated_scores(trained, tmp_path):
    cfg, preds = trained
    bundle = dm.export(cfg, tmp_path / "bundle")
    for name in ("model.pt", "calibration.json", "app.yaml"):
        assert (bundle / name).exists()
    state = torch.load(bundle / "model.pt", map_location="cpu", weights_only=False)
    assert "optimizer" not in state  # only what the app needs
    settings = (bundle / "app.yaml").read_text(encoding="utf-8")
    assert "label: score" in settings
    examples = sorted(p.name for p in (bundle / "examples").glob("*.png"))
    assert len(examples) >= 2
    assert dm.check_bundle(bundle, cfg, n=3) < 1e-4

    model = dm.DemoModel(bundle)
    assert model.label == "Escore do modelo"
    row = preds.iloc[0]
    with Image.open(Path(cfg["paths"]["data_dir"]) / "images" / row["image"]) as img:
        result = model.analyse(img.convert("L"))
    for c in CLASSES:  # displayed value = Platt applied to the evaluated logit
        assert result.values[c] == pytest.approx(apply_platt(row[f"logit_{c}"], 0.8, -0.3), abs=1e-4)


def test_pick_examples_covers_focus_classes_and_a_quiet_image():
    preds = pd.DataFrame({"image": list("abcd"), **{c: [1, 0, 0, 0] for c in CLASSES},
                          **{f"score_{c}": [0.9, 0.5, 0.4, 0.01] for c in CLASSES}})
    assert dm.pick_examples(preds, FOCUS, CLASSES) == ["a", "d"]


def test_app_answers_through_the_gradio_api(trained, tmp_path):
    cfg, preds = trained
    bundle = dm.export(cfg, tmp_path / "bundle", label="probability")
    spec = importlib.util.spec_from_file_location("demo_app", ROOT / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    blocks = app.build_demo(dm.DemoModel(bundle))
    blocks.launch(prevent_thread_lock=True, server_port=None, quiet=True)
    try:
        from gradio_client import Client, handle_file
        client = Client(blocks.local_url, verbose=False)
        image = Path(cfg["paths"]["data_dir"]) / "images" / preds["image"][0]
        summary, table, heatmap = client.predict(handle_file(str(image)), "Effusion", api_name="/analisar")
        assert "limiar" in summary and "Efusão pleural" in summary
        assert len(table["data"]) == len(CLASSES) and "Probabilidade estimada" in table["headers"]
        assert Path(heatmap).exists()
    finally:
        blocks.close()
