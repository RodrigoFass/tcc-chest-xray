import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from chestxray import evaluate as ev
from chestxray import inference as inf
from chestxray import train as tr
from chestxray.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
CLASSES = BASE["data"]["classes"]


def fake_preds(n_patients: int, seed: int, strength: float) -> pd.DataFrame:
    """Predictions with signal ``strength``: higher means better separated scores."""
    rng = np.random.default_rng(seed)
    patients = np.repeat(np.arange(n_patients), rng.integers(1, 4, n_patients))
    n = len(patients)
    y = (rng.random((n, len(CLASSES))) < 0.15).astype(int)
    y[:5] = 1  # every class has positives
    logits = strength * (2 * y - 1) + rng.normal(0, 1.5, y.shape) - 1.5
    table = pd.DataFrame({"image": [f"{i:08d}_000.png" for i in range(n)], "patient_id": patients,
                          "age": rng.integers(18, 90, n), "sex": rng.choice(["M", "F"], n),
                          "view": rng.choice(["PA", "AP"], n)})
    table[CLASSES] = y
    table[[f"logit_{c}" for c in CLASSES]] = logits
    table[[f"score_{c}" for c in CLASSES]] = 1 / (1 + np.exp(-logits))
    return table


def make_run(runs_dir: Path, name: str, strength: float) -> Path:
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    cfg = copy.deepcopy(BASE)
    cfg["experiment"] = name
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"best_epoch": 3, "epochs": 8, "train_time_min": 50.0}))
    fake_preds(150, seed=1, strength=strength).to_csv(run_dir / "preds_val.csv", index=False)
    fake_preds(150, seed=2, strength=strength).to_csv(run_dir / "preds_test.csv", index=False)
    return run_dir


def test_fmt_ci_keeps_the_sign_of_limits_near_zero():
    assert ev.fmt_ci(0.8204, [0.8101, 0.8299]) == "0,820 (0,810–0,830)"
    assert ev.fmt_ci(-0.003953, [-0.007608, -0.000348]) == "-0,004 (-0,008–-0,0003)"
    assert ev.fmt_ci(-0.005704, [-0.010416, 0.000157]) == "-0,006 (-0,010–0,0002)"


def test_evaluate_run_writes_metrics_tables_and_figures(tmp_path):
    run_dir = make_run(tmp_path / "results" / "runs", "exp_a", strength=1.5)
    results = tmp_path / "results"
    val = ev.evaluate_run(run_dir, "val", results, n_boot=50)
    test = ev.evaluate_run(run_dir, "test", results, n_boot=50)

    for m in (val, test):
        lo, hi = m["mean_auc"]["ci"]
        assert lo <= m["mean_auc"]["value"] <= hi
        effusion = m["classes"]["Effusion"]
        assert effusion["auc_ci"][0] <= effusion["auc"] <= effusion["auc_ci"][1]
    assert "thresholds_from_val" in test and "thresholds_from_val" not in val
    assert len(test["subgroups"]) == 3 * 7  # 3 focus classes x 7 subgroups

    calibration = json.loads((run_dir / "calibration.json").read_text())
    assert calibration["fitted_on"] == "val" and set(calibration["classes"]) == set(CLASSES)
    assert (run_dir / "metrics_test.json").exists()
    for table in ("metricas_val", "metricas_test", "limiares_teste", "calibracao_teste",
                  "subgrupos_teste", "comparacao_literatura"):
        for ext in ("csv", "md", "tex"):
            assert (results / "tables" / "exp_a" / f"{table}.{ext}").exists(), (table, ext)
    md = (results / "tables" / "exp_a" / "metricas_test.md").read_text(encoding="utf-8")
    assert "Efusão pleural *" in md and "," in md  # Portuguese names, decimal comma
    for fig in ("roc_foco_test", "roc_14_test", "pr_foco_test", "matriz_confusao_foco_test",
                "calibracao_foco_test", "subgrupos_foco_test"):
        assert (results / "figures" / "exp_a" / f"{fig}.png").exists(), fig


def test_compare_runs_and_paired_differences(tmp_path):
    results = tmp_path / "results"
    strong = make_run(results / "runs", "strong", strength=2.0)
    weak = make_run(results / "runs", "weak", strength=0.3)
    for run in (strong, weak):
        ev.evaluate_run(run, "test", results, n_boot=30)
    table, diffs = ev.compare_runs([strong, weak], [("strong", "weak")], results, n_boot=30)
    assert list(table["Experimento"]) == ["strong", "weak"]
    numeric = pd.read_csv(results / "tables" / "diferencas_pareadas_test.csv")
    mean_row = numeric[numeric["classe"] == "media"].iloc[0]
    assert mean_row["diferenca_auc"] > 0 and mean_row["significativa"]
    assert set(numeric["classe"]) == {"media", *BASE["data"]["focus_classes"]}


def test_compare_refuses_runs_on_different_images(tmp_path):
    results = tmp_path / "results"
    a = make_run(results / "runs", "a", strength=1.0)
    b = make_run(results / "runs", "b", strength=1.0)
    ev.evaluate_run(a, "test", results, n_boot=10)
    ev.evaluate_run(b, "test", results, n_boot=10)
    fake_preds(120, seed=3, strength=1.0).to_csv(b / "preds_test.csv", index=False)
    with pytest.raises(SystemExit, match="different images"):
        ev.compare_runs([a, b], [], results, n_boot=10)


def test_inference_matches_between_split_and_single_image(tmp_path):
    """The Phase 5 criterion in miniature: scoring one image gives the same scores as the
    batch predictions written to preds_<split>.csv."""
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    rows = []
    for i in range(12):
        name = f"{i:08d}_000.png"
        Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8), "L").save(data / "images" / name)
        rows.append({"image": name, "patient_id": i, "age": 50, "sex": "M", "view": "PA",
                     **{c: int(rng.random() < 0.3) for c in CLASSES}})
    for split in ("train", "val", "test"):
        pd.DataFrame(rows).to_csv(data / f"{split}.csv", index=False)
    cfg = copy.deepcopy(BASE)
    cfg["experiment"] = "tiny"
    cfg["paths"] = {"data_dir": str(data), "splits_dir": str(data), "runs_dir": str(tmp_path / "runs"),
                    "checkpoint_dir": str(tmp_path / "ckpt"), "results_dir": str(tmp_path / "results")}
    cfg["data"].update(image_size=64, stored_size=64)
    cfg["model"]["pretrained"] = False
    cfg["train"].update(device="cpu", amp=False, batch_size=4, num_workers=0, max_epochs=1)
    tr.train(cfg)

    model, trained_cfg = inf.load_trained_model(tmp_path / "ckpt" / "tiny" / "best.pt", torch.device("cpu"))
    preds = inf.predict_split(model, cfg, "test")
    assert list(preds.columns[:5]) == ["image", "patient_id", "age", "sex", "view"]
    assert len(preds) == 12 and preds[CLASSES].to_numpy().sum() == pd.DataFrame(rows)[CLASSES].to_numpy().sum()
    for i in (0, 7):
        image = Image.open(data / "images" / preds.loc[i, "image"])
        single = inf.predict_image(model, trained_cfg, image)
        assert np.allclose(single, preds.loc[i, [f"score_{c}" for c in CLASSES]].to_numpy(float), atol=1e-5)


def test_seeds_summary_reports_mean_and_sd(tmp_path):
    runs = tmp_path / "results" / "runs"
    dirs = []
    for seed, strength in ((42, 1.5), (43, 1.4), (44, 1.6)):
        run_dir = make_run(runs, f"exp_seed{seed}", strength=strength)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        cfg["seed"] = seed
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        ev.evaluate_run(run_dir, "test", tmp_path / "results", n_boot=20)
        dirs.append(run_dir)
    display = ev.seeds_summary(dirs, tmp_path / "results")
    numeric = pd.read_csv(tmp_path / "results" / "tables" / "seeds_test.csv")
    per_run = numeric["auc_media"][:3]
    assert numeric["auc_media"][3] == pytest.approx(per_run.mean())
    assert numeric["auc_media_dp"][3] == pytest.approx(per_run.std(ddof=1))
    assert "±" in display.iloc[-1]["AUC média"] and "Pneumonia: AUPRC" in display.columns
    with pytest.raises(SystemExit):
        ev.seeds_summary([dirs[0], dirs[0]], tmp_path / "results")


def test_training_curves_figure(tmp_path):
    run_dir = tmp_path / "results" / "runs" / "exp"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"epoch": [1, 2, 3], "lr": [1e-4, 1e-4, 1e-5], "train_loss": [0.2, 0.15, 0.1],
                  "val_loss": [0.18, 0.16, 0.17], "val_auc_mean": [0.7, 0.8, 0.78]}).to_csv(run_dir / "log.csv", index=False)
    ev.main(["--curves", str(run_dir)])
    assert (tmp_path / "results" / "figures" / "curvas_treino_exp.pdf").exists()
