import copy
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from chestxray import train as tr
from chestxray.config import load_config
from chestxray.data.dataset import build_dataloader

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
CLASSES = BASE["data"]["classes"]


def make_dataset(folder: Path, sizes: dict[str, int], seed: int = 0) -> None:
    """Small grayscale images with random multi-label targets, in the split-CSV format."""
    rng = np.random.default_rng(seed)
    (folder / "images").mkdir(parents=True)
    index = 0
    for split, n in sizes.items():
        rows = []
        for _ in range(n):
            name = f"{index:08d}_000.png"
            Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8), "L").save(folder / "images" / name)
            rows.append({"image": name, "patient_id": index, **{c: int(rng.random() < 0.3) for c in CLASSES}})
            index += 1
        pd.DataFrame(rows).to_csv(folder / f"{split}.csv", index=False)


@pytest.fixture
def cfg(tmp_path):
    make_dataset(tmp_path / "data", {"train": 24, "val": 16, "test": 8})
    cfg = copy.deepcopy(BASE)
    cfg["experiment"] = "tiny"
    cfg["paths"] = {
        "data_dir": str(tmp_path / "data"), "splits_dir": str(tmp_path / "data"),
        "runs_dir": str(tmp_path / "runs"), "checkpoint_dir": str(tmp_path / "ckpt"),
        "results_dir": str(tmp_path / "results"),
    }
    cfg["data"]["image_size"] = 64
    cfg["model"]["pretrained"] = False  # no download in tests
    cfg["train"].update(device="cpu", amp=False, batch_size=8, num_workers=0, max_epochs=2)
    return cfg


def read_log(cfg) -> list[dict]:
    with (Path(cfg["paths"]["runs_dir"]) / cfg["experiment"] / "log.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_training_run_writes_everything(cfg):
    summary = tr.train(cfg)
    run_dir = Path(cfg["paths"]["runs_dir"]) / "tiny"
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "tiny"
    for name in ("config.yaml", "environment.json", "log.csv", "train.log", "summary.json"):
        assert (run_dir / name).exists(), name
    assert (ckpt_dir / "last.pt").exists() and (ckpt_dir / "best.pt").exists()
    assert [row["epoch"] for row in read_log(cfg)] == ["1", "2"]
    assert summary["epochs"] == 2 and summary["best_epoch"] in (1, 2)
    best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == summary["best_epoch"] and set(best["val_auc"]) == set(CLASSES)
    assert json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))[0]["start_epoch"] == 1


def test_interrupted_run_resumes_to_the_same_result(cfg, tmp_path, monkeypatch):
    straight = copy.deepcopy(cfg)
    tr.train(straight)

    resumed = copy.deepcopy(cfg)
    resumed["paths"]["runs_dir"] = str(tmp_path / "runs_resumed")
    resumed["paths"]["checkpoint_dir"] = str(tmp_path / "ckpt_resumed")
    real_epoch, calls = tr.train_one_epoch, []

    def crash_in_second_epoch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt  # e.g. the PC or the Colab session goes down
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(tr, "train_one_epoch", crash_in_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        tr.train(resumed)
    assert [row["epoch"] for row in read_log(resumed)] == ["1"]
    monkeypatch.setattr(tr, "train_one_epoch", real_epoch)
    tr.train(resumed)

    log_straight, log_resumed = read_log(straight), read_log(resumed)
    assert [r["epoch"] for r in log_resumed] == ["1", "2"]  # no duplicated or missing row
    for a, b in zip(log_straight, log_resumed):
        for key in ("train_loss", "val_loss", "val_auc_mean"):
            assert float(a[key]) == pytest.approx(float(b[key]), rel=1e-4), key
    sessions = json.loads((Path(resumed["paths"]["runs_dir"]) / "tiny" / "environment.json").read_text(encoding="utf-8"))
    assert [s["start_epoch"] for s in sessions] == [1, 2]


def test_finished_run_is_not_run_again(cfg, monkeypatch):
    first = tr.train(cfg)
    monkeypatch.setattr(tr, "train_one_epoch", lambda *a, **k: pytest.fail("should not train again"))
    again = tr.train(cfg)
    assert (again["epochs"], again["best_epoch"]) == (first["epochs"], first["best_epoch"])


def test_existing_run_with_another_config_is_never_overwritten(cfg):
    tr.train(cfg)
    changed = copy.deepcopy(cfg)
    changed["train"]["lr"] = 1e-3
    with pytest.raises(SystemExit, match="different config"):
        tr.train(changed)
    moved = copy.deepcopy(cfg)
    moved["train"]["num_workers"] = 2  # machine-specific settings may change between sessions
    moved["paths"]["results_dir"] = "elsewhere"
    tr.prepare_run_dirs(moved)


def test_restart_discards_the_previous_run(cfg):
    tr.train(cfg)
    changed = copy.deepcopy(cfg)
    changed["train"]["max_epochs"] = 1
    assert tr.train(changed, restart=True)["epochs"] == 1
    assert [row["epoch"] for row in read_log(changed)] == ["1"]


def test_pos_weight_run(cfg):
    cfg["train"]["pos_weight"] = True
    cfg["train"]["max_epochs"] = 1
    assert tr.train(cfg)["epochs"] == 1


def test_queue_runs_every_config_even_after_a_failure(cfg, tmp_path):
    """``--config a b c``: each experiment runs in turn; a broken one does not stop the rest."""
    import yaml

    base = ROOT / "configs" / "base.yaml"
    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text(yaml.safe_dump({"paths": cfg["paths"]}), encoding="utf-8")
    overrides = {"data": {"image_size": 64}, "model": {"pretrained": False}, "train": {
        "device": "cpu", "amp": False, "batch_size": 8, "num_workers": 0, "max_epochs": 1}}
    configs = []
    for name, extra in (("first", {}), ("broken", {"model": {"name": "resnet"}}), ("last", {})):
        body = {"extends": str(base), "experiment": name, **overrides}
        for key, value in extra.items():
            body[key] = {**body.get(key, {}), **value}
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")
        configs.append(str(path))

    with pytest.raises(SystemExit, match="broken"):
        tr.main(["--config", *configs, "--paths", str(paths_file)])
    runs = Path(cfg["paths"]["runs_dir"])
    assert (runs / "first" / "summary.json").exists() and (runs / "last" / "summary.json").exists()
    assert not (runs / "broken" / "summary.json").exists()


@pytest.mark.slow
def test_model_can_overfit_a_few_images(cfg):
    """Sanity check from the plan: the pipeline learns (training loss falls near zero)."""
    cfg["augmentation"]["enabled"] = False
    cfg["data"]["max_images"] = 16
    torch.manual_seed(0)
    loader = build_dataloader(cfg, "train")
    model = tr.build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    losses = [tr.train_one_epoch(model, loader, criterion, optimizer, scaler, torch.device("cpu"), None, epoch)[0]
              for epoch in range(60)]
    assert losses[0] > 0.4
    assert min(losses[-5:]) < 0.05
