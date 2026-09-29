import csv
import os
from pathlib import Path

import pytest

from chestxray import monitor as mon

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Atelectasis"]


def write_run(run_dir: Path, aucs: list[float], progress: list[str] = (), summary: bool = False) -> Path:
    run_dir.mkdir(parents=True)
    best = -1.0
    with (run_dir / "log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_auc_mean", "epoch_time_s", "is_best"])
        writer.writeheader()
        for i, auc in enumerate(aucs, start=1):
            writer.writerow({"epoch": i, "val_auc_mean": auc, "epoch_time_s": 400 if i > 1 else 500,
                             "is_best": auc > best})
            best = max(best, auc)
    lines = [f"2026-09-28 22:00:00 INFO __main__: epoch {i}/30 | lr 1e-04 | val AUC {a}" for i, a in enumerate(aucs, 1)]
    (run_dir / "train.log").write_text("\n".join([*lines, *progress]) + "\n", encoding="utf-8")
    if summary:
        (run_dir / "summary.json").write_text('{"stopped_early": true}', encoding="utf-8")
    return run_dir


def test_running_experiment(tmp_path):
    run = write_run(tmp_path / "e3_noaug", [0.80, 0.82, 0.81],
                    ["2026-09-28 22:05:00 INFO __main__: epoch 4:  40% | train loss 0.1400 | 225 img/s"])
    s = mon.experiment_status(run)
    assert (s.state, s.epochs_done, s.best_auc, s.best_epoch, s.bad_epochs) == ("treinando", 3, 0.82, 2, 1)
    assert s.epoch_fraction == pytest.approx(0.4) and s.img_per_s == 225
    assert s.min_epochs_left == 4                  # 5 epochs without improvement minus the 1 already seen
    assert s.total_units == 14 and s.done_units == pytest.approx(3.4)
    assert "época 4 de até 30" in mon.describe(s, 400) and "40% da época" in mon.describe(s, 400)


def test_validating_waiting_and_stale(tmp_path):
    run = write_run(tmp_path / "a", [0.8], ["x INFO __main__: epoch 2: 100% | train loss 0.1 | 220 img/s"])
    assert mon.experiment_status(run).state == "validando"
    assert mon.experiment_status(tmp_path / "missing").state == "aguardando"
    later = os.path.getmtime(run / "train.log") + mon.STALE_AFTER_S + 1
    assert mon.experiment_status(run, now=later).state == "parado?"


def test_scratch_run_expects_all_epochs(tmp_path):
    run = write_run(tmp_path / "e4_scratch", [0.6, 0.65])
    assert mon.experiment_status(run, pretrained=False).total_units == 30


def test_finished_real_run():
    run = ROOT / "results" / "runs" / "e1_baseline"
    if not (run / "summary.json").exists():
        pytest.skip("E1 not trained")
    s = mon.experiment_status(run)
    assert (s.state, s.epochs_done, s.best_epoch, s.stopped_early) == ("concluído", 13, 8, True)
    assert s.total_units == s.done_units == 13
    assert mon.seconds_left(s, 400) == 0


def test_queue_marks_a_run_the_queue_left_behind_as_interrupted(tmp_path):
    cfg = {"train": {"max_epochs": 30, "early_stopping_patience": 5}, "model": {"pretrained": True}}
    first = write_run(tmp_path / "first", [0.8])  # no summary: it stopped
    second = write_run(tmp_path / "second", [0.8])
    statuses = mon.queue_status([("first", first, cfg), ("second", second, cfg), ("third", tmp_path / "third", cfg)])
    assert [s.state for s in statuses] == ["interrompido", "treinando", "aguardando"]
    assert mon.epoch_estimate(statuses) == mon.DEFAULT_EPOCH_S  # only first epochs measured so far
    assert mon.fmt_duration(3 * 3600 + 5 * 60) == "3h05" and mon.fmt_duration(20 * 60) == "20 min"
