"""Progress window for a queue of training runs. It only reads the logs: closing it never
stops training.

    python -m chestxray.monitor --config configs/experiments/e3_noaug.yaml configs/experiments/e4_scratch.yaml

For each experiment it shows the status, the epoch, the progress inside the epoch, the best
mean validation AUC so far, the epochs without improvement and the time left; for the queue,
the overall progress and the estimated end. Early stopping makes the number of epochs unknown
in advance, so estimates assume a typical length (:data:`TYPICAL_EPOCHS`) and are refined as
epochs and runs finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from chestxray.config import load_config

REFRESH_MS = 5000
STALE_AFTER_S = 20 * 60        # no log line for this long: something may be wrong
FIRST_EPOCH_EXTRA_S = 90       # worker start-up and warm-up of the first epoch
DEFAULT_EPOCH_S = 6.5 * 60     # RTX 2060, batch 32, until a run has measured its own
TYPICAL_EPOCHS = {True: 14, False: 30}  # pretrained on ImageNet or not (E1: 13, E2: 19)

PROGRESS_LINE = re.compile(r"epoch (\d+):\s+(\d+)% \| train loss ([\d.]+) \| (\d+) img/s")
EPOCH_LINE = re.compile(r"epoch (\d+)/(\d+) \|")

NAMES = {
    "e1_baseline": "E1 – base",
    "e2_posweight": "E2 – pos_weight",
    "e3_noaug": "E3 – sem aumento de dados",
    "e1_baseline_seed43": "E1 com seed 43",
    "e1_baseline_seed44": "E1 com seed 44",
    "e4_scratch": "E4 – sem ImageNet",
    "e5_official": "E5 – divisão oficial do NIH",
}


@dataclass
class Status:
    name: str
    state: str = "aguardando"   # aguardando | treinando | validando | concluído | interrompido | parado?
    epochs_done: int = 0
    max_epochs: int = 30
    patience: int = 5
    epoch_fraction: float = 0.0  # progress inside the current epoch (0-1)
    img_per_s: float | None = None
    best_auc: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    epoch_seconds: list[float] = field(default_factory=list)
    expected_epochs: int = 14
    stopped_early: bool | None = None
    last_update: float | None = None

    @property
    def done_units(self) -> float:
        """Epochs done, counting the current one partially."""
        if self.state == "concluído":
            return float(self.epochs_done)
        return self.epochs_done + (self.epoch_fraction if self.state in ("treinando", "validando") else 0.0)

    @property
    def total_units(self) -> float:
        """Epochs this run is expected to take (exact once it has finished)."""
        if self.state == "concluído":
            return float(self.epochs_done)
        return float(min(self.max_epochs, max(self.expected_epochs, self.epochs_done + self.min_epochs_left)))

    @property
    def min_epochs_left(self) -> int:
        """If the mean validation AUC does not improve any more, early stopping ends the run in
        ``patience - bad_epochs`` epochs; it can also run to ``max_epochs``."""
        if self.state == "concluído":
            return 0
        return max(0, min(self.max_epochs - self.epochs_done, self.patience - self.bad_epochs))


def read_log_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def experiment_status(run_dir: Path, max_epochs: int = 30, patience: int = 5, pretrained: bool = True,
                      now: float | None = None) -> Status:
    """Status of one run from its folder: summary.json, log.csv and train.log."""
    now = time.time() if now is None else now
    status = Status(run_dir.name, max_epochs=max_epochs, patience=patience, expected_epochs=TYPICAL_EPOCHS[pretrained])
    rows = read_log_rows(run_dir / "log.csv")
    status.epochs_done = len(rows)
    status.epoch_seconds = [float(r["epoch_time_s"]) for r in rows]
    for row in rows:
        auc = float(row["val_auc_mean"])
        if status.best_auc is None or auc > status.best_auc:
            status.best_auc, status.best_epoch = auc, int(row["epoch"])
    if rows:
        last_best = max(i for i, r in enumerate(rows) if r["is_best"] == "True") if any(
            r["is_best"] == "True" for r in rows) else -1
        status.bad_epochs = len(rows) - 1 - last_best

    summary = run_dir / "summary.json"
    if summary.exists():
        info = json.loads(summary.read_text(encoding="utf-8"))
        status.state, status.stopped_early = "concluído", info.get("stopped_early")
        return status

    log = run_dir / "train.log"
    if not log.exists():
        return status
    status.last_update = log.stat().st_mtime
    status.state = "treinando"
    fraction = 0.0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if (m := PROGRESS_LINE.search(line)) and int(m.group(1)) == status.epochs_done + 1:
            fraction, status.img_per_s = int(m.group(2)) / 100, float(m.group(4))
    status.epoch_fraction = min(fraction, 0.99)
    if fraction >= 0.95:
        status.state = "validando"
    if now - status.last_update > STALE_AFTER_S:
        status.state = "parado?"
    return status


def queue_status(runs: list[tuple[str, Path, dict]], now: float | None = None) -> list[Status]:
    """Statuses in queue order; a run that stopped while a later one is already going is
    marked as interrupted (the queue moves on after a failure)."""
    statuses = [experiment_status(run_dir, cfg["train"]["max_epochs"], cfg["train"]["early_stopping_patience"],
                                  cfg["model"]["pretrained"], now) for _, run_dir, cfg in runs]
    for i, s in enumerate(statuses):
        later_started = any(t.state != "aguardando" for t in statuses[i + 1:])
        if s.state in ("treinando", "validando", "parado?") and later_started:
            s.state = "interrompido"
    return statuses


def epoch_estimate(statuses: list[Status]) -> float:
    """Seconds per epoch after the first, measured on the queue's runs so far."""
    measured = [t for s in statuses for t in s.epoch_seconds[1:]]
    return sum(measured) / len(measured) if measured else DEFAULT_EPOCH_S


def seconds_left(status: Status, epoch_s: float) -> float:
    if status.state in ("concluído", "interrompido"):
        return 0.0
    left = (status.total_units - status.done_units) * epoch_s
    if status.state == "aguardando":
        left += FIRST_EPOCH_EXTRA_S
    return max(left, 0.0)


def fmt_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h{minutes % 60:02d}" if minutes >= 60 else f"{minutes} min"


def fmt_auc(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}".replace(".", ",")


def describe(status: Status, epoch_s: float) -> str:
    if status.state == "aguardando":
        return f"na fila • ~{status.expected_epochs} épocas previstas (~{fmt_duration(seconds_left(status, epoch_s))})"
    if status.state == "concluído":
        how = "parada antecipada" if status.stopped_early else "limite de épocas"
        return (f"{status.epochs_done} épocas ({how}) • melhor AUC val {fmt_auc(status.best_auc)} "
                f"na época {status.best_epoch}")
    if status.state == "interrompido":
        return f"parou na época {status.epochs_done + 1}; rode a fila de novo para retomar"
    current = status.epochs_done + 1
    parts = [f"época {current} de até {status.max_epochs}",
             "validando" if status.state == "validando" else f"{round(100 * status.epoch_fraction)}% da época"]
    if status.best_auc is not None:
        parts.append(f"melhor AUC val {fmt_auc(status.best_auc)} (época {status.best_epoch})")
        parts.append(f"{status.bad_epochs} sem melhorar (para em {status.patience})")
    parts.append(f"~{fmt_duration(seconds_left(status, epoch_s))} restantes")
    if status.state == "parado?":
        parts.append("SEM ATUALIZAÇÃO HÁ MAIS DE 20 MIN")
    return " • ".join(parts)


def gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().splitlines()[0]
        util, used, total, temp = (float(v) for v in out.split(","))
        return f"GPU: {util:.0f}% de uso • {used / 1024:.1f} de {total / 1024:.1f} GB • {temp:.0f} °C".replace(".", ",")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return "GPU: sem leitura do nvidia-smi"


def run_window(runs: list[tuple[str, Path, dict]]) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Treino – TCC raio X")
    root.geometry("860x" + str(170 + 78 * len(runs)))
    style = ttk.Style(root)
    style.configure("Big.TLabel", font=("Segoe UI", 13, "bold"))
    style.configure("Name.TLabel", font=("Segoe UI", 11, "bold"))
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)

    overall_text = ttk.Label(frame, style="Big.TLabel")
    overall_text.pack(anchor="w")
    overall_bar = ttk.Progressbar(frame, length=820, maximum=100)
    overall_bar.pack(fill="x", pady=(4, 12))
    rows = []
    for name, _, _ in runs:
        box = ttk.Frame(frame)
        box.pack(fill="x", pady=4)
        title = ttk.Label(box, text=NAMES.get(name, name), style="Name.TLabel")
        title.pack(anchor="w")
        bar = ttk.Progressbar(box, length=820, maximum=100)
        bar.pack(fill="x")
        detail = ttk.Label(box)
        detail.pack(anchor="w")
        rows.append((title, bar, detail))
    gpu = ttk.Label(frame)
    gpu.pack(anchor="w", pady=(12, 0))
    footer = ttk.Label(frame, foreground="#666666")
    footer.pack(anchor="w")

    def refresh() -> None:
        statuses = queue_status(runs)
        epoch_s = epoch_estimate(statuses)
        done = sum(s.done_units for s in statuses)
        total = sum(s.total_units for s in statuses)
        left = sum(seconds_left(s, epoch_s) for s in statuses)
        percent = 100 * done / total if total else 0
        finished = all(s.state in ("concluído", "interrompido") for s in statuses)
        if finished:
            overall_text.config(text=f"Fila terminada • {sum(s.state == 'concluído' for s in statuses)} de "
                                     f"{len(statuses)} experimentos concluídos")
            percent = 100
        else:
            end = datetime.now() + timedelta(seconds=left)
            overall_text.config(text=f"Fila: {percent:.0f}% • faltam ~{fmt_duration(left)} • "
                                     f"termina por volta das {end:%H:%M} (estimativa)")
        overall_bar["value"] = percent
        for (title, bar, detail), s in zip(rows, statuses):
            title.config(text=f"{NAMES.get(s.name, s.name)}  [{s.state}]")
            bar["value"] = 100 * s.done_units / s.total_units if s.total_units else 0
            detail.config(text=describe(s, epoch_s))
        gpu.config(text=gpu_line())
        footer.config(text=f"Atualizado às {datetime.now():%H:%M:%S}. Esta janela só lê os logs: fechá-la não "
                           f"interrompe o treino. Estimativas assumem ~{TYPICAL_EPOCHS[True]} épocas por "
                           f"experimento (30 no E4) e ficam mais precisas ao longo do treino.")
        root.after(REFRESH_MS, refresh)

    refresh()
    root.mainloop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", nargs="+", required=True, help="the experiment configs of the queue, in order")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    args = parser.parse_args(argv)
    runs = []
    for config in args.config:
        cfg = load_config(config, args.paths)
        runs.append((cfg["experiment"], Path(cfg["paths"]["runs_dir"]) / cfg["experiment"], cfg))
    run_window(runs)


if __name__ == "__main__":
    main()
