"""Train one experiment: checkpoints every epoch, automatic resume, early stopping.

    python -m chestxray.train --config configs/experiments/e1_baseline.yaml
    python -m chestxray.train --config configs/debug.yaml --restart   # 200 images, 1 epoch, CPU

Each experiment gets its own folders, named after ``experiment`` in the config:

- ``<runs_dir>/<experiment>/`` (versioned, no weights): ``config.yaml`` (resolved config),
  ``environment.json`` (versions, GPU and commit of every session), ``log.csv`` (one row per
  epoch), ``train.log`` and, when training ends, ``summary.json``.
- ``<checkpoint_dir>/<experiment>/``: ``last.pt`` (all state needed to resume, saved every
  epoch) and ``best.pt`` (weights at the best mean validation AUC).

Running the same command again resumes from ``last.pt``; a finished experiment is not run
again, and an experiment folder is never reused with a different config (plan, section 7).
``--restart`` deletes the experiment's folders first; it is meant for debug runs.

The recipe follows plan section 2 and Phase 2: BCEWithLogitsLoss (with ``pos_weight`` from
the training labels when enabled), Adam, ReduceLROnPlateau and early stopping on the mean
validation AUC, and mixed precision on GPU: fp16 with a GradScaler before Ampere (RTX 20xx,
T4), bf16 from Ampere on (RTX 30xx/40xx, A100). Validation during training also runs under
mixed precision; the final predictions of Phase 3 are computed in fp32.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from chestxray.config import load_config, save_config
from chestxray.data.dataset import build_dataloader
from chestxray.metrics import mean_auc, per_class_auc
from chestxray.models.densenet import build_model
from chestxray.utils import collect_environment_info, get_device, keep_awake, set_seed, setup_logging

logger = logging.getLogger(__name__)

PROGRESS_LINES = 10  # progress messages per training epoch
MEMORY_WARNING = 0.9  # share of GPU memory above which Windows may silently spill to system RAM


def save_atomic(obj, path: Path) -> None:
    """``torch.save`` to a temporary file, then rename: a crash never leaves a broken file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def select_amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    """bf16 where the GPU supports it natively (compute capability 8.0+), fp16 on older GPUs,
    none on CPU or with AMP disabled."""
    if not enabled or device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def compute_pos_weight(labels: torch.Tensor, max_weight: float | None = None) -> torch.Tensor:
    """negatives / positives per class in the training labels (plan 3.3), optionally capped."""
    positives = labels.sum(dim=0)
    weight = (len(labels) - positives) / positives.clamp(min=1)
    return weight.clamp(max=max_weight) if max_weight is not None else weight


def rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader_generator.get_state(),
    }


def restore_rng_state(state: dict, loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader"])


def comparable(cfg: dict) -> dict:
    """What must match to resume a run: everything but the machine-specific paths and the
    number of loader workers."""
    cfg = copy.deepcopy(cfg)
    cfg.pop("paths", None)
    cfg.get("train", {}).pop("num_workers", None)
    return cfg


def prepare_run_dirs(cfg: dict, restart: bool = False) -> tuple[Path, Path]:
    """Create (or check before reusing) the experiment's run and checkpoint folders."""
    name = cfg["experiment"]
    run_dir = Path(cfg["paths"]["runs_dir"]) / name
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / name
    if restart:
        for folder in (run_dir, ckpt_dir):
            shutil.rmtree(folder, ignore_errors=True)

    saved_config = run_dir / "config.yaml"
    if saved_config.exists():
        previous = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
        if comparable(previous) != comparable(cfg):
            raise SystemExit(
                f"{run_dir} holds a run of '{name}' made with a different config. Give the new "
                "experiment its own name (plan, section 7), or use --restart to discard the old run."
            )
    if (run_dir / "log.csv").exists() and not (ckpt_dir / "last.pt").exists():
        raise SystemExit(f"{run_dir} has a training log but {ckpt_dir / 'last.pt'} is missing: "
                         "copy the checkpoint back to resume, or use --restart.")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir


def log_columns(classes: list[str]) -> list[str]:
    return ["epoch", "lr", "train_loss", "val_loss", "val_auc_mean",
            *[f"val_auc_{c}" for c in classes], "epoch_time_s", "train_img_per_s", "is_best"]


def write_log(history: list[dict], classes: list[str], path: Path) -> None:
    """Rewrite log.csv from the history stored in the checkpoint, so that a resumed run never
    duplicates or loses a row."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_columns(classes))
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp, path)


def record_session(run_dir: Path, start_epoch: int, device: torch.device, amp_dtype) -> None:
    """Append this session's software/hardware versions to environment.json."""
    path = run_dir / "environment.json"
    sessions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    sessions.append({
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "start_epoch": start_epoch,
        "device": str(device),
        "amp": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        **collect_environment_info(),
    })
    path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_dtype, epoch: int):
    """One pass over the training set; returns (mean loss, images per second)."""
    model.train()
    total_loss = torch.zeros((), device=device)
    seen = 0
    start = time.perf_counter()
    report_every = max(1, len(loader) // PROGRESS_LINES)
    for step, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
        loss = criterion(logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach() * len(images)
        seen += len(images)
        if step % report_every == 0 and step < len(loader):
            logger.info("epoch %d: %3.0f%% | train loss %.4f | %.0f img/s", epoch, 100 * step / len(loader),
                        total_loss.item() / seen, seen / (time.perf_counter() - start))
    return total_loss.item() / seen, seen / (time.perf_counter() - start)


@torch.no_grad()
def validate(model, loader, criterion, device, amp_dtype) -> tuple[float, np.ndarray]:
    """Mean loss and per-class AUC on the validation set."""
    model.eval()
    total_loss = torch.zeros((), device=device)
    scores, targets = [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
        logits = logits.float()
        total_loss += criterion(logits, labels) * len(images)
        scores.append(torch.sigmoid(logits).cpu())
        targets.append(labels.cpu())
    y_true, y_score = torch.cat(targets).numpy(), torch.cat(scores).numpy()
    return total_loss.item() / len(y_true), per_class_auc(y_true, y_score)


def train(cfg: dict, restart: bool = False) -> dict:
    """Train (or resume) the experiment described by ``cfg``; return its summary."""
    run_dir, ckpt_dir = prepare_run_dirs(cfg, restart)
    setup_logging(run_dir / "train.log")
    try:
        return _train(cfg, run_dir, ckpt_dir)
    finally:
        setup_logging()  # closes train.log


def _train(cfg: dict, run_dir: Path, ckpt_dir: Path) -> dict:
    name, classes, train_cfg = cfg["experiment"], cfg["data"]["classes"], cfg["train"]
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    state = torch.load(last_path, map_location="cpu", weights_only=False) if last_path.exists() else None
    if state is not None and state["finished"]:
        logger.info("Experiment '%s' already finished; nothing to do (%s)", name, run_dir)
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    keep_awake()
    save_config(cfg, run_dir / "config.yaml")
    set_seed(cfg["seed"])
    device = get_device(train_cfg["device"])
    amp_dtype = select_amp_dtype(device, train_cfg["amp"])
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    model = build_model(cfg).to(device)
    pos_weight = None
    if train_cfg["pos_weight"]:
        pos_weight = compute_pos_weight(train_loader.dataset.labels, train_cfg["pos_weight_max"])
        logger.info("pos_weight: %s", {c: round(float(w), 1) for c, w in zip(classes, pos_weight)})
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=train_cfg["scheduler"]["factor"], patience=train_cfg["scheduler"]["patience"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    history: list[dict] = []
    best_auc, best_epoch, bad_epochs, start_epoch = float("-inf"), 0, 0, 1
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state["scaler"]:  # empty when saved without fp16 (CPU, or a bf16 GPU)
            scaler.load_state_dict(state["scaler"])
        history, best_auc, best_epoch = state["history"], state["best_auc"], state["best_epoch"]
        bad_epochs, start_epoch = state["bad_epochs"], state["epoch"] + 1
        restore_rng_state(state["rng"], train_loader.generator)
        write_log(history, classes, run_dir / "log.csv")
        logger.info("Resuming '%s' at epoch %d (best mean val AUC %.4f at epoch %d)",
                    name, start_epoch, best_auc, best_epoch)
    record_session(run_dir, start_epoch, device, amp_dtype)
    logger.info("Experiment '%s': %d train / %d val images, device %s, AMP %s, batch %d",
                name, len(train_loader.dataset), len(val_loader.dataset), device, amp_dtype,
                train_cfg["batch_size"])

    max_epochs, patience = train_cfg["max_epochs"], train_cfg["early_stopping_patience"]
    focus = cfg["data"]["focus_classes"]
    for epoch in range(start_epoch, max_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        start = time.perf_counter()
        train_loss, img_per_s = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                                device, amp_dtype, epoch)
        val_loss, aucs = validate(model, val_loader, criterion, device, amp_dtype)
        epoch_time = time.perf_counter() - start

        auc = mean_auc(aucs)
        improved = not np.isnan(auc) and auc > best_auc
        if not np.isnan(auc):
            scheduler.step(auc)
        if improved:
            best_auc, best_epoch, bad_epochs = auc, epoch, 0
            save_atomic({"model": model.state_dict(), "epoch": epoch, "val_auc_mean": auc,
                         "val_auc": dict(zip(classes, aucs.tolist())), "classes": classes, "config": cfg},
                        best_path)
        else:
            bad_epochs += 1

        history.append({
            "epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss, "val_auc_mean": auc,
            **{f"val_auc_{c}": a for c, a in zip(classes, aucs.tolist())},
            "epoch_time_s": round(epoch_time, 1), "train_img_per_s": round(img_per_s, 1), "is_best": improved,
        })
        finished = bad_epochs >= patience or epoch == max_epochs
        save_atomic({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_auc": best_auc, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "history": history, "rng": rng_state(train_loader.generator), "finished": finished,
        }, last_path)
        write_log(history, classes, run_dir / "log.csv")

        focus_text = ", ".join(f"{c} {a:.3f}" for c, a in zip(classes, aucs) if c in focus)
        logger.info("epoch %d/%d | lr %.0e | train loss %.4f | val loss %.4f | val AUC %.4f (%s)%s | %.1f min",
                    epoch, max_epochs, lr, train_loss, val_loss, auc, focus_text,
                    " | best" if improved else f" | {bad_epochs} without improvement", epoch_time / 60)
        if device.type == "cuda":
            used = torch.cuda.max_memory_reserved(device) / torch.cuda.get_device_properties(device).total_memory
            if used > MEMORY_WARNING:
                logger.warning("GPU memory %.0f%% full: on Windows this may silently spill to system RAM "
                               "and slow training a lot; consider a smaller batch_size", 100 * used)
        if finished:
            break

    best_row = history[best_epoch - 1] if best_epoch else {}
    summary = {
        "experiment": name,
        "epochs": len(history),
        "stopped_early": bad_epochs >= patience,
        "best_epoch": best_epoch,
        "best_val_auc_mean": best_auc if best_epoch else None,
        "best_val_auc": {c: best_row.get(f"val_auc_{c}") for c in classes},
        "train_time_min": round(sum(r["epoch_time_s"] for r in history) / 60, 1),
        "first_epoch_min": round(history[0]["epoch_time_s"] / 60, 1),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "amp": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "batch_size": train_cfg["batch_size"],
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Finished '%s': best mean val AUC %.4f at epoch %d of %d (%s)", name, best_auc, best_epoch,
                len(history), "early stopping" if summary["stopped_early"] else "max_epochs")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", nargs="+", required=True,
                        help="one or more experiment configs, trained one after the other")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    parser.add_argument("--restart", action="store_true",
                        help="delete this experiment's run and checkpoints first (for debug runs)")
    args = parser.parse_args(argv)
    setup_logging()
    if args.restart and len(args.config) > 1:
        parser.error("--restart works with a single config")

    # A queue (e.g. overnight): a failing experiment is reported and the next one still runs.
    # Running the same command again resumes unfinished experiments and skips finished ones.
    failed = []
    for config in args.config:
        try:
            train(load_config(config, args.paths), restart=args.restart)
        except (Exception, SystemExit) as error:
            logger.error("Experiment %s failed: %s", config, error, exc_info=not isinstance(error, SystemExit))
            failed.append(config)
    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
