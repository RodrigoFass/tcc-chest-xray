"""Seeding, device selection, logging and recording of the software/hardware environment."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

LOGGER_NAME = "chestxray"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Libraries whose versions are stored with every run (torch is recorded separately)
TRACKED_PACKAGES = (
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "pillow",
    "grad-cam",
    "matplotlib",
    "seaborn",
    "pyyaml",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) and make cuDNN deterministic.

    ``torch.use_deterministic_algorithms(True)`` is deliberately not enabled: the backward
    pass of DenseNet's adaptive average pooling has no deterministic CUDA kernel, so strict
    mode raises an error. Two GPU runs with the same seed may still differ slightly; that
    is why the final configuration is repeated with 3 seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Return the requested device; ``"auto"`` picks CUDA when available, else CPU."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{preference}' requested but CUDA is not available")
    return device


def keep_awake() -> None:
    """Ask Windows not to go to sleep while this process runs (e.g. a long training run).

    Uses ``SetThreadExecutionState``, the call video players make: it changes no system
    setting, lets the screen turn off, and lapses by itself when the process ends. Does
    nothing on other systems.
    """
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Send log records to stdout and, optionally, to a file.

    Configures the root logger, so that modules run as scripts (whose logger is named
    ``__main__``) are covered too. Calling this again replaces only the handlers it added
    before (closing their files) and leaves others, such as pytest's, alone.
    """
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, "_chestxray", False)]:
        handler.close()
        root.removeHandler(handler)
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    # A Windows console or pipe may use a code page without characters such as "−" (the tables
    # use it): replace them, instead of losing the whole record to an encoding error
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._chestxray = True
        root.addHandler(handler)
    return logging.getLogger(LOGGER_NAME)


def get_git_commit(repo_dir: str | Path | None = None) -> str | None:
    """Current commit hash, suffixed with ``-dirty`` if tracked files have uncommitted changes.

    Defaults to the directory of this package, which is inside the repository when it is
    installed with ``pip install -e .``. Returns None when git or the repository is missing.
    """
    cwd = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if changes else commit


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_info() -> dict:
    """Python, torch, CUDA/cuDNN versions, GPU names, git commit and key library versions."""
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,  # None on CPU-only builds
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "cuda_available": cuda_available,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if cuda_available
        else [],
        "git_commit": get_git_commit(),
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
    }


def save_environment_info(path: str | Path) -> dict:
    """Write :func:`collect_environment_info` to a JSON file and return it."""
    info = collect_environment_info()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    # Quick environment check, e.g. at the start of a Colab/Kaggle notebook
    print(json.dumps(collect_environment_info(), indent=2))
