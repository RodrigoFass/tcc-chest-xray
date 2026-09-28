"""Download NIH ChestX-ray14 from Kaggle as a zip, without extracting it.

    python -m chestxray.data.download --sample   # nih-chest-xrays/sample, ~5,600 images
    python -m chestxray.data.download            # nih-chest-xrays/data, ~42 GB

The zip goes to ``paths.raw_dir``. Credentials are read by the Kaggle CLI itself, from
``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME``/``KAGGLE_KEY`` environment variables
(on Colab, set them from Secrets); they never go into a versioned file. On a Kaggle
notebook nothing needs downloading: attach the dataset and preprocess it with
``--source dir``. Re-running skips a zip that is already complete.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from chestxray.config import load_config
from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

DATASETS = {
    "full": ("nih-chest-xrays/data", 112_120),
    "sample": ("nih-chest-xrays/sample", 5_606),
}


def find_kaggle_cli() -> str:
    """The ``kaggle`` executable of the running environment, else the one on PATH."""
    scripts_dir = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        candidate = scripts_dir / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("kaggle")
    if found is None:
        raise FileNotFoundError("Kaggle CLI not found; install it with `pip install kaggle`")
    return found


def has_kaggle_credentials() -> bool:
    """Whether the Kaggle CLI will find credentials (checks presence only, never reads them)."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").is_file()


def count_zip_images(zip_path: Path) -> int:
    """Number of distinct PNG names in the zip (reads only the central directory).

    Distinct names, because the Kaggle sample stores every image twice
    (``sample/images`` and ``sample/sample/images``, byte-identical copies).
    """
    with zipfile.ZipFile(zip_path) as zf:
        return len({n.rsplit("/", 1)[-1] for n in zf.namelist() if n.lower().endswith(".png")})


def download(dataset: str, dest: Path) -> Path:
    """Download ``dataset`` into ``dest`` as ``<name>.zip`` and return the zip path."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [find_kaggle_cli(), "datasets", "download", "-d", dataset, "-p", str(dest)]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    zip_path = dest / f"{dataset.split('/')[-1]}.zip"
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"{zip_path} is missing or not a valid zip file")
    return zip_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="download the ~5,600-image sample")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    args = parser.parse_args(argv)
    setup_logging()

    if not has_kaggle_credentials():
        sys.exit(
            "Kaggle credentials not found. Save kaggle.json in ~/.kaggle/ or set "
            "KAGGLE_USERNAME and KAGGLE_KEY (see README, section 'Dados')."
        )
    dataset, expected = DATASETS["sample" if args.sample else "full"]
    dest = Path(load_config(args.config, args.paths)["paths"]["raw_dir"])
    zip_path = download(dataset, dest)

    n_images = count_zip_images(zip_path)
    logger.info("%s: %.2f GB, %d PNG images", zip_path, zip_path.stat().st_size / 1e9, n_images)
    if n_images != expected:
        logger.warning("Expected %d images in %s, found %d", expected, dataset, n_images)


if __name__ == "__main__":
    main()
