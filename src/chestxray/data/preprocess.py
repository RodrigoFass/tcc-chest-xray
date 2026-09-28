"""Convert the NIH PNGs to 256x256 grayscale once and pack them into a single tar.

    python -m chestxray.data.preprocess --input E:/datasets/nih/data.zip
    python -m chestxray.data.preprocess --input /kaggle/input/data   # extracted folder

Each PNG is read straight from the Kaggle zip (never extracted) or from an extracted
folder, converted with ``.convert("L")`` (a few NIH images are RGBA), resized to 256x256
with Lanczos and written to ``<data_dir>/images/<name>.png``. Work runs in parallel and
is resumable: images already marked ``ok`` in ``<data_dir>/manifest.csv`` are skipped, and
each image is written atomically, so an interrupted run leaves no truncated files. The
manifest records every image's original mode and size and its status.

At the end the images, the manifest and the metadata files (Data_Entry_2017.csv etc.) are
packed into ``<data_dir>.tar`` (e.g. ``data/nih256.tar``), ready to be moved to another
training environment, and 100 random images are opened from it as a check. CPU only.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import multiprocessing as mp
import os
import random
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from chestxray.config import load_config
from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

METADATA_FILES = (
    "Data_Entry_2017.csv",
    "BBox_List_2017.csv",
    "train_val_list.txt",
    "test_list.txt",
    "sample_labels.csv",  # the Kaggle sample's label file
)
SUPPORTED_MODES = {"L", "LA", "P", "RGB", "RGBA"}
MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = ["image", "source", "orig_mode", "orig_width", "orig_height", "status"]
IMAGES_SUBDIR = "images"

# Per-worker state, set by _init_worker
_zip: zipfile.ZipFile | None = None
_out_dir: Path | None = None
_size: int = 256


def detect_source(input_path: Path) -> str:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found: download it first or pass --input")
    if input_path.is_dir():
        return "dir"
    if zipfile.is_zipfile(input_path):
        return "zip"
    raise ValueError(f"{input_path} is neither a directory nor a zip file")


def list_sources(input_path: Path, source: str) -> tuple[dict[str, str], dict[str, str]]:
    """Map image name -> location, and metadata file name -> location.

    A location is a zip member name (``source="zip"``) or a file path (``"dir"``). Image
    names are the PNG basenames, which are unique in the NIH dataset; if a name appears
    more than once only the first location (in sorted order) is kept.
    """
    if source == "zip":
        with zipfile.ZipFile(input_path) as zf:
            locations = sorted(zf.namelist())
    else:
        locations = sorted(str(p) for p in input_path.rglob("*") if p.is_file())

    images: dict[str, str] = {}
    metadata: dict[str, str] = {}
    duplicates = 0
    for loc in locations:
        name = loc.replace("\\", "/").rsplit("/", 1)[-1]
        if name.lower().endswith(".png"):
            if name in images:
                duplicates += 1
            else:
                images[name] = loc
        elif name in METADATA_FILES and name not in metadata:
            metadata[name] = loc
    if duplicates:
        logger.warning("%d duplicate image names ignored (kept the first location)", duplicates)
    return images, metadata


def _init_worker(zip_path: str | None, out_dir: str, size: int) -> None:
    global _zip, _out_dir, _size
    _zip = zipfile.ZipFile(zip_path) if zip_path else None
    _out_dir = Path(out_dir)
    _size = size


def _close_worker() -> None:
    global _zip
    if _zip is not None:
        _zip.close()
        _zip = None


def _process_one(task: tuple[str, str]) -> dict:
    """Convert one image; never raises, so one bad file does not stop the run."""
    name, location = task
    row = {"image": name, "source": location, "orig_mode": "", "orig_width": "",
           "orig_height": "", "status": "ok"}
    try:
        data = _zip.read(location) if _zip is not None else Path(location).read_bytes()
        with Image.open(io.BytesIO(data)) as img:
            row["orig_mode"] = img.mode
            row["orig_width"], row["orig_height"] = img.size
            if img.mode not in SUPPORTED_MODES:
                raise ValueError(f"unsupported mode {img.mode}")
            small = img.convert("L").resize((_size, _size), Image.Resampling.LANCZOS)
        out = _out_dir / name
        tmp = out.with_name(out.name + ".tmp")
        small.save(tmp, format="PNG")
        os.replace(tmp, out)
    except Exception as exc:  # recorded in the manifest and reported at the end
        row["status"] = f"error: {exc}"
    return row


def read_manifest(path: Path) -> dict[str, dict]:
    """Rows by image name; the manifest is appended to, so the last row of a name wins."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["image"]: row for row in csv.DictReader(f)}


def write_manifest(rows: dict[str, dict], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))
    os.replace(tmp, path)


def convert_images(
    input_path: Path, data_dir: Path, source: str = "auto", size: int = 256, workers: int = 1
) -> dict[str, dict]:
    """Convert every image not yet done and copy the metadata files; return the manifest."""
    source = detect_source(input_path) if source == "auto" else source
    images, metadata = list_sources(input_path, source)
    logger.info("Found %d images and metadata %s in %s", len(images), sorted(metadata), input_path)

    out_dir = data_dir / IMAGES_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for tmp in out_dir.glob("*.tmp"):  # leftovers from an interrupted run
        tmp.unlink()

    manifest_path = data_dir / MANIFEST_NAME
    manifest = read_manifest(manifest_path)
    todo = [
        (name, loc) for name, loc in images.items()
        if manifest.get(name, {}).get("status") != "ok" or not (out_dir / name).exists()
    ]
    logger.info("%d images already done, %d to convert", len(images) - len(todo), len(todo))

    zip_arg = str(input_path) if source == "zip" else None
    init_args = (zip_arg, str(out_dir), size)
    new_file = not manifest_path.exists()
    # Rows are appended as images finish, so an interrupted run (Ctrl+C, crash) keeps its
    # progress; images whose row was not written yet are simply converted again next time.
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if new_file:
            writer.writeheader()

        def record(results) -> None:
            for row in tqdm(results, total=len(todo), desc="Converting", unit="img"):
                writer.writerow(row)
                manifest[row["image"]] = row

        if workers > 1:
            # Leaving the block terminates the workers, also on Ctrl+C
            with mp.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
                record(pool.imap_unordered(_process_one, todo, chunksize=32))
        else:
            _init_worker(*init_args)
            try:
                record(map(_process_one, todo))
            finally:
                _close_worker()
    write_manifest(manifest, manifest_path)

    if source == "zip":
        with zipfile.ZipFile(input_path) as zf:
            for name, member in metadata.items():
                (data_dir / name).write_bytes(zf.read(member))
    else:
        for name, path in metadata.items():
            (data_dir / name).write_bytes(Path(path).read_bytes())
    return manifest


def summarize(manifest: dict[str, dict]) -> Counter:
    statuses = Counter("ok" if r["status"] == "ok" else "error" for r in manifest.values())
    modes = Counter(r["orig_mode"] for r in manifest.values())
    sizes = Counter(f'{r["orig_width"]}x{r["orig_height"]}' for r in manifest.values())
    logger.info("Status: %s", dict(statuses))
    logger.info("Original modes: %s", dict(modes))
    logger.info("Original sizes (top 5): %s", dict(sizes.most_common(5)))
    for row in manifest.values():
        if row["status"] != "ok":
            logger.error("%s: %s", row["image"], row["status"])
    return statuses


def pack_tar(data_dir: Path, tar_path: Path) -> int:
    """Pack images/, the manifest and the metadata files into an uncompressed tar."""
    images = sorted((data_dir / IMAGES_SUBDIR).glob("*.png"))
    extras = [data_dir / n for n in (*METADATA_FILES, MANIFEST_NAME) if (data_dir / n).exists()]
    tmp = tar_path.with_name(tar_path.name + ".tmp")
    with tarfile.open(tmp, "w") as tar:  # PNGs are already compressed
        for path in tqdm([*extras, *images], desc="Packing", unit="file"):
            tar.add(path, arcname=path.relative_to(data_dir).as_posix())
    os.replace(tmp, tar_path)
    return len(images)


def verify_tar(tar_path: Path, n: int = 100, size: int = 256, seed: int = 0) -> int:
    """Open ``n`` random images from the tar; return the number of images it contains."""
    with tarfile.open(tar_path) as tar:
        names = [m for m in tar.getnames() if m.startswith(f"{IMAGES_SUBDIR}/") and m.endswith(".png")]
        for name in random.Random(seed).sample(names, min(n, len(names))):
            with Image.open(tar.extractfile(name)) as img:
                img.load()
                if img.size != (size, size) or img.mode != "L":
                    raise ValueError(f"{name}: expected {size}x{size} L, got {img.size} {img.mode}")
    return len(names)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", help="Kaggle zip or extracted folder (default: <raw_dir>/data.zip)")
    parser.add_argument("--source", choices=["auto", "zip", "dir"], default="auto")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--no-tar", action="store_true", help="skip packing nih256.tar")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = load_config(args.config, args.paths)
    data_dir = Path(cfg["paths"]["data_dir"])
    size = cfg["data"]["stored_size"]
    input_path = Path(args.input) if args.input else Path(cfg["paths"]["raw_dir"]) / "data.zip"

    manifest = convert_images(input_path, data_dir, args.source, size, args.workers)
    if summarize(manifest)["error"]:
        raise SystemExit("Some images failed (see above); fix and re-run, done images are kept")
    if args.no_tar:
        return
    tar_path = data_dir.with_suffix(".tar")
    n_packed = pack_tar(data_dir, tar_path)
    n_in_tar = verify_tar(tar_path, size=size)
    logger.info("%s: %d images, %.2f GB; 100 random images opened fine",
                tar_path, n_in_tar, tar_path.stat().st_size / 1e9)
    assert n_in_tar == n_packed


if __name__ == "__main__":
    main()
