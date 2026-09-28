import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chestxray.config import load_config
from chestxray.data import preprocess as pp

ROOT = Path(__file__).resolve().parents[1]
SIZE = 16


def png_bytes(mode: str, size: tuple[int, int]) -> bytes:
    rng = np.random.default_rng(0)
    if mode == "I;16":
        img = Image.fromarray(rng.integers(0, 4096, size[::-1], dtype=np.uint16))
    else:
        channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
        arr = rng.integers(0, 256, (*size[::-1], channels), dtype=np.uint8).squeeze()
        img = Image.fromarray(arr, mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_zip(tmp_path) -> Path:
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("images_001/images/00000001_000.png", png_bytes("L", (64, 64)))
        zf.writestr("images_001/images/00000001_001.png", png_bytes("RGBA", (80, 60)))
        zf.writestr("images_002/images/00000002_000.png", png_bytes("L", (100, 100)))
        zf.writestr("copy/images/00000002_000.png", png_bytes("L", (100, 100)))  # duplicate name
        zf.writestr("Data_Entry_2017.csv", "Image Index,Finding Labels\n")
        zf.writestr("test_list.txt", "00000002_000.png\n")
    return path


def test_convert_from_zip(fake_zip, tmp_path):
    data_dir = tmp_path / "nih256"
    manifest = pp.convert_images(fake_zip, data_dir, size=SIZE)

    assert sorted(manifest) == ["00000001_000.png", "00000001_001.png", "00000002_000.png"]
    assert all(row["status"] == "ok" for row in manifest.values())
    assert manifest["00000001_001.png"]["orig_mode"] == "RGBA"
    assert (manifest["00000001_001.png"]["orig_width"], manifest["00000001_001.png"]["orig_height"]) == (80, 60)
    for name in manifest:
        with Image.open(data_dir / "images" / name) as img:
            assert img.size == (SIZE, SIZE) and img.mode == "L"
    assert (data_dir / "Data_Entry_2017.csv").exists() and (data_dir / "test_list.txt").exists()
    assert len(pp.read_manifest(data_dir / pp.MANIFEST_NAME)) == 3


def test_rerun_skips_finished_images(fake_zip, tmp_path):
    data_dir = tmp_path / "nih256"
    pp.convert_images(fake_zip, data_dir, size=SIZE)
    images = sorted((data_dir / "images").glob("*.png"))
    mtimes = [p.stat().st_mtime_ns for p in images]
    images[0].unlink()  # simulate an image lost before the run finished

    pp.convert_images(fake_zip, data_dir, size=SIZE)
    assert images[0].exists()
    assert [p.stat().st_mtime_ns for p in images[1:]] == mtimes[1:]


def test_parallel_and_folder_source_match_serial_zip(fake_zip, tmp_path):
    serial, parallel, from_dir = tmp_path / "serial", tmp_path / "parallel", tmp_path / "from_dir"
    pp.convert_images(fake_zip, serial, size=SIZE, workers=1)
    pp.convert_images(fake_zip, parallel, size=SIZE, workers=2)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(fake_zip) as zf:
        zf.extractall(extracted)
    pp.convert_images(extracted, from_dir, size=SIZE)

    for path in (serial / "images").glob("*.png"):
        reference = np.asarray(Image.open(path))
        for other in (parallel, from_dir):
            assert np.array_equal(reference, np.asarray(Image.open(other / "images" / path.name)))
    assert (from_dir / "Data_Entry_2017.csv").exists()


def test_unsupported_mode_is_recorded_not_raised(tmp_path):
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a/ok.png", png_bytes("L", (32, 32)))
        zf.writestr("a/sixteen_bit.png", png_bytes("I;16", (32, 32)))
    manifest = pp.convert_images(zip_path, tmp_path / "out", size=SIZE)
    assert manifest["ok.png"]["status"] == "ok"
    assert manifest["sixteen_bit.png"]["status"].startswith("error: unsupported mode")
    assert pp.summarize(manifest)["error"] == 1


def test_pack_and_verify_tar(fake_zip, tmp_path):
    data_dir = tmp_path / "nih256"
    pp.convert_images(fake_zip, data_dir, size=SIZE)
    tar_path = data_dir.with_suffix(".tar")
    assert pp.pack_tar(data_dir, tar_path) == 3

    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
    assert {"Data_Entry_2017.csv", "test_list.txt", "manifest.csv", "images/00000001_000.png"} <= names
    assert pp.verify_tar(tar_path, n=100, size=SIZE) == 3


def test_real_tar_opens_100_random_images():
    """Acceptance check for Phase 1; runs once the full dataset has been preprocessed."""
    cfg = load_config(ROOT / "configs" / "base.yaml", ROOT / "configs" / "paths" / "local.yaml")
    tar_path = Path(cfg["paths"]["data_dir"]).with_suffix(".tar")
    if not tar_path.exists():
        pytest.skip(f"{tar_path} not built yet")
    assert pp.verify_tar(tar_path, n=100, size=cfg["data"]["stored_size"]) == 112_120
