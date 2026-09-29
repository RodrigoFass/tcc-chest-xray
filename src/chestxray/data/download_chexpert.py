"""Download the CheXpert validation set (images and radiologist labels) from Stanford AIMI on Redivis.

    python -m chestxray.data.download_chexpert --out E:/datasets/chexpert

Stanford AIMI no longer distributes CheXpert's original ``valid.csv``. What the external
validation needs (plan, Phase 7, and chestxray.external) now comes from two AIMI datasets:

- ``PNG_valid/``: the 234 validation images (~700 MB), from CheXpert Plus (``chexpert_plus``);
- ``CheXlocalize/gt_annotations_val.json``: the radiologists' annotations of those images, from
  CheXlocalize (``chexlocalize``), which list each image's positive observations.

CheXpert Plus's own labels are extracted automatically from the reports, so they are not fetched,
and neither is anything else (training images, DICOM, segmentations, heatmaps). Access needs a
Redivis account with the AIMI membership and the Stanford research agreement accepted for both
datasets. The first run opens the browser to sign in; the client keeps its credentials in
``~/.redivis``, so nothing secret goes into the project. Needs ``pip install redivis`` (extra
``data``). Files are fetched one at a time and a file already on disk with the server's size is
skipped, so a rerun resumes: the client's whole-folder download stalled near the end twice.
"""

from __future__ import annotations

import argparse
import functools
import logging
from pathlib import Path

from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

ORGANIZATION = "AIMI"
# (dataset, table, folder under --out, files to keep: None for all)
SOURCES = [
    ("chexpert_plus", "png_valid", "PNG_valid", None),
    ("chexlocalize", "chexlocalize", "CheXlocalize", {"gt_annotations_val.json"}),
]


def download(out: Path) -> None:
    try:
        import niquests
        import redivis
    except ImportError as err:
        raise SystemExit("The Redivis client is missing: pip install redivis") from err

    # The client uses HTTP/3 (QUIC, on UDP) when the server offers it; behind a VPN such as
    # Cloudflare WARP the QUIC handshake fails and every file errors out. HTTP/2 over TCP works.
    for session in (niquests.Session, niquests.AsyncSession):
        session.__init__ = functools.partialmethod(session.__init__, disable_http3=True)

    organization = redivis.organization(ORGANIZATION)
    for dataset_name, table_name, folder, keep in SOURCES:
        directory = organization.dataset(dataset_name).table(table_name).to_directory()
        files = [f for f in directory.list(mode="files", recursive=True)
                 if keep is None or Path(str(f.path)).name in keep]
        logger.info("%s: %d files to %s", table_name, len(files), out / folder)
        fetched = 0
        for i, file in enumerate(files, start=1):
            target = out / folder / (Path("/") / file.path).relative_to(directory.path)
            if target.is_file() and target.stat().st_size == int(file.size):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            file.download(path=str(target), overwrite=True, progress=False)
            if target.stat().st_size != int(file.size):
                raise SystemExit(f"{target}: got {target.stat().st_size} bytes, expected {file.size}")
            fetched += 1
            if fetched % 20 == 0:
                logger.info("  %d of %d", i, len(files))
        logger.info("%s: %d downloaded, %d already there", table_name, fetched, len(files) - fetched)
    logger.info("Done: %s", out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("E:/datasets/chexpert"),
                        help="folder that receives PNG_valid/ and 'CheXpert Labels'/")
    args = parser.parse_args(argv)
    setup_logging()
    download(args.out)


if __name__ == "__main__":
    main()
