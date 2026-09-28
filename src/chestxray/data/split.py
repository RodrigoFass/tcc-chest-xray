"""Parse the NIH labels and split the images into train/val/test by patient.

    python -m chestxray.data.split
    python -m chestxray.data.split --paths configs/paths/local_sample.yaml

Reads ``Data_Entry_2017.csv`` (or the Kaggle sample's ``sample_labels.csv``) from
``data_dir`` and turns "Finding Labels" (joined by ``|``) into one binary column per
class, named exactly as in the CSV; "No Finding" means all zeros. Writes
``<splits_dir>/{train,val,test}.csv`` and ``<results_dir>/tables/prevalencia_splits.csv``.

Strategies (``split.strategy``):

- ``patient_random``: patients are shuffled with ``split.seed`` and taken whole, in that
  order, until the train and then the validation share of *images* is reached. The split
  is by patient (no patient in two splits) while the image proportions stay at 70/15/15
  even though some patients have more than a hundred images.
- ``official``: test = ``test_list.txt``; ``train_val_list.txt`` is split by patient into
  train and val keeping the configured train:val ratio.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from chestxray.config import load_config
from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

LABEL_FILES = ("Data_Entry_2017.csv", "sample_labels.csv")
NO_FINDING = "No Finding"
SPLITS = ("train", "val", "test")
SPLIT_NAMES_PT = {"train": "treino", "val": "val", "test": "teste"}
META_COLUMNS = ["image", "patient_id", "age", "sex", "view"]

# Section 1 of the plan: used to check the label parsing on the full dataset
REFERENCE_COUNTS = {
    "images": 112_120,
    "patients": 30_805,
    NO_FINDING: 60_361,
    "Atelectasis": 11_559,
    "Effusion": 13_317,
    "Pneumonia": 1_431,
    "Hernia": 227,
}


def find_labels_file(data_dir: Path) -> Path:
    for name in LABEL_FILES:
        if (data_dir / name).exists():
            return data_dir / name
    raise FileNotFoundError(f"None of {LABEL_FILES} found in {data_dir}")


def parse_age(value) -> float:
    """Age in years from ``58`` or the sample's ``"058Y"`` / ``"006M"`` / ``"020D"``."""
    match = re.fullmatch(r"\s*(\d+)\s*([YMD]?)\s*", str(value), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse age {value!r}")
    number, unit = int(match.group(1)), match.group(2).upper()
    return number / {"": 1, "Y": 1, "M": 12, "D": 365}[unit]


def load_labels(csv_path: Path, classes: list[str]) -> pd.DataFrame:
    """One row per image: image, patient_id, one 0/1 column per class, age, sex, view."""
    raw = pd.read_csv(csv_path)
    findings = raw["Finding Labels"].str.split("|").apply(lambda labels: [s.strip() for s in labels])

    unknown = set().union(*findings) - set(classes) - {NO_FINDING}
    if unknown:
        raise ValueError(f"Labels not in the class list: {sorted(unknown)}")
    mixed = findings.apply(lambda labels: NO_FINDING in labels and len(labels) > 1)
    if mixed.any():
        raise ValueError(f"{int(mixed.sum())} images combine '{NO_FINDING}' with a disease")

    df = pd.DataFrame({
        "image": raw["Image Index"],
        "patient_id": raw["Patient ID"].astype(int),
    })
    for cls in classes:
        df[cls] = findings.apply(lambda labels, c=cls: int(c in labels))
    df["age"] = raw["Patient Age"].apply(parse_age)
    df["sex"] = raw["Patient Gender"]
    df["view"] = raw["View Position"]

    if df["image"].duplicated().any():
        raise ValueError(f"{csv_path}: duplicated image names")
    return df


def assign_by_patient(patient_ids: pd.Series, fractions: dict[str, float], seed: int) -> pd.Series:
    """Split label per row, keeping each patient's images together.

    Patients are shuffled (from a sorted start, so the result depends only on the seed)
    and each one goes to the split whose share of images contains the midpoint of that
    patient's images in the cumulative count.
    """
    names = list(fractions)
    shares = np.array([fractions[n] for n in names], dtype=float)
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")

    counts = patient_ids.value_counts().sort_index()
    counts = counts.iloc[np.random.default_rng(seed).permutation(len(counts))]
    end = counts.cumsum().to_numpy() / counts.sum()
    midpoint = end - counts.to_numpy() / counts.sum() / 2
    index = np.searchsorted(np.cumsum(shares)[:-1], midpoint, side="right")
    patient_split = pd.Series(np.array(names)[index], index=counts.index)
    return patient_ids.map(patient_split)


def split_patient_random(df: pd.DataFrame, split_cfg: dict) -> pd.Series:
    fractions = {s: split_cfg[f"{s}_frac"] for s in SPLITS}
    return assign_by_patient(df["patient_id"], fractions, split_cfg["seed"])


def split_official(df: pd.DataFrame, split_cfg: dict, data_dir: Path) -> pd.Series:
    def read_list(name: str) -> set[str]:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is required by the official split")
        return set(path.read_text(encoding="utf-8").split())

    test_images, train_val_images = read_list("test_list.txt"), read_list("train_val_list.txt")
    if test_images & train_val_images or set(df["image"]) != test_images | train_val_images:
        raise ValueError("test_list.txt and train_val_list.txt must partition the label file")

    split = pd.Series("test", index=df.index)
    train_val = df["image"].isin(train_val_images)
    total = split_cfg["train_frac"] + split_cfg["val_frac"]
    fractions = {"train": split_cfg["train_frac"] / total, "val": split_cfg["val_frac"] / total}
    split[train_val] = assign_by_patient(df.loc[train_val, "patient_id"], fractions, split_cfg["seed"])
    return split


def patients_in_several_splits(df: pd.DataFrame, split: pd.Series) -> set[int]:
    per_patient = split.groupby(df["patient_id"]).nunique()
    return set(per_patient[per_patient > 1].index)


def prevalence_table(df: pd.DataFrame, split: pd.Series, classes: list[str]) -> pd.DataFrame:
    """Counts and percentages per split: images and patients (as a share of the total), then
    each class and "No Finding" (as the prevalence within the split)."""
    groups = {name: df[split == name] for name in SPLITS}
    groups["total"] = df
    no_finding = df[classes].sum(axis=1) == 0

    rows = []
    for label in ["Imagens", "Pacientes", *classes, NO_FINDING]:
        row = {"classe": label}
        for name, part in groups.items():
            pt = SPLIT_NAMES_PT.get(name, name)
            if label == "Imagens":
                n, base = len(part), len(df)
            elif label == "Pacientes":
                n, base = part["patient_id"].nunique(), df["patient_id"].nunique()
            else:
                n = int(no_finding[part.index].sum()) if label == NO_FINDING else int(part[label].sum())
                base = len(part)
            row[f"n_{pt}"] = n
            row[f"pct_{pt}"] = round(100 * n / base, 3) if base else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def prevalence_failures(table: pd.DataFrame, classes: list[str], tolerance: float = 0.20) -> list[str]:
    """Classes whose prevalence in some split differs from the overall one by more than
    ``tolerance`` (relative), e.g. Pneumonia at 1.3% overall must stay within 1.04-1.56%."""
    failures = []
    for cls in classes:
        row = table.set_index("classe").loc[cls]
        for name in SPLITS:
            pt = SPLIT_NAMES_PT[name]
            if abs(row[f"pct_{pt}"] - row["pct_total"]) > tolerance * row["pct_total"]:
                failures.append(f"{cls} in {name}: {row[f'pct_{pt}']:.2f}% vs {row['pct_total']:.2f}% overall")
    return failures


def check_reference_counts(df: pd.DataFrame, classes: list[str]) -> list[str]:
    """Differences above 1% between the parsed counts and the plan's reference numbers."""
    counts = {
        "images": len(df),
        "patients": df["patient_id"].nunique(),
        NO_FINDING: int((df[classes].sum(axis=1) == 0).sum()),
        **{c: int(df[c].sum()) for c in REFERENCE_COUNTS if c in classes},
    }
    problems = []
    for key, reference in REFERENCE_COUNTS.items():
        logger.info("  %-12s %7d (reference ~%d)", key, counts[key], reference)
        if abs(counts[key] - reference) > 0.01 * reference:
            problems.append(f"{key}: {counts[key]} vs reference ~{reference}")
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = load_config(args.config, args.paths)
    paths, classes, split_cfg = cfg["paths"], cfg["data"]["classes"], cfg["split"]
    data_dir = Path(paths["data_dir"])

    labels_file = find_labels_file(data_dir)
    df = load_labels(labels_file, classes)
    logger.info("%s: %d images, %d patients", labels_file.name, len(df), df["patient_id"].nunique())

    images_dir = data_dir / "images"
    missing = set(df["image"]) - {p.name for p in images_dir.glob("*.png")}
    if missing:
        raise SystemExit(f"{len(missing)} labelled images are missing from {images_dir}, "
                         f"e.g. {sorted(missing)[:3]}; run preprocess first")

    if labels_file.name == "Data_Entry_2017.csv":
        logger.info("Label counts vs the plan's reference numbers:")
        for problem in check_reference_counts(df, classes):
            logger.warning("Count differs from reference: %s", problem)

    strategy = split_cfg["strategy"]
    if strategy == "patient_random":
        split = split_patient_random(df, split_cfg)
    elif strategy == "official":
        split = split_official(df, split_cfg, data_dir)
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}")

    shared = patients_in_several_splits(df, split)
    if shared:
        raise RuntimeError(f"{len(shared)} patients appear in more than one split")

    splits_dir = Path(paths["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        part = df[split == name][[*META_COLUMNS[:2], *classes, *META_COLUMNS[2:]]]
        part.to_csv(splits_dir / f"{name}.csv", index=False)

    table = prevalence_table(df, split, classes)
    tables_dir = Path(paths["results_dir"]) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "prevalencia_splits.csv", index=False)

    shares = table.set_index("classe").loc["Imagens"]
    logger.info("Images: train %.2f%%, val %.2f%%, test %.2f%% (strategy %s, seed %d)",
                shares["pct_treino"], shares["pct_val"], shares["pct_teste"], strategy, split_cfg["seed"])
    focus = cfg["data"]["focus_classes"]
    logger.info("Prevalence (%%) train/val/test/total:\n%s",
                table.set_index("classe").loc[focus, ["pct_treino", "pct_val", "pct_teste", "pct_total"]])
    failures = prevalence_failures(table, focus)
    for failure in failures:
        logger.warning("Prevalence differs by more than 20%%: %s", failure)
    if not failures:
        logger.info("Prevalence of %s within 20%% across splits", ", ".join(focus))
    logger.info("Wrote %s and %s", splits_dir, tables_dir / "prevalencia_splits.csv")


if __name__ == "__main__":
    main()
