from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chestxray.config import load_config
from chestxray.data import split as sp

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml", ROOT / "configs" / "paths" / "local.yaml")
CLASSES = CFG["data"]["classes"]
FOCUS = CFG["data"]["focus_classes"]


def synthetic_patients(n_patients: int = 3000, seed: int = 0) -> pd.Series:
    """Patient id per image, with a long tail of patients who have many images."""
    rng = np.random.default_rng(seed)
    counts = np.minimum(rng.geometric(0.3, n_patients), 150)
    counts[:5] = 120  # a few very large patients, as in the NIH data
    return pd.Series(np.repeat(np.arange(n_patients), counts))


def write_labels_csv(path: Path, rows: list[tuple[str, str, int, str]]) -> Path:
    # Same header as Data_Entry_2017.csv, including its awkward bracketed columns
    header = ("Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,"
              "View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],\n")
    lines = [f"{img},{labels},0,{pid},{age},M,PA,2500,2048,0.143,0.143,\n" for img, labels, pid, age in rows]
    path.write_text(header + "".join(lines), encoding="utf-8")
    return path


@pytest.mark.parametrize("value, expected", [(58, 58), ("058Y", 58), ("006M", 0.5), ("365D", 1.0), (" 7 ", 7)])
def test_parse_age(value, expected):
    assert sp.parse_age(value) == pytest.approx(expected)


def test_parse_age_rejects_garbage():
    with pytest.raises(ValueError):
        sp.parse_age("abc")


def test_load_labels(tmp_path):
    csv = write_labels_csv(tmp_path / "Data_Entry_2017.csv", [
        ("a.png", "No Finding", 1, "40"),
        ("b.png", "Effusion|Pleural_Thickening", 1, "41"),
        ("c.png", "Pneumonia", 2, "070Y"),
    ])
    df = sp.load_labels(csv, CLASSES)
    assert list(df.columns) == ["image", "patient_id", *CLASSES, "age", "sex", "view"]
    assert df.loc[0, CLASSES].sum() == 0
    assert df.loc[1, "Effusion"] == 1 and df.loc[1, "Pleural_Thickening"] == 1
    assert df.loc[1, CLASSES].sum() == 2
    assert df.loc[2, "Pneumonia"] == 1 and df.loc[2, "age"] == 70
    assert df["patient_id"].tolist() == [1, 1, 2]


@pytest.mark.parametrize("labels, message", [("Pneumonia|Flu", "not in the class list"),
                                             ("No Finding|Edema", "combine")])
def test_load_labels_rejects_bad_labels(tmp_path, labels, message):
    csv = write_labels_csv(tmp_path / "labels.csv", [("a.png", labels, 1, "40")])
    with pytest.raises(ValueError, match=message):
        sp.load_labels(csv, CLASSES)


def test_assign_by_patient_keeps_patients_together_and_hits_image_shares():
    patients = synthetic_patients()
    split = sp.assign_by_patient(patients, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)
    assert split.groupby(patients).nunique().max() == 1
    shares = split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.01)
    assert shares["val"] == pytest.approx(0.15, abs=0.01)
    assert shares["test"] == pytest.approx(0.15, abs=0.01)


def test_assign_by_patient_depends_only_on_seed():
    patients = synthetic_patients()
    fractions = {"train": 0.70, "val": 0.15, "test": 0.15}
    first = sp.assign_by_patient(patients, fractions, seed=1)
    shuffled = patients.sample(frac=1, random_state=0)  # same patients, different row order
    assert first.equals(sp.assign_by_patient(shuffled, fractions, seed=1).sort_index())
    assert not first.equals(sp.assign_by_patient(patients, fractions, seed=2))


def test_official_split(tmp_path):
    df = pd.DataFrame({"image": [f"{i}.png" for i in range(8)], "patient_id": [0, 0, 1, 2, 3, 4, 5, 5]})
    (tmp_path / "test_list.txt").write_text("0.png\n1.png\n", encoding="utf-8")
    (tmp_path / "train_val_list.txt").write_text("\n".join(f"{i}.png" for i in range(2, 8)), encoding="utf-8")
    split = sp.split_official(df, CFG["split"], tmp_path)
    assert split[:2].tolist() == ["test", "test"]
    assert set(split[2:]) <= {"train", "val"}
    assert split[6] == split[7]  # same patient


def test_prevalence_table_and_failures():
    df = pd.DataFrame({"patient_id": range(8), **{c: 0 for c in CLASSES}})
    df.loc[[0, 4], "Pneumonia"] = 1  # 1 of 4 in train, 1 of 2 in val, 0 of 2 in test
    split = pd.Series(["train"] * 4 + ["val"] * 2 + ["test"] * 2)
    table = sp.prevalence_table(df, split, CLASSES).set_index("classe")
    assert table.loc["Imagens", "pct_treino"] == 50
    assert table.loc["Pneumonia", ["pct_treino", "pct_val", "pct_teste", "pct_total"]].tolist() == [25, 50, 0, 25]
    assert table.loc["No Finding", "n_total"] == 6
    failures = sp.prevalence_failures(table.reset_index(), ["Pneumonia"])
    assert len(failures) == 2  # val (50% vs 25%) and test (0% vs 25%)


# Acceptance checks for Phase 1, run on the versioned splits once they exist
SPLITS_DIR = Path(CFG["paths"]["splits_dir"])
real_splits = pytest.mark.skipif(not (SPLITS_DIR / "train.csv").exists(), reason="splits not generated yet")


@pytest.fixture(scope="module")
def saved_splits():
    return {name: pd.read_csv(SPLITS_DIR / f"{name}.csv") for name in sp.SPLITS}


@real_splits
def test_saved_splits_share_no_patient(saved_splits):
    patients = [set(df["patient_id"]) for df in saved_splits.values()]
    assert not (patients[0] & patients[1] or patients[0] & patients[2] or patients[1] & patients[2])


@real_splits
def test_saved_splits_proportions(saved_splits):
    sizes = {name: len(df) for name, df in saved_splits.items()}
    total = sum(sizes.values())
    assert total == 112_120
    assert 0.69 <= sizes["train"] / total <= 0.71
    assert 0.14 <= sizes["val"] / total <= 0.16
    assert 0.14 <= sizes["test"] / total <= 0.16


@real_splits
def test_saved_splits_prevalence_within_20_percent(saved_splits):
    everything = pd.concat(saved_splits.values())
    for cls in FOCUS:
        overall = everything[cls].mean()
        for name, df in saved_splits.items():
            assert abs(df[cls].mean() - overall) <= 0.20 * overall, f"{cls} in {name}"
