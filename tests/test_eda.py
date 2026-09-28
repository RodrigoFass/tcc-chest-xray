import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from chestxray import eda
from chestxray.config import load_config
from chestxray.data.split import NO_FINDING, load_labels
from chestxray.plotting import _tick_label, br_number, pt

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
CLASSES = BASE["data"]["classes"]


@pytest.mark.parametrize("value, decimals, expected", [
    (11559.5, 1, "11.559,5"), (0.25, 2, "0,25"), (1000, 0, "1.000"), (-3.5, 1, "-3,5"),
])
def test_br_number(value, decimals, expected):
    assert br_number(value, decimals) == expected


@pytest.mark.parametrize("value, expected", [(0.5, "0,5"), (1000, "1.000"), (40, "40"), (0.05, "0,05")])
def test_tick_labels_use_decimal_comma(value, expected):
    assert _tick_label(value) == expected


def test_portuguese_names_cover_every_class():
    assert all(pt(c) != c or c in ("Pneumonia", "Edema") for c in CLASSES)
    assert pt(NO_FINDING) == "Sem achados"


@pytest.fixture
def data_dir(tmp_path) -> Path:
    """A tiny dataset in the Data_Entry_2017.csv format, with 256x256 images."""
    rng = np.random.default_rng(0)
    findings = ["No Finding", "Pneumonia", "Effusion", "Atelectasis", "Effusion|Atelectasis",
                "No Finding", "Pneumonia", "Effusion", "Atelectasis", "Hernia"]
    rows = []
    (tmp_path / "images").mkdir()
    for i, finding in enumerate(findings):
        name = f"{i:08d}_000.png"
        Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8), "L").save(tmp_path / "images" / name)
        rows.append({
            "Image Index": name, "Finding Labels": finding, "Follow-up #": 0, "Patient ID": i // 2,
            "Patient Age": 150 if i == 0 else 30 + i, "Patient Gender": "MF"[i % 2],
            "View Position": "AP" if i < 4 else "PA",
        })
    pd.DataFrame(rows).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    return tmp_path


@pytest.fixture
def labels(data_dir) -> pd.DataFrame:
    return load_labels(data_dir / "Data_Entry_2017.csv", CLASSES)


def test_class_counts(labels):
    counts = eda.class_counts(labels, CLASSES).set_index("classe")
    assert counts.loc["Effusion", "n"] == 3 and counts.loc["Effusion", "pct"] == pytest.approx(30)
    assert counts.loc["Pneumonia", "pct_ap"] == pytest.approx(50)  # images 1 (AP) and 6 (PA)
    assert counts.loc[NO_FINDING, "n"] == 2
    assert counts.loc["Hernia", "classe_pt"] == "Hérnia"


def test_cooccurrence(labels):
    matrix = eda.cooccurrence(labels, CLASSES)
    assert (matrix.to_numpy() == matrix.to_numpy().T).all()
    assert matrix.loc["Effusion", "Atelectasis"] == 1
    assert matrix.loc["Effusion", "Effusion"] == 3


def test_summary_table(labels):
    summary = eda.summary_table(labels, CLASSES).set_index("metrica")["valor"]
    assert summary["Imagens"] == 10 and summary["Pacientes"] == 5
    assert summary["Idades acima de 100 anos (excluídas do gráfico)"] == 1
    assert summary["Imagens com 2 ou mais doenças (%)"] == pytest.approx(10)


def test_pick_examples_only_single_label_and_seeded(labels):
    picks = eda.pick_examples(labels, CLASSES, [NO_FINDING, "Effusion"], per_class=2, seed=0)
    assert set(picks["Effusion"]) == {"00000002_000.png", "00000007_000.png"}  # not the Effusion|Atelectasis one
    assert set(picks[NO_FINDING]) == {"00000000_000.png", "00000005_000.png"}
    assert picks == eda.pick_examples(labels, CLASSES, [NO_FINDING, "Effusion"], per_class=2, seed=0)


def test_run_eda_writes_figures_and_tables(data_dir, tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["paths"] = {"data_dir": str(data_dir), "results_dir": str(tmp_path / "results")}
    eda.run_eda(cfg)
    figures = tmp_path / "results" / "figures"
    for name in ("eda_prevalencia", "eda_coocorrencia", "eda_imagens_por_paciente",
                 "eda_idade_sexo", "eda_ap_pa", "eda_exemplos"):
        assert (figures / f"{name}.png").stat().st_size > 0
        assert (figures / f"{name}.pdf").stat().st_size > 0
    assert (tmp_path / "results" / "tables" / "eda_resumo.csv").exists()
    assert (tmp_path / "results" / "tables" / "eda_contagens_por_classe.csv").exists()
