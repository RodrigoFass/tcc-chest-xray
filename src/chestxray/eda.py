"""Exploratory analysis of the NIH ChestX-ray14 labels, with figures for the monograph.

    python -m chestxray.eda
    python -m chestxray.eda --paths configs/paths/local_sample.yaml

Works on the whole label file (not the splits): class counts and prevalence,
co-occurrence of diseases, images per patient, age and sex, AP x PA per class and example
images. Figures go to ``<results_dir>/figures/eda_*.{png,pdf}``, tables to
``<results_dir>/tables/eda_*.csv``. The CSV has a few impossible ages (above 100 years):
they are left out of the age plot only, and counted in the summary table.
``notebooks/01_eda.ipynb`` calls these functions and discusses the results.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from PIL import Image

from chestxray.config import load_config
from chestxray.data.split import NO_FINDING, find_labels_file, load_labels
from chestxray.plotting import (
    FOCUS_COLOR,
    NEUTRAL_COLOR,
    OTHER_COLOR,
    br_number,
    pt,
    save_figure,
    setup_style,
    use_decimal_comma,
)
from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

MAX_PLAUSIBLE_AGE = 100
SEX_PT = {"M": "Masculino", "F": "Feminino"}


def no_finding_mask(df: pd.DataFrame, classes: list[str]) -> pd.Series:
    return df[classes].sum(axis=1) == 0


def class_counts(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Per class (and "No Finding"): number of images, prevalence (%) and share of AP exams (%)."""
    masks = {c: df[c] == 1 for c in classes}
    masks[NO_FINDING] = no_finding_mask(df, classes)
    rows = [{
        "classe": name,
        "classe_pt": pt(name),
        "n": int(mask.sum()),
        "pct": 100 * mask.mean(),
        "pct_ap": 100 * (df.loc[mask, "view"] == "AP").mean() if mask.any() else np.nan,
    } for name, mask in masks.items()]
    return pd.DataFrame(rows)


def cooccurrence(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Images having both classes; the diagonal holds each class's total."""
    x = df[classes].to_numpy(dtype=np.int64)
    return pd.DataFrame(x.T @ x, index=classes, columns=classes)


def summary_table(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    per_patient = df["patient_id"].value_counts()
    n_diseases = df[classes].sum(axis=1)
    ages = df.loc[df["age"] <= MAX_PLAUSIBLE_AGE, "age"]
    rows = [
        ("Imagens", len(df)),
        ("Pacientes", df["patient_id"].nunique()),
        ("Imagens sem achados (%)", 100 * (n_diseases == 0).mean()),
        ("Imagens com 2 ou mais doenças (%)", 100 * (n_diseases >= 2).mean()),
        ("Imagens por paciente: mediana", per_patient.median()),
        ("Imagens por paciente: máximo", per_patient.max()),
        ("Pacientes com 1 imagem (%)", 100 * (per_patient == 1).mean()),
        ("Idade: mediana (anos)", ages.median()),
        ("Idade: 1º quartil (anos)", ages.quantile(0.25)),
        ("Idade: 3º quartil (anos)", ages.quantile(0.75)),
        (f"Idades acima de {MAX_PLAUSIBLE_AGE} anos (excluídas do gráfico)", int((df["age"] > MAX_PLAUSIBLE_AGE).sum())),
        ("Imagens de pacientes do sexo feminino (%)", 100 * (df["sex"] == "F").mean()),
        ("Exames AP (%)", 100 * (df["view"] == "AP").mean()),
    ]
    return pd.DataFrame(rows, columns=["metrica", "valor"])


def _bar_colors(names: list[str], focus: list[str]) -> list[str]:
    return [FOCUS_COLOR if n in focus else NEUTRAL_COLOR if n == NO_FINDING else OTHER_COLOR for n in names]


def plot_class_counts(counts: pd.DataFrame, focus: list[str]) -> plt.Figure:
    data = counts.sort_values("n")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data["classe_pt"], data["n"], color=_bar_colors(data["classe"].tolist(), focus))
    for y, (n, pct) in enumerate(zip(data["n"], data["pct"])):
        ax.text(n, y, f"  {br_number(n)} ({br_number(pct, 1)}%)", va="center", fontsize=9)
    ax.set_xlim(0, data["n"].max() * 1.3)
    ax.set_xlabel("Número de imagens")
    ax.set_title("Imagens por classe (em destaque, as três doenças do TCC)")
    use_decimal_comma(ax.xaxis)
    return fig


def plot_cooccurrence(matrix: pd.DataFrame) -> plt.Figure:
    labels = [pt(c) for c in matrix.index]
    annot = matrix.map(lambda v: br_number(v) if v else "")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix.replace(0, np.nan), norm=LogNorm(), annot=annot, fmt="", annot_kws={"fontsize": 7},
                cmap="Blues", xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Número de imagens (escala log)"})
    ax.set_title("Co-ocorrência de doenças (diagonal: total de cada classe; vazio: nenhuma)")
    ax.grid(False)
    use_decimal_comma(ax.collections[0].colorbar.ax.yaxis)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    return fig


def plot_images_per_patient(df: pd.DataFrame) -> plt.Figure:
    per_patient = df["patient_id"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(per_patient, bins=np.arange(1, per_patient.max() + 2) - 0.5, color=OTHER_COLOR)
    ax.set_yscale("log")
    ax.set_xlabel("Imagens por paciente")
    ax.set_ylabel("Número de pacientes (escala log)")
    ax.set_title("Distribuição do número de imagens por paciente")
    text = (f"Mediana: {br_number(per_patient.median(), 0)}\nMáximo: {br_number(per_patient.max())}\n"
            f"Com 1 imagem: {br_number(100 * (per_patient == 1).mean(), 1)}% dos pacientes")
    ax.text(0.97, 0.95, text, transform=ax.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "edgecolor": "#cccccc"})
    use_decimal_comma(ax.xaxis, ax.yaxis)
    return fig


def plot_age_sex(df: pd.DataFrame) -> plt.Figure:
    valid = df[df["age"] <= MAX_PLAUSIBLE_AGE].assign(sexo=lambda d: d["sex"].map(SEX_PT))
    excluded = len(df) - len(valid)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(valid, x="age", hue="sexo", bins=np.arange(0, MAX_PLAUSIBLE_AGE + 5, 5),
                 multiple="dodge", shrink=0.85, ax=ax, palette=[FOCUS_COLOR, "#d98c3f"],
                 hue_order=["Masculino", "Feminino"])
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel("Número de imagens")
    excluded_text = (f"{br_number(excluded)} idade acima de {MAX_PLAUSIBLE_AGE} anos excluída" if excluded == 1
                     else f"{br_number(excluded)} idades acima de {MAX_PLAUSIBLE_AGE} anos excluídas")
    ax.set_title(f"Idade e sexo por imagem ({excluded_text})")
    ax.get_legend().set_title("Sexo")
    use_decimal_comma(ax.xaxis, ax.yaxis)
    return fig


def plot_view_by_class(counts: pd.DataFrame, overall_ap: float, focus: list[str]) -> plt.Figure:
    data = counts.sort_values("pct_ap")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data["classe_pt"], data["pct_ap"], color=_bar_colors(data["classe"].tolist(), focus))
    ax.axvline(overall_ap, color="black", linestyle="--", linewidth=1)
    ax.text(overall_ap, len(data) - 0.4, f" geral: {br_number(overall_ap, 1)}%", fontsize=9)
    ax.set_xlabel("Exames na incidência AP (%)")
    ax.set_title("Proporção de exames AP (em vez de PA) por classe")
    use_decimal_comma(ax.xaxis)
    return fig


def pick_examples(df: pd.DataFrame, classes: list[str], names: list[str], per_class: int, seed: int) -> dict:
    """Images with exactly that one label (or none, for "No Finding"), drawn with a fixed seed."""
    n_labels = df[classes].sum(axis=1)
    rng = np.random.default_rng(seed)
    picks = {}
    for name in names:
        mask = n_labels == 0 if name == NO_FINDING else (df[name] == 1) & (n_labels == 1)
        candidates = sorted(df.loc[mask, "image"])
        picks[name] = list(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False))
    return picks


def plot_examples(picks: dict, images_dir: Path) -> plt.Figure:
    per_class = max(len(v) for v in picks.values())
    fig, axes = plt.subplots(per_class, len(picks), figsize=(3 * len(picks), 3 * per_class), squeeze=False)
    for col, (name, images) in enumerate(picks.items()):
        for row in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if row < len(images):
                with Image.open(images_dir / images[row]) as img:
                    ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{pt(name)}\n{images[row]}" if row == 0 else images[row], fontsize=9)
    fig.suptitle("Exemplos de imagens com um único rótulo (256×256, sorteio com seed fixa)")
    fig.tight_layout()
    return fig


def run_eda(cfg: dict) -> pd.DataFrame:
    """Generate every EDA figure and table; return the summary table."""
    setup_style()
    paths, classes = cfg["paths"], cfg["data"]["classes"]
    focus = cfg["data"]["focus_classes"]
    data_dir = Path(paths["data_dir"])
    figures = Path(paths["results_dir"]) / "figures"
    tables = Path(paths["results_dir"]) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df = load_labels(find_labels_file(data_dir), classes)
    counts = class_counts(df, classes)
    summary = summary_table(df, classes)
    counts.to_csv(tables / "eda_contagens_por_classe.csv", index=False)
    summary.to_csv(tables / "eda_resumo.csv", index=False)

    picks = pick_examples(df, classes, [NO_FINDING, *focus], per_class=2, seed=cfg["seed"])
    overall_ap = 100 * (df["view"] == "AP").mean()
    figs = {
        "eda_prevalencia": plot_class_counts(counts, focus),
        "eda_coocorrencia": plot_cooccurrence(cooccurrence(df, classes)),
        "eda_imagens_por_paciente": plot_images_per_patient(df),
        "eda_idade_sexo": plot_age_sex(df),
        "eda_ap_pa": plot_view_by_class(counts, overall_ap, focus),
        "eda_exemplos": plot_examples(picks, data_dir / "images"),
    }
    for name, fig in figs.items():
        save_figure(fig, figures / name)
        plt.close(fig)
    logger.info("Wrote %d figures to %s and tables to %s", len(figs), figures, tables)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    summary = run_eda(load_config(args.config, args.paths))
    logger.info("Summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
