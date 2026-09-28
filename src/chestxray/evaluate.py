"""Evaluate trained experiments from their prediction files (CPU only, no images needed).

    python -m chestxray.evaluate --run results/runs/e1_baseline --split val
    python -m chestxray.evaluate --run results/runs/e1_baseline --split test
    python -m chestxray.evaluate --compare results/runs/e1_baseline results/runs/e2_posweight \
        --pairs e2_posweight:e1_baseline

For one run and split: AUC-ROC per class and the mean over the 14 classes, with 95% CIs from
a patient bootstrap (plan 3.9); AUPRC next to the prevalence; ROC and precision-recall
curves. On the test split, whatever must be fitted is fitted on the validation predictions
and only applied to test: the Youden threshold of each class (accuracy, sensitivity,
specificity, precision, F1 and confusion matrices on test) and the Platt recalibration
(``calibration.json``, reliability diagrams and Brier score before and after). The test
evaluation also covers subgroups (sex, age, view) and the comparison with the literature.

Outputs: ``<run>/metrics_<split>.json``, tables (CSV, Markdown and LaTeX) in
``results/tables/<experiment>/`` and figures (PNG and PDF) in ``results/figures/<experiment>/``.
``--compare`` writes the side-by-side table of experiments and the paired differences
(both models scored on the same bootstrap samples) to ``results/tables/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from chestxray.metrics import (
    apply_platt,
    brier_score,
    fit_platt,
    mean_auc,
    patient_bootstrap,
    per_class_auc,
    per_class_average_precision,
    percentile_ci,
    threshold_stats,
    youden_threshold,
)
from chestxray.plotting import br_number, pt, save_figure, save_table, setup_style, use_decimal_comma
from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

SPLIT_PT = {"val": "validação", "test": "teste"}
FOCUS_COLORS = ["#1f5fa8", "#d98c3f", "#3a9a5b"]

# AUC per class in Table 2 of CheXNet (arXiv:1711.05225v3), which also reports Wang et al.
# (2017); checked on 27/09/2026 (plan, Phase 3). Different splits: an approximate comparison.
LITERATURE = {
    "Atelectasis": (0.716, 0.8094), "Cardiomegaly": (0.807, 0.9248), "Effusion": (0.784, 0.8638),
    "Infiltration": (0.609, 0.7345), "Mass": (0.706, 0.8676), "Nodule": (0.671, 0.7802),
    "Pneumonia": (0.633, 0.7680), "Pneumothorax": (0.806, 0.8887), "Consolidation": (0.708, 0.7901),
    "Edema": (0.835, 0.8878), "Emphysema": (0.815, 0.9371), "Fibrosis": (0.769, 0.8047),
    "Pleural_Thickening": (0.708, 0.8062), "Hernia": (0.767, 0.9164),
}

MAX_PLAUSIBLE_AGE = 100
SUBGROUPS = {
    "Sexo": {"Masculino": lambda d: d["sex"] == "M", "Feminino": lambda d: d["sex"] == "F"},
    "Idade": {
        "< 40 anos": lambda d: d["age"] < 40,
        "40–60 anos": lambda d: (d["age"] >= 40) & (d["age"] <= 60),
        "> 60 anos": lambda d: (d["age"] > 60) & (d["age"] <= MAX_PLAUSIBLE_AGE),
    },
    "Incidência": {"PA": lambda d: d["view"] == "PA", "AP": lambda d: d["view"] == "AP"},
}


def fmt(value: float, decimals: int = 3) -> str:
    return "—" if value is None or np.isnan(value) else br_number(value, decimals)


def fmt_ci(value: float, ci: list[float], decimals: int = 3) -> str:
    """'0,820 (0,810–0,830)'"""
    return f"{fmt(value, decimals)} ({fmt(ci[0], decimals)}–{fmt(ci[1], decimals)})"


class Predictions:
    """One run's predictions on one split, as arrays aligned with the class list."""

    def __init__(self, path: Path, classes: list[str]):
        self.table = pd.read_csv(path)
        self.classes = classes
        self.y = self.table[classes].to_numpy(dtype=float)
        self.logits = self.table[[f"logit_{c}" for c in classes]].to_numpy(dtype=float)
        self.scores = self.table[[f"score_{c}" for c in classes]].to_numpy(dtype=float)
        self.patients = self.table["patient_id"].to_numpy()

    def subset(self, mask: np.ndarray) -> "Predictions":
        sub = object.__new__(Predictions)
        sub.table, sub.classes = self.table[mask].reset_index(drop=True), self.classes
        sub.y, sub.logits, sub.scores = self.y[mask], self.logits[mask], self.scores[mask]
        sub.patients = self.patients[mask]
        return sub


def bootstrap_auc_ap(preds: Predictions, n_boot: int, seed: int) -> dict:
    """Point estimates and patient-bootstrap CIs of AUC (per class and mean) and AUPRC."""
    aucs, aps, means = [], [], []
    for idx in patient_bootstrap(preds.patients, n_boot, seed):
        auc = per_class_auc(preds.y[idx], preds.scores[idx])
        aucs.append(auc)
        means.append(mean_auc(auc))
        aps.append(per_class_average_precision(preds.y[idx], preds.scores[idx]))
    auc_lo, auc_hi = percentile_ci(np.array(aucs))
    ap_lo, ap_hi = percentile_ci(np.array(aps))
    mean_lo, mean_hi = percentile_ci(np.array(means))
    auc = per_class_auc(preds.y, preds.scores)
    ap = per_class_average_precision(preds.y, preds.scores)
    return {
        "mean_auc": {"value": mean_auc(auc), "ci": [float(mean_lo), float(mean_hi)]},
        "classes": {
            c: {
                "n_positive": int(preds.y[:, k].sum()),
                "prevalence": float(preds.y[:, k].mean()),
                "auc": float(auc[k]), "auc_ci": [float(auc_lo[k]), float(auc_hi[k])],
                "auprc": float(ap[k]), "auprc_ci": [float(ap_lo[k]), float(ap_hi[k])],
            }
            for k, c in enumerate(preds.classes)
        },
    }


def thresholds_and_calibration(val: Predictions, test: Predictions) -> dict:
    """Per class, fitted on validation only: Youden threshold and Platt parameters; with the
    threshold statistics and the Brier scores (raw and recalibrated) on test."""
    result = {}
    for k, c in enumerate(val.classes):
        threshold = youden_threshold(val.y[:, k], val.scores[:, k])
        a, b = fit_platt(val.logits[:, k], val.y[:, k])
        logit_threshold = np.log(threshold / (1 - threshold)) if 0 < threshold < 1 else np.nan
        calibrated = apply_platt(test.logits[:, k], a, b)
        result[c] = {
            "threshold": threshold,
            "threshold_calibrated": float(apply_platt(logit_threshold, a, b)) if a > 0 else None,
            "platt_a": a, "platt_b": b,
            "test_at_threshold": threshold_stats(test.y[:, k], test.scores[:, k], threshold),
            "brier_raw": brier_score(test.y[:, k], test.scores[:, k]),
            "brier_calibrated": brier_score(test.y[:, k], calibrated),
        }
    return result


def subgroup_aucs(test: Predictions, focus: list[str], n_boot: int, seed: int) -> list[dict]:
    """AUC with patient-bootstrap CI of each focus class inside each subgroup."""
    rows = []
    focus_idx = [test.classes.index(c) for c in focus]
    for family, groups in SUBGROUPS.items():
        for name, select in groups.items():
            sub = test.subset(select(test.table).to_numpy())
            aucs = np.array([per_class_auc(sub.y[idx][:, focus_idx], sub.scores[idx][:, focus_idx])
                             for idx in patient_bootstrap(sub.patients, n_boot, seed)])
            lo, hi = percentile_ci(aucs)
            point = per_class_auc(sub.y[:, focus_idx], sub.scores[:, focus_idx])
            for j, c in enumerate(focus):
                rows.append({
                    "classe": c, "grupo": family, "subgrupo": name, "n_imagens": len(sub.y),
                    "n_positivos": int(sub.y[:, focus_idx[j]].sum()),
                    "auc": float(point[j]), "auc_ci": [float(lo[j]), float(hi[j])],
                })
    return rows


# ----------------------------------------------------------------------------- figures

def plot_roc_focus(preds: Predictions, focus: list[str], metrics: dict, split: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for color, c in zip(FOCUS_COLORS, focus):
        k = preds.classes.index(c)
        fpr, tpr, _ = roc_curve(preds.y[:, k], preds.scores[:, k])
        m = metrics["classes"][c]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{pt(c)}: AUC {fmt_ci(m['auc'], m['auc_ci'])}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("1 − especificidade (taxa de falsos positivos)")
    ax.set_ylabel("Sensibilidade")
    ax.set_title(f"Curvas ROC no conjunto de {SPLIT_PT[split]}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    use_decimal_comma(ax.xaxis, ax.yaxis)
    return fig


def plot_roc_grid(preds: Predictions, metrics: dict, split: str) -> plt.Figure:
    fig, axes = plt.subplots(4, 4, figsize=(13, 13), sharex=True, sharey=True)
    for ax in axes.flat[len(preds.classes):]:
        ax.axis("off")
    for k, (ax, c) in enumerate(zip(axes.flat, preds.classes)):
        fpr, tpr, _ = roc_curve(preds.y[:, k], preds.scores[:, k])
        ax.plot(fpr, tpr, color=FOCUS_COLORS[0], lw=1.5)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
        m = metrics["classes"][c]
        ax.set_title(f"{pt(c)}\nAUC {fmt_ci(m['auc'], m['auc_ci'])}", fontsize=9)
        use_decimal_comma(ax.xaxis, ax.yaxis)
    fig.supxlabel("1 − especificidade")
    fig.supylabel("Sensibilidade")
    fig.suptitle(f"Curvas ROC das 14 classes ({SPLIT_PT[split]})")
    fig.tight_layout()
    return fig


def plot_pr_focus(preds: Predictions, focus: list[str], metrics: dict, split: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for color, c in zip(FOCUS_COLORS, focus):
        k = preds.classes.index(c)
        precision, recall, _ = precision_recall_curve(preds.y[:, k], preds.scores[:, k])
        m = metrics["classes"][c]
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{pt(c)}: AUPRC {fmt_ci(m['auprc'], m['auprc_ci'])}")
        ax.axhline(m["prevalence"], color=color, ls=":", lw=1)
    ax.set_xlabel("Sensibilidade (revocação)")
    ax.set_ylabel("Precisão (valor preditivo positivo)")
    ax.set_title(f"Curvas precisão-revocação ({SPLIT_PT[split]}); pontilhado: prevalência")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    use_decimal_comma(ax.xaxis, ax.yaxis)
    return fig


def plot_confusion_focus(focus: list[str], thresholds: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, len(focus), figsize=(4.2 * len(focus), 4))
    for ax, c in zip(np.atleast_1d(axes), focus):
        s = thresholds[c]["test_at_threshold"]
        matrix = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        annot = [[f"{br_number(matrix[i, j])}\n({br_number(100 * shares[i, j], 1)}%)" for j in range(2)]
                 for i in range(2)]
        sns.heatmap(shares, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=["Negativo", "Positivo"], yticklabels=["Ausente", "Presente"])
        ax.set_xlabel("Predição do modelo")
        ax.set_ylabel("Rótulo")
        ax.set_title(f"{pt(c)} (limiar {fmt(thresholds[c]['threshold'])})")
    fig.suptitle("Matrizes de confusão no teste, no limiar de Youden da validação")
    fig.tight_layout()
    return fig


def plot_calibration_focus(test: Predictions, focus: list[str], thresholds: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, len(focus), figsize=(4.5 * len(focus), 4.6))
    for ax, color, c in zip(np.atleast_1d(axes), FOCUS_COLORS, focus):
        k = test.classes.index(c)
        t = thresholds[c]
        calibrated = apply_platt(test.logits[:, k], t["platt_a"], t["platt_b"])
        for probs, style, label in ((test.scores[:, k], "--", f"escore bruto (Brier {fmt(t['brier_raw'], 4)})"),
                                    (calibrated, "-", f"após Platt (Brier {fmt(t['brier_calibrated'], 4)})")):
            frac, mean_pred = calibration_curve(test.y[:, k], probs, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac, style, marker="o", ms=4, color=color, label=label)
        top = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, top], [0, top], ":", color="grey", lw=1)
        ax.set_xlabel("Escore médio previsto")
        ax.set_ylabel("Fração observada de positivos")
        ax.set_title(pt(c))
        ax.legend(fontsize=8, loc="upper left")
        use_decimal_comma(ax.xaxis, ax.yaxis)
    fig.suptitle("Calibração no teste (10 faixas com o mesmo número de imagens)")
    fig.tight_layout()
    return fig


def plot_subgroups(rows: list[dict], focus: list[str], metrics: dict) -> plt.Figure:
    # No shared y axis: each panel labels its subgroups with that class's own case counts
    fig, axes = plt.subplots(1, len(focus), figsize=(5.2 * len(focus), 4.8))
    for ax, color, c in zip(np.atleast_1d(axes), FOCUS_COLORS, focus):
        sel = [r for r in rows if r["classe"] == c]
        y = np.arange(len(sel))[::-1]
        points = [r["auc"] for r in sel]
        errors = [[r["auc"] - r["auc_ci"][0] for r in sel], [r["auc_ci"][1] - r["auc"] for r in sel]]
        ax.errorbar(points, y, xerr=errors, fmt="o", color=color, capsize=3)
        ax.axvline(metrics["classes"][c]["auc"], color="grey", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['subgrupo']} (n={br_number(r['n_positivos'])})" for r in sel])
        ax.set_title(pt(c))
        ax.set_xlabel("AUC (IC95%)")
        use_decimal_comma(ax.xaxis)
    fig.suptitle("AUC por subgrupo no teste (tracejado: AUC geral; n = casos positivos)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- tables

def class_metrics_table(metrics: dict, focus: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, display = [], []
    for c, m in metrics["classes"].items():
        rows.append({"classe": c, "n_positivos": m["n_positive"], "prevalencia": m["prevalence"],
                     "auc": m["auc"], "auc_ic_inf": m["auc_ci"][0], "auc_ic_sup": m["auc_ci"][1],
                     "auprc": m["auprc"], "auprc_ic_inf": m["auprc_ci"][0], "auprc_ic_sup": m["auprc_ci"][1]})
        display.append({"Doença": pt(c) + (" *" if c in focus else ""),
                        "Casos": br_number(m["n_positive"]),
                        "Prevalência": f"{br_number(100 * m['prevalence'], 1)}%",
                        "AUC (IC95%)": fmt_ci(m["auc"], m["auc_ci"]),
                        "AUPRC (IC95%)": fmt_ci(m["auprc"], m["auprc_ci"])})
    mean = metrics["mean_auc"]
    rows.append({"classe": "Média (14 classes)", "auc": mean["value"],
                 "auc_ic_inf": mean["ci"][0], "auc_ic_sup": mean["ci"][1]})
    display.append({"Doença": "Média (14 classes)", "Casos": "", "Prevalência": "",
                    "AUC (IC95%)": fmt_ci(mean["value"], mean["ci"]), "AUPRC (IC95%)": ""})
    return pd.DataFrame(rows), pd.DataFrame(display)


def threshold_table(thresholds: dict, focus: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, display = [], []
    for c, t in thresholds.items():
        s = t["test_at_threshold"]
        rows.append({"classe": c, "limiar": t["threshold"], **{k: s[k] for k in (
            "accuracy", "sensitivity", "specificity", "precision", "npv", "f1", "tp", "fp", "fn", "tn")}})
        display.append({"Doença": pt(c) + (" *" if c in focus else ""), "Limiar": fmt(t["threshold"]),
                        "Acurácia": fmt(s["accuracy"]), "Sensibilidade": fmt(s["sensitivity"]),
                        "Especificidade": fmt(s["specificity"]), "Precisão (VPP)": fmt(s["precision"]),
                        "F1": fmt(s["f1"])})
    return pd.DataFrame(rows), pd.DataFrame(display)


def calibration_table(thresholds: dict, focus: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"classe": c, "platt_a": t["platt_a"], "platt_b": t["platt_b"],
             "brier_bruto": t["brier_raw"], "brier_calibrado": t["brier_calibrated"]} for c, t in thresholds.items()]
    display = [{"Doença": pt(r["classe"]) + (" *" if r["classe"] in focus else ""),
                "Brier (escore bruto)": fmt(r["brier_bruto"], 4), "Brier (após Platt)": fmt(r["brier_calibrado"], 4)}
               for r in rows]
    return pd.DataFrame(rows), pd.DataFrame(display)


def subgroup_table(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = pd.DataFrame([{**{k: v for k, v in r.items() if k != "auc_ci"},
                             "auc_ic_inf": r["auc_ci"][0], "auc_ic_sup": r["auc_ci"][1]} for r in rows])
    display = pd.DataFrame([{"Doença": pt(r["classe"]), "Subgrupo": r["subgrupo"],
                             "Imagens": br_number(r["n_imagens"]), "Casos": br_number(r["n_positivos"]),
                             "AUC (IC95%)": fmt_ci(r["auc"], r["auc_ci"])} for r in rows])
    return numeric, display


def literature_table(metrics: dict, focus: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, display = [], []
    for c, m in metrics["classes"].items():
        wang, chexnet = LITERATURE[c]
        rows.append({"classe": c, "wang_2017": wang, "chexnet_2017": chexnet, "este_trabalho": m["auc"],
                     "ic_inf": m["auc_ci"][0], "ic_sup": m["auc_ci"][1]})
        display.append({"Doença": pt(c) + (" *" if c in focus else ""), "Wang et al. (2017)": fmt(wang),
                        "CheXNet (2017)": fmt(chexnet, 4), "Este trabalho (IC95%)": fmt_ci(m["auc"], m["auc_ci"])})
    means = [np.mean([v[i] for v in LITERATURE.values()]) for i in (0, 1)]
    display.append({"Doença": "Média (14 classes)", "Wang et al. (2017)": fmt(means[0]),
                    "CheXNet (2017)": fmt(means[1], 4),
                    "Este trabalho (IC95%)": fmt_ci(metrics["mean_auc"]["value"], metrics["mean_auc"]["ci"])})
    return pd.DataFrame(rows), pd.DataFrame(display)


# ----------------------------------------------------------------------------- per run

def evaluate_run(run_dir: Path, split: str, results_dir: Path, n_boot: int | None = None) -> dict:
    """Evaluate one run on one split and write its JSON, tables and figures."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    name, classes, focus = cfg["experiment"], cfg["data"]["classes"], cfg["data"]["focus_classes"]
    n_boot = n_boot or cfg["eval"]["bootstrap_samples"]
    seed = cfg["eval"]["bootstrap_seed"]
    preds = Predictions(run_dir / f"preds_{split}.csv", classes)
    logger.info("%s / %s: %d images, %d patients, %d bootstrap samples", name, split, len(preds.y),
                len(np.unique(preds.patients)), n_boot)

    metrics = {"experiment": name, "split": split, "n_images": len(preds.y),
               "n_patients": int(len(np.unique(preds.patients))),
               "bootstrap": {"samples": n_boot, "seed": seed, "unit": "patient", "level": 0.95},
               **bootstrap_auc_ap(preds, n_boot, seed)}
    tables_dir, figures_dir = results_dir / "tables" / name, results_dir / "figures" / name
    setup_style()
    save_table(*class_metrics_table(metrics, focus), tables_dir / f"metricas_{split}")
    figures = {f"roc_foco_{split}": plot_roc_focus(preds, focus, metrics, split),
               f"roc_14_{split}": plot_roc_grid(preds, metrics, split),
               f"pr_foco_{split}": plot_pr_focus(preds, focus, metrics, split)}

    if split == "test":
        val = Predictions(run_dir / "preds_val.csv", classes)
        thresholds = thresholds_and_calibration(val, preds)
        calibration = {"fitted_on": "val", "n_val_images": len(val.y),
                       "classes": {c: {k: t[k] for k in ("threshold", "threshold_calibrated", "platt_a", "platt_b")}
                                   for c, t in thresholds.items()}}
        (run_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        subgroups = subgroup_aucs(preds, focus, n_boot, seed)
        metrics["thresholds_from_val"] = thresholds
        metrics["subgroups"] = subgroups
        save_table(*threshold_table(thresholds, focus), tables_dir / "limiares_teste")
        save_table(*calibration_table(thresholds, focus), tables_dir / "calibracao_teste")
        save_table(*subgroup_table(subgroups), tables_dir / "subgrupos_teste")
        save_table(*literature_table(metrics, focus), tables_dir / "comparacao_literatura")
        figures.update({"matriz_confusao_foco_test": plot_confusion_focus(focus, thresholds),
                        "calibracao_foco_test": plot_calibration_focus(preds, focus, thresholds),
                        "subgrupos_foco_test": plot_subgroups(subgroups, focus, metrics)})

    for fig_name, fig in figures.items():
        save_figure(fig, figures_dir / fig_name)
        plt.close(fig)
    (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    focus_text = ", ".join(f"{c} {metrics['classes'][c]['auc']:.3f}" for c in focus)
    logger.info("%s / %s: mean AUC %.4f (95%% CI %.4f-%.4f); %s", name, split, metrics["mean_auc"]["value"],
                *metrics["mean_auc"]["ci"], focus_text)
    return metrics


# ----------------------------------------------------------------------------- comparisons

def compare_runs(run_dirs: list[Path], pairs: list[tuple[str, str]], results_dir: Path, split: str = "test",
                 n_boot: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Side-by-side table of experiments and paired bootstrap differences of AUC."""
    runs = {}
    for run_dir in run_dirs:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        runs[cfg["experiment"]] = {
            "cfg": cfg,
            "metrics": json.loads((run_dir / f"metrics_{split}.json").read_text(encoding="utf-8")),
            "summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
            "preds": Predictions(run_dir / f"preds_{split}.csv", cfg["data"]["classes"]),
        }
    first = next(iter(runs.values()))
    classes, focus = first["cfg"]["data"]["classes"], first["cfg"]["data"]["focus_classes"]
    images = first["preds"].table["image"]
    if any(not r["preds"].table["image"].equals(images) for r in runs.values()):
        raise SystemExit("The runs were evaluated on different images (e.g. E5's official split); "
                         "compare only runs that share the same split")
    n_boot = n_boot or first["cfg"]["eval"]["bootstrap_samples"]
    seed = first["cfg"]["eval"]["bootstrap_seed"]

    rows, display = [], []
    for name, r in runs.items():
        m, s = r["metrics"], r["summary"]
        row = {"experimento": name, "auc_media": m["mean_auc"]["value"],
               "auc_media_ic_inf": m["mean_auc"]["ci"][0], "auc_media_ic_sup": m["mean_auc"]["ci"][1],
               "melhor_epoca": s["best_epoch"], "epocas": s["epochs"], "tempo_treino_min": s["train_time_min"]}
        shown = {"Experimento": name, "AUC média (IC95%)": fmt_ci(m["mean_auc"]["value"], m["mean_auc"]["ci"])}
        for c in focus:
            cm = m["classes"][c]
            row.update({f"auc_{c}": cm["auc"], f"auc_{c}_ic_inf": cm["auc_ci"][0], f"auc_{c}_ic_sup": cm["auc_ci"][1],
                        f"auprc_{c}": cm["auprc"], f"auprc_{c}_ic_inf": cm["auprc_ci"][0],
                        f"auprc_{c}_ic_sup": cm["auprc_ci"][1]})
            shown[f"{pt(c)}: AUC"] = fmt_ci(cm["auc"], cm["auc_ci"])
            shown[f"{pt(c)}: AUPRC"] = fmt_ci(cm["auprc"], cm["auprc_ci"])
        shown["Épocas (melhor / total)"] = f"{s['best_epoch']} / {s['epochs']}"
        shown["Treino (min)"] = br_number(s["train_time_min"], 1)
        rows.append(row)
        display.append(shown)
    experiments = (pd.DataFrame(rows), pd.DataFrame(display))
    save_table(*experiments, results_dir / "tables" / f"experimentos_{split}")

    diff_rows, diff_display = [], []
    for a, b in pairs:
        pa, pb = runs[a]["preds"], runs[b]["preds"]
        samples = []
        for idx in patient_bootstrap(pa.patients, n_boot, seed):
            auc_a, auc_b = per_class_auc(pa.y[idx], pa.scores[idx]), per_class_auc(pb.y[idx], pb.scores[idx])
            samples.append(np.r_[mean_auc(auc_a) - mean_auc(auc_b), auc_a - auc_b])
        lo, hi = percentile_ci(np.array(samples))
        auc_a, auc_b = per_class_auc(pa.y, pa.scores), per_class_auc(pb.y, pb.scores)
        point = np.r_[mean_auc(auc_a) - mean_auc(auc_b), auc_a - auc_b]
        for j, label in enumerate(["Média (14 classes)", *classes]):
            if j and classes[j - 1] not in focus:
                continue
            significant = not (lo[j] <= 0 <= hi[j])
            diff_rows.append({"comparacao": f"{a} - {b}", "classe": "media" if j == 0 else label,
                              "diferenca_auc": point[j], "ic_inf": lo[j], "ic_sup": hi[j], "significativa": significant})
            diff_display.append({"Comparação": f"{a} − {b}", "Classe": pt(label) if j else label,
                                 "Diferença de AUC (IC95%)": fmt_ci(point[j], [lo[j], hi[j]]),
                                 "Significativa?": "sim" if significant else "não"})
    differences = (pd.DataFrame(diff_rows), pd.DataFrame(diff_display))
    if pairs:
        save_table(*differences, results_dir / "tables" / f"diferencas_pareadas_{split}")
    return experiments[1], differences[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path, help="run folder (results/runs/<experiment>)")
    mode.add_argument("--compare", type=Path, nargs="+", help="run folders to put side by side")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--pairs", nargs="*", default=[], help="paired differences as A:B (A minus B)")
    parser.add_argument("--results-dir", type=Path, help="default: the folder above results/runs")
    parser.add_argument("--bootstrap", type=int, help="bootstrap samples (default: eval.bootstrap_samples)")
    args = parser.parse_args(argv)
    setup_logging()

    first_run = args.run or args.compare[0]
    results_dir = args.results_dir or first_run.resolve().parent.parent
    if args.run:
        evaluate_run(args.run, args.split, results_dir, args.bootstrap)
    else:
        pairs = [tuple(p.split(":")) for p in args.pairs]
        table, diffs = compare_runs(args.compare, pairs, results_dir, args.split, args.bootstrap)
        logger.info("Experiments (%s):\n%s", args.split, table.to_string(index=False))
        if len(diffs):
            logger.info("Paired differences (%s):\n%s", args.split, diffs.to_string(index=False))


if __name__ == "__main__":
    main()
