"""Figure conventions for the monograph: Portuguese names, decimal comma, PNG 300 dpi + PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axis import Axis
from matplotlib.ticker import FuncFormatter

CLASS_NAMES_PT = {
    "Atelectasis": "Atelectasia",
    "Cardiomegaly": "Cardiomegalia",
    "Effusion": "Efusão pleural",
    "Infiltration": "Infiltração",
    "Mass": "Massa",
    "Nodule": "Nódulo",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumotórax",
    "Consolidation": "Consolidação",
    "Edema": "Edema",
    "Emphysema": "Enfisema",
    "Fibrosis": "Fibrose",
    "Pleural_Thickening": "Espessamento pleural",
    "Hernia": "Hérnia",
    "No Finding": "Sem achados",
}

FOCUS_COLOR = "#1f5fa8"   # the 3 TCC classes
OTHER_COLOR = "#9aa5b1"   # the other classes
NEUTRAL_COLOR = "#d0d5db"  # "No Finding"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight"})


def pt(name: str) -> str:
    """Portuguese display name of a class (unchanged if unknown)."""
    return CLASS_NAMES_PT.get(name, name)


def br_number(value: float, decimals: int = 0) -> str:
    """Brazilian format: 11559.5 -> '11.559,5'."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def _tick_label(value: float, _pos=None) -> str:
    text = f"{value:.6g}"
    decimals = len(text.split(".")[1]) if "." in text and "e" not in text else 0
    return br_number(value, decimals)


def use_decimal_comma(*axes: Axis) -> None:
    """Tick labels with decimal comma and dot as thousands separator."""
    for axis in axes:
        axis.set_major_formatter(FuncFormatter(_tick_label))


def save_figure(fig: plt.Figure, stem: str | Path) -> list[Path]:
    """Save ``<stem>.png`` (300 dpi) and ``<stem>.pdf`` (vector) and return both paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=300)
    fig.savefig(outputs[1])
    return outputs
