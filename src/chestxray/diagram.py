"""Workflow diagram of the project (monograph, Figure 1).

    python -m chestxray.diagram            # results/figures/fluxo_trabalho.png and .pdf

Drawn with matplotlib, like every other figure, so it follows the same conventions (Portuguese,
PNG at 300 dpi and a vector PDF). The numbers in the boxes are the dataset's and the split's
(Sections 3.2 and 3.4 of the monograph).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from chestxray.plotting import FOCUS_COLOR, save_figure, setup_style
from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

TEXT_COLOR = "#1c2430"
ARROW_COLOR = "#5b6676"


@dataclass(frozen=True)
class Style:
    face: str
    edge: str
    text: str = TEXT_COLOR


DATA = Style("#eef1f4", "#9aa5b1")
MODEL = Style("#e8f0fa", FOCUS_COLOR)
TEST = Style("#fdf1e3", "#c77c1e")
OUTPUT = Style("#e9f5ec", "#3c8d5a")
FINAL = Style(FOCUS_COLOR, FOCUS_COLOR, "#ffffff")
NOTE = Style("#ffffff", "#9aa5b1")


FONT_SIZE = 8.6
LINE_SPACING = 1.35


def _box(ax, x: float, y: float, w: float, h: float, title: str, body: str = "", style: Style = DATA,
         dashed: bool = False) -> tuple[float, float, float, float]:
    """Rounded box with a bold title line and a regular body, centred as one block."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=style.face, edgecolor=style.edge, linewidth=1.3,
                                linestyle=(0, (4, 3)) if dashed else "solid"))
    line = FONT_SIZE * LINE_SPACING  # points
    n_body = body.count("\n") + 1 if body else 0
    block = (1 + n_body) * line
    centre = (x + w / 2, y + h / 2)
    kwargs = dict(xy=centre, textcoords="offset points", ha="center", va="center", fontsize=FONT_SIZE,
                  color=style.text, linespacing=LINE_SPACING)
    ax.annotate(title, xytext=(0, block / 2 - line / 2), fontweight="bold", **kwargs)
    if body:
        ax.annotate(body, xytext=(0, -line / 2), **kwargs)
    return x, y, w, h


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
                                 color=ARROW_COLOR, shrinkA=0, shrinkB=0))


def draw_workflow() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.6, 9.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 124)
    ax.axis("off")

    # data
    nih = _box(ax, 22, 112, 56, 10, "NIH ChestX-ray14",
               "112.120 radiografias frontais de 30.805 pacientes", DATA)
    pre = _box(ax, 22, 99, 56, 8.5, "Pré-processamento", "tons de cinza, 1024 → 256 pixels (Lanczos)", DATA)
    _arrow(ax, (50, nih[1]), (50, pre[1] + pre[3]))
    split = _box(ax, 12, 85.5, 76, 8.5, "Divisão por paciente (semente 42)",
                 "nenhum paciente aparece em mais de um conjunto", DATA)
    _arrow(ax, (50, pre[1]), (50, split[1] + split[3]))

    col_x, col_w = (3, 36, 69), 28
    sets = [("Treino", "78.486 imagens (70%)"), ("Validação", "16.812 imagens (15%)"),
            ("Teste", "16.822 imagens (15%)")]
    for x, (title, body) in zip(col_x, sets):
        _box(ax, x, 72, col_w, 8.5, title, body, DATA)
        _arrow(ax, (x + col_w / 2, split[1]), (x + col_w / 2, 72 + 8.5))

    # training, validation and test
    train = _box(ax, col_x[0], 40, col_w, 27, "Treino da DenseNet-121",
                 "pré-treinada no ImageNet\n\nE1: base (escolhido)\nE2: ponderação de classes\n"
                 "E3: sem aumento de dados\nE4: sem ImageNet\nE1 com sementes 43 e 44", MODEL)
    val = _box(ax, col_x[1], 40, col_w, 27, "Escolhas na validação",
               "a cada época: AUC média,\nparada antecipada\n\nno fim: escolha da\nconfiguração (E1),"
               "\nlimiares de Youden\ne Platt scaling", MODEL)
    test = _box(ax, col_x[2], 40, col_w, 27, "Avaliação no teste",
                "usado só depois da escolha\n\nAUC e AUPRC com IC95%\n(bootstrap por paciente),\n"
                "comparações pareadas,\ncalibração e subgrupos", TEST)
    for x in col_x:
        _arrow(ax, (x + col_w / 2, 72), (x + col_w / 2, 67))
    _arrow(ax, (train[0] + col_w, 53.5), (val[0], 53.5))
    _arrow(ax, (val[0] + col_w, 53.5), (test[0], 53.5))

    # final model and what is built on it
    final = _box(ax, 33, 25, 34, 9, "Modelo final (E1)", "AUC média de 0,841 no teste", FINAL)
    _arrow(ax, (val[0] + col_w / 2, val[1]), (50, final[1] + final[3]))
    _box(ax, 70, 25.5, 28, 8, "E5 (à parte)", "mesma configuração na\ndivisão oficial do NIH", NOTE,
         dashed=True)

    outputs = [(3, "Grad-CAM", "galeria de acertos e erros;\npointing game com as\ncaixas dos radiologistas"),
               (36, "Validação externa", "CheXpert: 202 imagens,\nrótulos de radiologistas,\nsem novo treino"),
               (69, "Interface web", "Hugging Face Spaces;\nmodelo roda no navegador\n(ONNX Runtime Web)")]
    for x, title, body in outputs:
        box = _box(ax, x, 3, col_w, 15, title, body, OUTPUT)
        _arrow(ax, (50, final[1]), (x + col_w / 2, box[1] + box[3]))
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/figures/fluxo_trabalho"))
    args = parser.parse_args(argv)
    setup_logging()
    setup_style()
    fig = draw_workflow()
    paths = save_figure(fig, args.out)
    plt.close(fig)
    logger.info("Wrote %s", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
