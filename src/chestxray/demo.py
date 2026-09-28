"""Demo system (Phase 5): the logic behind ``app/app.py`` and the export of the model bundle.

    python -m chestxray.demo --config configs/experiments/e1_baseline.yaml --out app/model

The export writes a self-contained folder that the app (locally or on Hugging Face Spaces)
loads without the dataset or the training outputs:

- ``model.pt``: only the weights and the training config (~30 MB, never versioned in git);
- ``calibration.json``: per class, the validation threshold and the Platt parameters;
- ``app.yaml``: which experiment it is and how the value is named in the interface;
- ``examples/``: a few test images (the default rule takes, for each focus class, the true
  positive with the highest score, plus the test image with the lowest scores overall).

What the interface shows (plan, Phase 5):

- the recalibrated value of each class, ``sigmoid(a * logit + b)`` with the Platt parameters
  fitted on validation, next to the validation threshold in the same scale;
- the value is called "probabilidade estimada" only if ``--label probability`` is passed at
  export, which is decided from the test calibration curves (log it in section 10 of the plan);
  otherwise "escore do modelo";
- a summary sentence that never says "normal": the model only knows 14 diseases.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from chestxray.config import load_config
from chestxray.gradcam import DEFAULT_LAYER, Explainer, Explanation
from chestxray.metrics import apply_platt
from chestxray.plotting import br_number, pt
from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

LABELS = {"score": "Escore do modelo", "probability": "Probabilidade estimada"}
DISCLAIMER = "**Protótipo acadêmico. Não usar para diagnóstico.**"
SCOPE_NOTE = (
    "O modelo foi treinado com radiografias de tórax frontais de adultos do NIH ChestX-ray14 e só "
    "avalia 14 doenças. Imagens muito diferentes dessas (foto de tela, criança, incidência lateral, "
    "outro exame) geram resultados sem sentido."
)
PROBABILITY_NOTE = ("A probabilidade estimada foi calibrada na população do NIH; em outra população, "
                    "com outra prevalência das doenças, ela deixa de valer.")
NO_FINDING_SENTENCE = "Nenhum achado acima do limiar entre as 14 doenças avaliadas."
FOCUS_GROUP, OTHER_GROUP = "Foco do TCC", "Demais doenças"
ABOVE, BELOW = "acima do limiar", "abaixo do limiar"


def to_logit(score: np.ndarray) -> np.ndarray:
    score = np.clip(np.asarray(score, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(score / (1 - score))


@dataclass
class Result:
    """What the interface shows for one image."""

    summary: str
    table: pd.DataFrame
    values: dict[str, float]  # displayed (recalibrated) value per class
    raw_scores: dict[str, float]


class DemoModel:
    """Model bundle exported by :func:`export`, ready to score an image."""

    def __init__(self, bundle: str | Path, device: str | torch.device = "cpu"):
        bundle = Path(bundle)
        self.settings = yaml.safe_load((bundle / "app.yaml").read_text(encoding="utf-8"))
        self.calibration = json.loads((bundle / "calibration.json").read_text(encoding="utf-8"))["classes"]
        self.explainer = Explainer.from_checkpoint(bundle / "model.pt", device, [DEFAULT_LAYER])
        self.cfg = self.explainer.cfg
        self.classes = self.cfg["data"]["classes"]
        self.focus = self.cfg["data"]["focus_classes"]
        self.label = LABELS[self.settings["label"]]
        examples = bundle / "examples"
        self.examples = sorted(str(p) for p in examples.glob("*.png")) if examples.is_dir() else []

    def calibrated(self, scores: np.ndarray) -> np.ndarray:
        logits = to_logit(scores)
        return np.array([apply_platt(logits[k], self.calibration[c]["platt_a"], self.calibration[c]["platt_b"])
                         for k, c in enumerate(self.classes)], dtype=float)

    def thresholds(self) -> dict[str, float]:
        """Validation thresholds in the displayed (recalibrated) scale."""
        result = {}
        for c in self.classes:
            t = self.calibration[c]
            if t.get("threshold_calibrated") is None:  # degenerate fit (a <= 0): map the raw threshold
                t["threshold_calibrated"] = float(apply_platt(to_logit(t["threshold"]), t["platt_a"], t["platt_b"]))
            result[c] = t["threshold_calibrated"]
        return result

    def analyse(self, image: Image.Image) -> Result:
        raw = self.explainer.scores(image)
        values = self.calibrated(raw)
        return build_result(self.classes, self.focus, values, self.thresholds(), self.label, raw)

    def heatmap(self, image: Image.Image, class_name: str) -> Explanation:
        return self.explainer.explain(image, class_name)


def summary_sentence(above: list[str]) -> str:
    """Never "normal" or "healthy": the model knows only 14 diseases and misses cases."""
    if not above:
        return NO_FINDING_SENTENCE
    return "Achados acima do limiar: " + ", ".join(pt(c) for c in above) + "."


def build_result(classes: list[str], focus: list[str], values: np.ndarray, thresholds: dict[str, float],
                 label: str, raw: np.ndarray | None = None) -> Result:
    """Table (focus classes first, each group by descending value) and the summary sentence."""
    rows = []
    for k, c in enumerate(classes):
        above = values[k] >= thresholds[c]
        rows.append({"Grupo": FOCUS_GROUP if c in focus else OTHER_GROUP, "Doença": pt(c), "_class": c,
                     "_value": float(values[k]), label: f"{br_number(100 * values[k], 1)}%",
                     "Limiar": f"{br_number(100 * thresholds[c], 1)}%", "Situação": ABOVE if above else BELOW})
    table = pd.DataFrame(rows)
    table["_focus"] = table["Grupo"] == FOCUS_GROUP
    table = table.sort_values(["_focus", "_value"], ascending=[False, False])
    above = list(table.loc[table["Situação"] == ABOVE, "_class"])
    shown = table.drop(columns=["_class", "_value", "_focus"]).reset_index(drop=True)
    raw = values if raw is None else raw
    return Result(summary_sentence(above), shown,
                  {c: float(values[k]) for k, c in enumerate(classes)},
                  {c: float(raw[k]) for k, c in enumerate(classes)})


# ----------------------------------------------------------------------------- export

def pick_examples(preds: pd.DataFrame, focus: list[str], classes: list[str]) -> list[str]:
    """One true positive per focus class (highest score) and the image with the lowest mean
    score, so the demo shows both summary sentences."""
    chosen = []
    for c in focus:
        positives = preds[preds[c] == 1].sort_values([f"score_{c}", "image"], ascending=[False, True])
        chosen += [img for img in positives["image"] if img not in chosen][:1]
    mean = preds[[f"score_{c}" for c in classes]].mean(axis=1)
    quiet = preds.assign(_m=mean).sort_values(["_m", "image"])["image"]
    chosen += [img for img in quiet if img not in chosen][:1]
    return chosen


def export(cfg: dict, out: Path, label: str = "score", examples: list[str] | None = None) -> Path:
    """Write the app bundle for the experiment in ``cfg`` (see the module docstring)."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {sorted(LABELS)}")
    name = cfg["experiment"]
    run_dir = Path(cfg["paths"]["runs_dir"]) / name
    calibration = run_dir / "calibration.json"
    if not calibration.exists():
        raise SystemExit(f"{calibration} not found: run the test evaluation first")
    state = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / name / "best.pt", map_location="cpu",
                       weights_only=False)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": state["config"], "model": state["model"], "epoch": state.get("epoch")}, out / "model.pt")
    shutil.copyfile(calibration, out / "calibration.json")

    preds_path = run_dir / "preds_test.csv"
    if examples is None:
        preds = pd.read_csv(preds_path)
        examples = pick_examples(preds, cfg["data"]["focus_classes"], cfg["data"]["classes"])
    examples_dir = out / "examples"
    if examples_dir.exists():
        shutil.rmtree(examples_dir)
    examples_dir.mkdir()
    for image in examples:
        shutil.copyfile(Path(cfg["paths"]["data_dir"]) / "images" / image, examples_dir / image)
    settings = {"experiment": name, "label": label, "best_epoch": state.get("epoch"), "examples": examples}
    (out / "app.yaml").write_text(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Exported %s to %s (%s; %d examples)", name, out, LABELS[label], len(examples))
    return out


def check_bundle(bundle: Path, cfg: dict, n: int = 3, tolerance: float = 1e-4) -> float:
    """Phase 5 acceptance: the bundle reproduces ``preds_test.csv`` on ``n`` test images.
    Returns the largest absolute difference in raw score."""
    model = DemoModel(bundle)
    preds = pd.read_csv(Path(cfg["paths"]["runs_dir"]) / cfg["experiment"] / "preds_test.csv").head(n)
    worst = 0.0
    for row in preds.itertuples(index=False):
        with Image.open(Path(cfg["paths"]["data_dir"]) / "images" / row.image) as img:
            result = model.analyse(img.convert("L"))
        expected = np.array([getattr(row, f"score_{c}") for c in model.classes])
        got = np.array([result.raw_scores[c] for c in model.classes])
        worst = max(worst, float(np.abs(got - expected).max()))
    if worst > tolerance:
        raise SystemExit(f"Bundle scores differ from preds_test.csv by {worst:.2e} (> {tolerance})")
    logger.info("Bundle matches preds_test.csv on %d images (max difference %.2e)", n, worst)
    return worst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the model bundle used by the demo app")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    parser.add_argument("--out", type=Path, default=Path("app/model"))
    parser.add_argument("--label", choices=sorted(LABELS), default="score",
                        help="'probability' only if the test calibration after Platt is close to the diagonal")
    parser.add_argument("--examples", nargs="*", help="test images to ship as examples (default: fixed rule)")
    parser.add_argument("--check", type=int, default=3, help="test images used to check the bundle (0 = skip)")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.paths)
    export(cfg, args.out, args.label, args.examples)
    if args.check:
        check_bundle(args.out, cfg, args.check)


if __name__ == "__main__":
    main()
