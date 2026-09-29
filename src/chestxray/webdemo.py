"""Static web demo (Phase 5): the model runs in the visitor's browser with ONNX Runtime Web.

    python -m chestxray.webdemo --config configs/experiments/e1_baseline.yaml --out webapp/model

Hugging Face now charges for Gradio Spaces, while static Spaces stay free (plan, section 10), so
the public demo is the page in ``webapp/``: plain HTML and JavaScript that download the model
once and analyse the image on the visitor's computer, which never uploads it. The Gradio app in
``app/`` stays as the local version. This module writes what the page loads:

- ``model.onnx``: input 1x3x224x224, normalized as in evaluation; outputs the 14 logits and, for
  every class, the Grad-CAM map on the default target layer before upsampling (14x7x7), computed
  in closed form by :class:`GradCamNet`;
- ``params.json``: class names, the validation thresholds and Platt parameters (the same values
  as the Gradio bundle), the interface texts and the colormap;
- ``selftest.json``: for the example images, what the Python pipeline gives (network input,
  scores, heatmaps). ``index.html?selftest`` recomputes them in the browser and reports the
  largest differences;
- ``examples/``: the example test images, picked by the same rule as the Gradio bundle.

The page reproduces the Python pipeline step by step: grayscale conversion and Lanczos/bilinear
resizing ported from Pillow (``webapp/preprocess.js``), then the upsampling and normalization of
pytorch-grad-cam and the jet colormap of matplotlib (``webapp/heatmap.js``). Tests compare each
port with the Python original.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import scale_cam_image
from torch import nn

from chestxray.config import load_config
from chestxray.demo import (ABOVE, BELOW, FOCUS_GROUP, LABELS, NO_FINDING_SENTENCE, OTHER_GROUP,
                            PROBABILITY_NOTE, SCOPE_NOTE, pick_examples)
from chestxray.gradcam import DEFAULT_LAYER, HEATMAP_ALPHA, HEATMAP_CMAP, Explainer, network_view
from chestxray.inference import load_trained_model, prepare_image
from chestxray.plotting import CLASS_NAMES_PT
from chestxray.utils import setup_logging

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PRIVACY_NOTE = ("A imagem é analisada no seu próprio navegador: ela não é enviada a nenhum servidor.")


class GradCamNet(nn.Module):
    """The trained network with a second output: the Grad-CAM map of every class on the output
    of the last dense block (the default target layer, ``denseblock4``).

    Grad-CAM weighs each feature map A_k by the mean, over its positions, of the gradient of the
    class logit z_c. Between ``denseblock4`` and the logit there are only the final batch norm
    (in eval mode an affine map per channel, B_k = s_k A_k + t_k), a ReLU, the global average
    pool and the linear layer, so that gradient has a closed form,

        dz_c / dA_kij = w_ck * s_k * [B_kij > 0] / (H W),

    and the Grad-CAM weight is its mean, alpha_ck = w_ck * s_k * n_k / (H W)^2, where n_k counts
    the positions with B_kij > 0. The map is ReLU(sum_k alpha_ck A_k), what pytorch-grad-cam
    computes before its normalization and upsampling; a test checks that both agree.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        blocks = list(model.features.children())
        self.body = nn.Sequential(*blocks[:-1])  # up to denseblock4
        self.norm = model.features.norm5
        self.classifier = model.classifier
        scale = self.norm.weight / torch.sqrt(self.norm.running_var + self.norm.eps)
        self.register_buffer("scale", scale.detach().clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.body(x)
        b = self.norm(a)
        logits = self.classifier(torch.relu(b).mean(dim=(2, 3)))
        positions = a.shape[2] * a.shape[3]
        positive = (b > 0).to(a.dtype).sum(dim=(2, 3))                                  # N x K
        alpha = self.classifier.weight[None] * (self.scale * positive / positions ** 2)[:, None]  # N x C x K
        cams = torch.relu(torch.einsum("nck,nkhw->nchw", alpha, a))
        return logits, cams


def heatmap_from_cam(cam: np.ndarray, size: int) -> np.ndarray:
    """pytorch-grad-cam's steps after the weighted sum, for one target layer: normalize to
    [0, 1], upsample to the input size, then ReLU and normalize again."""
    upsampled = scale_cam_image(np.asarray(cam, dtype=np.float32)[None], (size, size))
    return scale_cam_image(np.maximum(upsampled, 0))[0]


def jet_lut(n: int = 256) -> list[list[float]]:
    """matplotlib's colormap as a table: for a heatmap value v in [0, 1], matplotlib takes
    entry ``min(int(v * n), n - 1)`` (RGB in [0, 1])."""
    cmap = matplotlib.colormaps[HEATMAP_CMAP].resampled(n)
    return [[float(v) for v in cmap(i)[:3]] for i in range(n)]


def export_onnx(model: nn.Module, cfg: dict, path: Path) -> None:
    size = cfg["data"]["image_size"]
    net = GradCamNet(model.cpu().eval()).eval()
    dummy = torch.zeros(1, 3, size, size)
    # one self-contained file (the weights inside), which the page loads with a single request
    torch.onnx.export(net, (dummy,), str(path), input_names=["input"], output_names=["logits", "cams"],
                      opset_version=18, dynamo=True, external_data=False)


def webapp_params(cfg: dict, calibration: dict, label: str, examples: list[str]) -> dict:
    classes = cfg["data"]["classes"]
    notes = [SCOPE_NOTE, PRIVACY_NOTE] + ([PROBABILITY_NOTE] if label == "probability" else [])
    return {
        "experiment": cfg["experiment"],
        "classes": classes,
        "names_pt": {c: CLASS_NAMES_PT.get(c, c) for c in classes},
        "focus": cfg["data"]["focus_classes"],
        "stored_size": cfg["data"]["stored_size"],
        "image_size": cfg["data"]["image_size"],
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "layer": DEFAULT_LAYER,
        "platt": {c: [calibration[c]["platt_a"], calibration[c]["platt_b"]] for c in classes},
        "thresholds": {c: calibration[c]["threshold_calibrated"] for c in classes},
        "label": LABELS[label],
        "texts": {"notes": notes, "no_finding": NO_FINDING_SENTENCE, "above": ABOVE, "below": BELOW,
                  "focus_group": FOCUS_GROUP, "other_group": OTHER_GROUP},
        "heatmap": {"alpha": HEATMAP_ALPHA, "lut": jet_lut()},
        "examples": examples,
    }


def selftest_data(explainer: Explainer, cfg: dict, images: dict[str, Image.Image]) -> dict:
    """What the Python pipeline gives for each example, for the page to reproduce: the image
    as the network sees it (0-255), the 14 scores and the heatmap of one focus class (a
    different one for each example, to keep the file small)."""
    focus = cfg["data"]["focus_classes"]
    out = {}
    for i, (name, image) in enumerate(images.items()):
        class_name = focus[i % len(focus)]
        out[name] = {"view": np.rint(network_view(image, cfg) * 255).astype(int).tolist(),
                     "scores": [float(s) for s in explainer.scores(image)],
                     "class": class_name,
                     "heatmap": explainer.heatmap(image, class_name).round(5).tolist()}
    return out


def export(cfg: dict, out: Path, label: str = "probability", examples: list[str] | None = None) -> Path:
    name = cfg["experiment"]
    run_dir = Path(cfg["paths"]["runs_dir"]) / name
    calibration_path = run_dir / "calibration.json"
    if not calibration_path.exists():
        raise SystemExit(f"{calibration_path} not found: run the test evaluation first")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["classes"]
    model, _ = load_trained_model(Path(cfg["paths"]["checkpoint_dir"]) / name / "best.pt", torch.device("cpu"))

    if examples is None:
        preds = pd.read_csv(run_dir / "preds_test.csv")
        examples = pick_examples(preds, cfg["data"]["focus_classes"], cfg["data"]["classes"])
    if out.exists():
        shutil.rmtree(out)
    (out / "examples").mkdir(parents=True)
    images = {}
    for image in examples:
        source = Path(cfg["paths"]["data_dir"]) / "images" / image
        shutil.copyfile(source, out / "examples" / image)
        with Image.open(source) as img:
            images[image] = img.convert("L")

    export_onnx(model, cfg, out / "model.onnx")
    params = webapp_params(cfg, calibration, label, examples)
    (out / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    explainer = Explainer(model, cfg, [DEFAULT_LAYER])
    try:
        selftest = selftest_data(explainer, cfg, images)
    finally:
        explainer.close()
    (out / "selftest.json").write_text(json.dumps(selftest), encoding="utf-8")
    logger.info("Exported %s to %s (%s; %d examples)", name, out, LABELS[label], len(examples))
    return out


def check_onnx(out: Path, cfg: dict, n: int = 3, tolerance: float = 1e-4) -> tuple[float, float]:
    """Phase 5 acceptance for the web version: on ``n`` test images, the ONNX model gives the
    evaluated scores (``preds_test.csv``) and the heatmaps of the Python Grad-CAM, for the three
    focus classes. Returns the largest absolute differences (score, heatmap)."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    classes, size = cfg["data"]["classes"], cfg["data"]["image_size"]
    model, _ = load_trained_model(Path(cfg["paths"]["checkpoint_dir"]) / cfg["experiment"] / "best.pt",
                                  torch.device("cpu"))
    explainer = Explainer(model, cfg, [DEFAULT_LAYER])
    preds = pd.read_csv(Path(cfg["paths"]["runs_dir"]) / cfg["experiment"] / "preds_test.csv").head(n)
    worst_score = worst_map = 0.0
    try:
        for row in preds.itertuples(index=False):
            with Image.open(Path(cfg["paths"]["data_dir"]) / "images" / row.image) as img:
                image = img.convert("L")
            logits, cams = session.run(None, {"input": prepare_image(image, cfg).unsqueeze(0).numpy()})
            scores = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float64)))
            expected = np.array([getattr(row, f"score_{c}") for c in classes])
            worst_score = max(worst_score, float(np.abs(scores - expected).max()))
            for c in cfg["data"]["focus_classes"]:
                got = heatmap_from_cam(cams[0][classes.index(c)], size)
                worst_map = max(worst_map, float(np.abs(got - explainer.heatmap(image, c)).max()))
    finally:
        explainer.close()
    if max(worst_score, worst_map) > tolerance:
        raise SystemExit(f"ONNX model differs from the Python pipeline: scores by {worst_score:.2e}, "
                         f"heatmaps by {worst_map:.2e} (> {tolerance})")
    logger.info("ONNX model matches preds_test.csv and the Python Grad-CAM on %d images "
                "(max differences: score %.2e, heatmap %.2e)", n, worst_score, worst_map)
    return worst_score, worst_map


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    parser.add_argument("--out", type=Path, default=Path("webapp/model"))
    parser.add_argument("--label", choices=sorted(LABELS), default="probability",
                        help="how the value is named; 'probability' was decided from the test calibration")
    parser.add_argument("--examples", nargs="*", help="test images to ship as examples (default: fixed rule)")
    parser.add_argument("--check", type=int, default=3, help="test images used to check the ONNX model (0 = skip)")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.paths)
    out = export(cfg, args.out, args.label, args.examples)
    if args.check:
        check_onnx(out, cfg, args.check)


if __name__ == "__main__":
    main()
