import copy
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from PIL import Image

from chestxray import gradcam as gc
from chestxray import webdemo as wd
from chestxray.config import load_config
from chestxray.inference import prepare_image
from chestxray.models.densenet import build_model

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
CLASSES = BASE["data"]["classes"]


@pytest.fixture
def tiny():
    cfg = copy.deepcopy(BASE)
    cfg["data"]["image_size"] = 64
    cfg["model"]["pretrained"] = False
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    # non-trivial batch-norm statistics, so the closed form is tested with a real affine map
    norm = model.features.norm5
    with torch.no_grad():
        norm.running_mean.uniform_(-0.5, 0.5)
        norm.running_var.uniform_(0.5, 2.0)
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.uniform_(-0.2, 0.2)
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8), "L") for _ in range(3)]
    return cfg, model, images


def test_gradcamnet_logits_and_heatmaps_match_pytorch_grad_cam(tiny):
    cfg, model, images = tiny
    net = wd.GradCamNet(model).eval()
    explainer = gc.Explainer(model, cfg, ["denseblock4"])
    for image in images:
        x = prepare_image(image, cfg).unsqueeze(0)
        with torch.no_grad():
            logits, cams = net(x)
            expected_logits = model(x)
        np.testing.assert_allclose(logits.numpy(), expected_logits.numpy(), atol=1e-5)
        assert cams.shape == (1, len(CLASSES), 2, 2)  # 64x64 input -> 2x2 grid
        for c in ["Pneumonia", "Atelectasis", "Effusion", "Hernia"]:
            got = wd.heatmap_from_cam(cams[0, CLASSES.index(c)].numpy(), 64)
            np.testing.assert_allclose(got, explainer.heatmap(image, c, layer="denseblock4"), atol=1e-5)
    explainer.close()


def test_onnx_export_matches_torch(tiny, tmp_path):
    pytest.importorskip("onnxscript")
    onnxruntime = pytest.importorskip("onnxruntime")
    cfg, model, images = tiny
    path = tmp_path / "model.onnx"
    wd.export_onnx(model, cfg, path)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    net = wd.GradCamNet(model).eval()
    for image in images:
        x = prepare_image(image, cfg).unsqueeze(0)
        logits, cams = session.run(None, {"input": x.numpy()})
        with torch.no_grad():
            expected_logits, expected_cams = net(x)
        np.testing.assert_allclose(logits, expected_logits.numpy(), atol=1e-4)
        np.testing.assert_allclose(cams, expected_cams.numpy(), atol=1e-4)


def test_jet_lut_is_matplotlibs_colormap():
    lut = np.array(wd.jet_lut())
    values = np.linspace(0, 1, 1001, dtype=np.float32)
    index = np.minimum((values * len(lut)).astype(int), len(lut) - 1)
    np.testing.assert_array_equal(lut[index], matplotlib.colormaps["jet"](values)[:, :3])


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_javascript_ports_reproduce_python(tmp_path):
    """webapp/preprocess.js and webapp/heatmap.js give what Pillow, torchvision,
    pytorch-grad-cam and matplotlib give in the Python pipeline."""
    rng = np.random.default_rng(1)
    cfg = copy.deepcopy(BASE)

    def gray(h, w):
        return rng.integers(0, 256, (h, w), dtype=np.uint8)

    resize_cases = [(gray(280, 300), 256, 256, "lanczos"), (gray(256, 256), 224, 224, "bilinear"),
                    (gray(90, 100), 256, 256, "lanczos"), (gray(300, 256), 256, 256, "lanczos"),
                    (gray(256, 500), 256, 256, "lanczos"), (gray(1024, 1024), 256, 256, "lanczos")]
    filters = {"lanczos": Image.Resampling.LANCZOS, "bilinear": Image.Resampling.BILINEAR}
    rgba = rng.integers(0, 256, (40, 50, 4), dtype=np.uint8)
    prepare_cases = [gray(280, 300), gray(256, 256)]
    cams = [np.maximum(rng.normal(size=(7, 7)), 0).astype(np.float32), np.zeros((7, 7), np.float32)]
    view = gray(32, 32)
    heat = rng.random((32, 32), dtype=np.float32)
    heat[0, 0], heat[0, 1] = 0.0, 1.0

    fixture = {
        "params": {"stored_size": 256, "image_size": 224, "mean": wd.IMAGENET_MEAN, "std": wd.IMAGENET_STD},
        "resize": [{"pixels": a.ravel().tolist(), "width": a.shape[1], "height": a.shape[0],
                    "out_width": ow, "out_height": oh, "filter": f} for a, ow, oh, f in resize_cases],
        "gray": [{"rgba": rgba.ravel().tolist(), "width": 50, "height": 40}],
        "prepare": [{"pixels": a.ravel().tolist(), "width": a.shape[1], "height": a.shape[0]} for a in prepare_cases],
        "heatmap": [{"cam": c.ravel().tolist(), "height": 7, "width": 7, "size": 224} for c in cams],
        "overlay": [{"view": view.ravel().tolist(), "heatmap": heat.ravel().tolist()}],
        "lut": wd.jet_lut(),
        "alpha": gc.HEATMAP_ALPHA,
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    run = subprocess.run(["node", str(ROOT / "tests" / "webapp_ports.mjs"), str(path)],
                         capture_output=True, text=True, check=True)
    got = json.loads(run.stdout)

    for (a, ow, oh, f), result in zip(resize_cases, got["resize"]):
        expected = np.asarray(Image.fromarray(a).resize((ow, oh), filters[f]))
        np.testing.assert_array_equal(np.array(result, dtype=np.uint8).reshape(oh, ow), expected)

    expected_gray = np.asarray(Image.fromarray(rgba, "RGBA").convert("L"))
    np.testing.assert_array_equal(np.array(got["gray"][0], dtype=np.uint8).reshape(40, 50), expected_gray)

    for a, result in zip(prepare_cases, got["prepare"]):
        image = Image.fromarray(a)
        expected_view = np.rint(gc.network_view(image, cfg) * 255).astype(np.uint8)
        np.testing.assert_array_equal(np.array(result["view"], dtype=np.uint8).reshape(224, 224), expected_view)
        expected_input = prepare_image(image, cfg).numpy().ravel()
        np.testing.assert_allclose(np.array(result["input"], dtype=np.float32), expected_input, atol=1e-6)

    for cam, result in zip(cams, got["heatmap"]):
        np.testing.assert_allclose(np.array(result).reshape(224, 224), wd.heatmap_from_cam(cam, 224), atol=1e-5)

    expected_overlay = gc.overlay_heatmap(view.astype(np.float32) / 255.0, heat)
    np.testing.assert_array_equal(np.array(got["overlay"][0], dtype=np.uint8).reshape(32, 32, 3), expected_overlay)
