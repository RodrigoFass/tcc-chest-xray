import numpy as np
import pytest
import torch
from torch import nn

from chestxray.metrics import mean_auc, per_class_auc
from chestxray.models.densenet import DenseNet121MultiLabel
from chestxray.train import compute_pos_weight, select_amp_dtype


def test_model_outputs_logits_and_has_non_inplace_relu():
    model = DenseNet121MultiLabel(num_classes=14, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 14)
    assert isinstance(model.relu, nn.ReLU) and not model.relu.inplace  # Grad-CAM target (plan 3.11)
    assert model.classifier.in_features == 1024


def test_memory_efficient_model_gives_the_same_output():
    torch.manual_seed(0)
    plain = DenseNet121MultiLabel(pretrained=False).eval()
    efficient = DenseNet121MultiLabel(pretrained=False, memory_efficient=True).eval()
    efficient.load_state_dict(plain.state_dict())
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(plain(x), efficient(x), atol=1e-5)


def test_per_class_auc_is_nan_without_both_classes():
    y_true = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]])
    y_score = np.array([[0.9, 0.1, 0.5], [0.2, 0.3, 0.4], [0.8, 0.2, 0.6], [0.1, 0.4, 0.7]])
    aucs = per_class_auc(y_true, y_score)
    assert aucs[0] == 1.0 and np.isnan(aucs[1]) and np.isnan(aucs[2])
    assert mean_auc(aucs) == 1.0
    assert np.isnan(mean_auc(np.array([np.nan, np.nan])))


def test_pos_weight_is_negatives_over_positives():
    labels = torch.tensor([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    assert compute_pos_weight(labels).tolist() == [1.0, 3.0, 4.0]  # no positives: negatives / 1
    assert compute_pos_weight(labels, max_weight=2.0).tolist() == [1.0, 2.0, 2.0]


def test_amp_dtype_selection():
    assert select_amp_dtype(torch.device("cpu"), enabled=True) is None
    assert select_amp_dtype(torch.device("cuda"), enabled=False) is None
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        expected = torch.bfloat16 if major >= 8 else torch.float16
        assert select_amp_dtype(torch.device("cuda"), enabled=True) == expected
