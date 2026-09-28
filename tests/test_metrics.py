import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from chestxray import metrics as mt


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = (rng.random((500, 4)) < [0.3, 0.05, 0.5, 0.01]).astype(float)
    scores = np.clip(y * 0.3 + rng.random((500, 4)), 0, 1)
    scores[:, 2] = np.round(scores[:, 2], 1)  # many ties
    return y, scores


def test_auc_and_average_precision_match_sklearn(data):
    y, scores = data
    auc = mt.per_class_auc(y, scores)
    ap = mt.per_class_average_precision(y, scores)
    for k in range(y.shape[1]):
        assert auc[k] == pytest.approx(roc_auc_score(y[:, k], scores[:, k]), abs=1e-12)
        assert ap[k] == pytest.approx(average_precision_score(y[:, k], scores[:, k]), abs=1e-12)


def test_auc_is_nan_without_both_classes():
    y = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]], dtype=float)
    scores = np.array([[0.9, 0.1, 0.5], [0.2, 0.3, 0.4], [0.8, 0.2, 0.6], [0.1, 0.4, 0.7]])
    auc = mt.per_class_auc(y, scores)
    assert auc[0] == 1.0 and np.isnan(auc[1]) and np.isnan(auc[2])
    assert mt.mean_auc(auc) == 1.0
    assert np.isnan(mt.mean_auc(np.array([np.nan, np.nan])))
    assert np.isnan(mt.per_class_average_precision(y, scores)[1])


def test_patient_bootstrap_keeps_patients_whole_and_is_seeded():
    patients = np.array([7, 7, 7, 3, 3, 9, 1, 1, 1, 1])
    samples = list(mt.patient_bootstrap(patients, n_boot=50, seed=1))
    for idx in samples:
        drawn = patients[idx]
        for p in np.unique(drawn):  # a drawn patient always brings all of its images
            assert (drawn == p).sum() % (patients == p).sum() == 0
        assert len(np.unique(drawn)) <= 4
    again = list(mt.patient_bootstrap(patients, n_boot=50, seed=1))
    assert all(np.array_equal(a, b) for a, b in zip(samples, again))


def test_percentile_ci_ignores_nan():
    samples = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 7.0]])
    lo, hi = mt.percentile_ci(samples, level=0.5)
    assert lo[0] == pytest.approx(1.5) and hi[0] == pytest.approx(2.5)
    assert lo[1] == pytest.approx(5.5) and hi[1] == pytest.approx(6.5)


def test_youden_threshold_and_stats():
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.5, 0.7, 0.9])
    threshold = mt.youden_threshold(y, scores)
    stats = mt.threshold_stats(y, scores, threshold)
    assert threshold == pytest.approx(0.5)  # sens 1, spec 3/4 -> J = 0.75, the best cut
    assert (stats["tp"], stats["fp"], stats["fn"], stats["tn"]) == (3, 1, 0, 3)
    assert stats["sensitivity"] == 1.0 and stats["specificity"] == pytest.approx(0.75)
    assert stats["f1"] == pytest.approx(2 * 0.75 / 1.75)


def test_platt_recovers_the_true_calibration():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-(0.5 * logits - 1.0)))
    a, b = mt.fit_platt(logits, y)
    assert a == pytest.approx(0.5, abs=0.05) and b == pytest.approx(-1.0, abs=0.05)
    calibrated = mt.apply_platt(logits, a, b)
    assert mt.brier_score(y, calibrated) < mt.brier_score(y, 1 / (1 + np.exp(-logits)))
