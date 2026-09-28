"""Metrics shared by training (validation AUC) and evaluation (bootstrap CIs, thresholds,
calibration).

AUC and average precision are computed for all classes at once and without sklearn's input
checks, because the patient bootstrap evaluates them thousands of times; tests check that
they match ``sklearn.metrics.roc_auc_score`` and ``average_precision_score``.
"""

from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve


def per_class_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """ROC AUC per column; NaN where a class has only positives or only negatives.

    Uses the Mann-Whitney statistic with mid-ranks for ties, which equals the area under the
    ROC curve. Small sets can lack a class (e.g. the 200-image debug run has no Hernia case);
    the mean over classes then uses ``np.nanmean`` (:func:`mean_auc`).
    """
    y_true = np.asarray(y_true, dtype=float)
    n_pos = y_true.sum(axis=0)
    n_neg = len(y_true) - n_pos
    ranks = rankdata(y_score, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((ranks * y_true).sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def mean_auc(aucs: np.ndarray) -> float:
    """Mean over the classes with a defined AUC (NaN if none)."""
    if np.isnan(aucs).all():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(aucs))


def per_class_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Average precision (area under the precision-recall curve, as defined by sklearn) per
    column; NaN where a class has no positive."""
    y_true = np.asarray(y_true, dtype=float)
    result = np.full(y_true.shape[1], np.nan)
    for k in range(y_true.shape[1]):
        n_pos = y_true[:, k].sum()
        if n_pos == 0:
            continue
        order = np.argsort(-y_score[:, k], kind="mergesort")
        y, s = y_true[order, k], y_score[order, k]
        last_of_tie = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # thresholds: distinct scores
        tps = np.cumsum(y)[last_of_tie]
        precision = tps / (last_of_tie + 1)
        recall = tps / n_pos
        result[k] = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return result


def patient_bootstrap(patient_ids: np.ndarray, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Row indices of ``n_boot`` bootstrap samples drawn by patient (plan 3.9).

    Each sample draws as many patients as there are, with replacement, and takes all the
    images of every drawn patient: images of one patient are correlated, and resampling
    images instead would give intervals that are too narrow. The same ``seed`` and patients
    give the same samples, which makes paired comparisons between models possible.
    """
    codes, _ = _factorize(patient_ids)
    n_patients = codes.max() + 1
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_patients, n_patients)
        lengths = counts[chosen]
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        yield order[np.repeat(starts[chosen], lengths) + offsets]


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniques, codes = np.unique(np.asarray(values), return_inverse=True)
    return codes.ravel(), uniques


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval over axis 0, ignoring NaN (samples where a class had no case)."""
    tail = 100 * (1 - level) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(samples, tail, axis=0), np.nanpercentile(samples, 100 - tail, axis=0)


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Score threshold that maximizes sensitivity + specificity - 1 (plan 3.7)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best = np.argmax(tpr - fpr)
    return float(min(thresholds[best], 1.0))


def threshold_stats(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Confusion matrix and the usual rates for ``score >= threshold``."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_score) >= threshold
    tp, fp = int((y_pred & y_true).sum()), int((y_pred & ~y_true).sum())
    fn, tn = int((~y_pred & y_true).sum()), int((~y_pred & ~y_true).sum())

    def ratio(a, b):
        return a / b if b else float("nan")

    precision, sensitivity = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": ratio(tp + tn, len(y_true)),
        "sensitivity": sensitivity,
        "specificity": ratio(tn, tn + fp),
        "precision": precision,
        "npv": ratio(tn, tn + fn),
        "f1": ratio(2 * precision * sensitivity, precision + sensitivity) if tp else 0.0,
    }


def fit_platt(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Platt scaling: ``calibrated = sigmoid(a * logit + b)``, fitted by (unregularized)
    logistic regression on one class's validation logits."""
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(np.asarray(logits, dtype=float).reshape(-1, 1), np.asarray(y_true).astype(int))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(logits, dtype=float) + b)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))
