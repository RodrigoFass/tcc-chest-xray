"""Metrics shared by training (validation AUC) and evaluation."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import roc_auc_score


def per_class_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """ROC AUC per column; NaN where a class has only positives or only negatives.

    That happens in small sets (e.g. the 200-image debug run has no Hernia case), and the
    mean over classes then uses ``np.nanmean``.
    """
    aucs = np.full(y_true.shape[1], np.nan)
    for k in range(y_true.shape[1]):
        if 0 < y_true[:, k].sum() < len(y_true):
            aucs[k] = roc_auc_score(y_true[:, k], y_score[:, k])
    return aucs


def mean_auc(aucs: np.ndarray) -> float:
    """Mean over the classes with a defined AUC (NaN if none)."""
    if np.isnan(aucs).all():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(aucs))
