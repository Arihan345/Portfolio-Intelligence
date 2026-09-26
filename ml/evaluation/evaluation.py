"""Evaluation: compares a model's predictions against the baseline on
the held-out chronological test set, using precision/recall/F1/ROC-AUC/
PR-AUC. Reuses sklearn.metrics rather than reimplementing these
standard formulas.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict:
    """y_pred is the 0/1 predicted label; y_score (if given) is a
    continuous score (e.g. predicted probability) used for ROC-AUC and
    PR-AUC, which need a ranking, not just a hard 0/1 call. If y_score
    is omitted, y_pred itself is used for both (degrades ROC-AUC/PR-AUC
    to their hard-label special case -- exactly what the baseline,
    which only produces 0/1, requires)."""
    score = y_score if y_score is not None else y_pred.astype(float)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, score)) if len(set(y_true)) > 1 else float("nan"),
    }


def compare_to_baseline(baseline_metrics: dict, model_metrics: dict) -> dict:
    """Per-metric delta (model - baseline) and an explicit verdict on
    whether the model beats the baseline on the primary metrics
    (F1 and PR-AUC -- chosen over accuracy/ROC-AUC because this is
    plausibly an imbalanced label, where those two are more informative
    of practical usefulness on the positive/"high vol" class)."""
    deltas = {k: model_metrics[k] - baseline_metrics[k] for k in baseline_metrics}
    beats_baseline = deltas["f1"] > 0 and deltas["pr_auc"] > 0
    return {"deltas": deltas, "model_beats_baseline_on_f1_and_pr_auc": beats_baseline}
