"""Walk-forward (rolling-origin) cross-validation: a supplementary
diagnostic alongside the primary single chronological split (Phase 7's
train_and_evaluate remains the production promotion gate; this module
never replaces it -- see README.md's "Known limitations" section for
why this exists).

Definition: split the full sorted date range into N+1 contiguous
blocks. For fold i (1..N):
    train = blocks[0..i-1]   (an EXPANDING window: fold i's train set
                               is a superset of fold i-1's)
    test  = block[i]         (the next block forward in time)
Report baseline vs. model F1 (and the delta) on EACH fold's own test
block separately, rather than a single aggregate number -- this is the
whole point: it shows whether the model beats baseline consistently
across regimes or only in some of them, which a single 70/15/15 split
cannot distinguish from "got lucky/unlucky with one split."

SCOPING DECISION (documented, not hidden): this reuses the SAME
per-ticker label threshold already computed once in ml.dataset.build_
dataset (from that ticker's global 70%-train-period, per the Task 1
threshold fix). It does NOT re-derive a fresh point-in-time threshold
per fold. A fully rigorous walk-forward evaluation would recompute the
threshold per fold from only that fold's own training data; doing so
would require restructuring build_dataset to run once per fold (a
larger change). This module walks the MODEL's train/test split forward
across folds while holding the label definition fixed -- which is
still far more informative than a single split, and is honestly
disclosed as a simplification here and in README.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from ml.baseline import predict_baseline
from ml.dataset import FEATURE_COLUMNS
from ml.evaluation.evaluation import compare_to_baseline, evaluate_binary_classifier

DEFAULT_N_FOLDS = 5


@dataclass
class WalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    label_rate_train: float
    label_rate_test: float
    baseline_f1: float
    model_f1: float
    delta_f1: float
    model_beats_baseline: bool


def walk_forward_splits(dates: pd.Series, n_folds: int = DEFAULT_N_FOLDS) -> list[dict]:
    """Returns a list of n_folds {train, test} boolean-mask dicts,
    expanding-window, rolling-origin. Block i+1 (index i) of n_folds+1
    contiguous equal-sized blocks over the sorted distinct dates is
    each fold's test set; everything before it is that fold's train set."""
    unique_dates = np.sort(dates.unique())
    n = len(unique_dates)
    n_blocks = n_folds + 1
    block_edges = [unique_dates[int(n * k / n_blocks) - 1] for k in range(1, n_blocks + 1)]
    # block_edges[k] = last date of block k (0-indexed); block_edges[-1] == unique_dates[-1]

    folds = []
    for i in range(n_folds):
        train_end = block_edges[i]
        test_end = block_edges[i + 1]
        train_mask = (dates <= train_end).to_numpy()
        test_mask = ((dates > train_end) & (dates <= test_end)).to_numpy()
        folds.append({"train": train_mask, "test": test_mask})
    return folds


def run_walk_forward_evaluation(
    dataset: pd.DataFrame, n_folds: int = DEFAULT_N_FOLDS
) -> list[WalkForwardFold]:
    """Runs the same baseline + XGBoost comparison as
    ml.train.train_and_evaluate, once per fold, using the SAME
    hyperparameters (see ml.train's own docstring for why they were
    chosen) -- only the train/test split changes per fold."""
    folds = walk_forward_splits(dataset["date"], n_folds)
    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]

    results = []
    for i, masks in enumerate(folds, start=1):
        X_train, y_train = X[masks["train"]], y[masks["train"]]
        X_test, y_test = X[masks["test"]], y[masks["test"]]

        if y_train.nunique() < 2 or y_test.nunique() < 2 or len(X_test) == 0:
            # Not enough class diversity in this fold to fit/evaluate a
            # classifier meaningfully -- skip rather than report a
            # fabricated/degenerate metric.
            continue

        n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
        )
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_score = model.predict_proba(X_test)[:, 1]
        model_metrics = evaluate_binary_classifier(y_test.to_numpy(), test_pred, test_score)

        baseline_pred_full = predict_baseline(dataset["volatility_21d"], dataset["vol_threshold"])
        baseline_pred_test = baseline_pred_full[masks["test"]].astype(int).to_numpy()
        baseline_metrics = evaluate_binary_classifier(y_test.to_numpy(), baseline_pred_test)

        comparison = compare_to_baseline(baseline_metrics, model_metrics)

        fold_dates = dataset.loc[masks["train"], "date"]
        test_dates = dataset.loc[masks["test"], "date"]
        results.append(
            WalkForwardFold(
                fold=i,
                train_start=str(fold_dates.min()),
                train_end=str(fold_dates.max()),
                test_start=str(test_dates.min()),
                test_end=str(test_dates.max()),
                train_size=len(X_train),
                test_size=len(X_test),
                label_rate_train=float(y_train.mean()),
                label_rate_test=float(y_test.mean()),
                baseline_f1=baseline_metrics["f1"],
                model_f1=model_metrics["f1"],
                delta_f1=comparison["deltas"]["f1"],
                model_beats_baseline=comparison["deltas"]["f1"] > 0,
            )
        )
    return results
