"""Trains the XGBoost volatility-regime classifier and evaluates it
against the persistence baseline on the chronological test split.

Hyperparameter choices (deliberately minimal -- no grid/Bayesian search
for a project this size, per this phase's own scope instruction):
  - n_estimators=200, max_depth=3, learning_rate=0.05: shallow trees
    and a modest learning rate are standard defaults for a SMALL
    tabular dataset (~1,600 training rows here) to limit overfitting;
    depth=3 keeps each tree simple relative to only 14 features.
  - subsample=0.8, colsample_bytree=0.8: standard stochastic-GBM
    regularization (each tree sees 80% of rows/columns), again aimed
    at not overfitting a small sample.
  - scale_pos_weight = (negative count / positive count) in the
    TRAINING set only: the label is imbalanced (~76/24 in this data,
    which is expected -- a 75th-percentile threshold makes ~25%
    positive by construction), so this reweights the minority "high
    vol" class instead of leaving the model biased toward predicting
    the majority class.
No hyperparameter here was tuned against the validation or test set;
the validation split exists (per the chronological split requirement)
for a future iteration to tune against, but this phase does not run a
search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ml.baseline import predict_baseline
from ml.dataset import FEATURE_COLUMNS
from ml.evaluation.evaluation import compare_to_baseline, evaluate_binary_classifier
from ml.splits import chronological_split


def train_and_evaluate(dataset: pd.DataFrame) -> dict:
    """dataset: output of ml.dataset.build_dataset (must include
    FEATURE_COLUMNS, 'label', 'date', 'ticker', 'trailing 21d vol' via
    the merged 'volatility_21d' feature column and 'vol_threshold' from
    labels). Splits chronologically PER the full set of dates (shared
    across tickers, so both tickers' rows for a date land in the same
    split -- see ml.splits.chronological_split)."""
    masks = chronological_split(dataset["date"], train_frac=0.70, val_frac=0.15)

    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]

    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["val"]], y[masks["val"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_score = model.predict_proba(X_test)[:, 1]
    model_metrics = evaluate_binary_classifier(y_test.to_numpy(), test_pred, test_score)

    # Baseline: persistence of the CURRENT trailing-21d vol regime,
    # using this same test set's own trailing_vol / threshold columns.
    baseline_pred_full = predict_baseline(
        dataset["volatility_21d"], dataset["vol_threshold"].iloc[0]
    )
    baseline_pred_test = baseline_pred_full[masks["test"]].astype(int).to_numpy()
    baseline_metrics = evaluate_binary_classifier(y_test.to_numpy(), baseline_pred_test)

    comparison = compare_to_baseline(baseline_metrics, model_metrics)

    feature_importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    return {
        "model": model,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "scale_pos_weight": float(scale_pos_weight),
        "label_rate_by_split": {
            "train": float(y_train.mean()),
            "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "comparison": comparison,
        "feature_importances": feature_importances,
        "test_dates_range": (dataset.loc[masks["test"], "date"].min(), dataset.loc[masks["test"], "date"].max()),
        "train_dates_range": (dataset.loc[masks["train"], "date"].min(), dataset.loc[masks["train"], "date"].max()),
    }
