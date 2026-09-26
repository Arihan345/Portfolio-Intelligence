"""Integration test against the real MLflow store: trains a
deliberately bad model (random labels, so it has no real relationship
to the test set) and confirms the promotion gate refuses to promote it.
Runs against the actual sqlite-backed store (mlops.config.TRACKING_URI),
not a mock -- this is meant to prove the gate really works, not just
that its arithmetic is correct in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ml.baseline import predict_baseline
from ml.dataset import FEATURE_COLUMNS, build_dataset
from ml.data.fetch_price_history import load_cached
from ml.evaluation.evaluation import compare_to_baseline, evaluate_binary_classifier
from ml.splits import chronological_split
from mlops.registry import get_model_info, promote_if_beats_baseline, register_model
from mlops.tracking import log_training_run

TEST_MODEL_NAME = "test_promotion_gate_model"


def _train_deliberately_bad_model(dataset: pd.DataFrame) -> dict:
    """Same pipeline as ml.train.train_and_evaluate, except the model
    is fit on RANDOMLY SHUFFLED training labels -- so anything it
    learned is noise, unrelated to the real test-set labels, and it
    must not be able to clear a real improvement-over-baseline bar."""
    masks = chronological_split(dataset["date"], train_frac=0.70, val_frac=0.15)
    X, y = dataset[FEATURE_COLUMNS], dataset["label"]

    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]

    rng = np.random.default_rng(0)
    y_train_shuffled = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)

    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=0, eval_metric="logloss")
    model.fit(X_train, y_train_shuffled)

    test_pred = model.predict(X_test)
    test_score = model.predict_proba(X_test)[:, 1]
    model_metrics = evaluate_binary_classifier(y_test.to_numpy(), test_pred, test_score)

    baseline_pred_full = predict_baseline(dataset["volatility_21d"], dataset["vol_threshold"].iloc[0])
    baseline_pred_test = baseline_pred_full[masks["test"]].astype(int).to_numpy()
    baseline_metrics = evaluate_binary_classifier(y_test.to_numpy(), baseline_pred_test)

    comparison = compare_to_baseline(baseline_metrics, model_metrics)
    feature_importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)

    return {
        "model": model,
        "train_size": len(X_train),
        "val_size": int(masks["val"].sum()),
        "test_size": len(X_test),
        "label_rate_by_split": {
            "train": float(y_train.mean()),
            "val": float(y[masks["val"]].mean()),
            "test": float(y_test.mean()),
        },
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "comparison": comparison,
        "feature_importances": feature_importances,
    }


def test_promotion_gate_rejects_a_model_trained_on_random_labels():
    prices = load_cached()
    dataset = build_dataset(prices)

    bad_result = _train_deliberately_bad_model(dataset)
    print(f"\nBad model (random labels) metrics: {bad_result['model_metrics']}")
    print(f"Baseline metrics: {bad_result['baseline_metrics']}")
    print(f"delta_f1 (model - baseline): {bad_result['comparison']['deltas']['f1']:.4f}")

    run_id = log_training_run(prices, bad_result)
    version = register_model(run_id, model_name=TEST_MODEL_NAME)

    result = promote_if_beats_baseline(TEST_MODEL_NAME, version)

    assert result.promoted is False, (
        f"promotion gate incorrectly promoted a random-label model! "
        f"f1_improvement={result.candidate_f1_improvement}, reason={result.reason}"
    )
    assert "rejected" in result.reason

    # The rejection must be visible on the registered version itself,
    # not just returned and discarded by this test.
    info = get_model_info(TEST_MODEL_NAME, alias_or_version=version)
    assert info["promotion_status"] == "rejected"
