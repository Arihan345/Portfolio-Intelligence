"""Logs a Phase 7 training run to MLflow: hyperparameters, both models'
metrics side by side, dataset version, feature version, and the
trained model artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import pandas as pd

from mlops.config import ARTIFACT_ROOT, EXPERIMENT_NAME, TRACKING_URI, ensure_store_dirs
from mlops.versioning import compute_dataset_version, compute_feature_version


def _init_mlflow() -> None:
    ensure_store_dirs()
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_run(price_history: pd.DataFrame, train_result: dict) -> str:
    """train_result: the dict returned by ml.train.train_and_evaluate.
    Returns the MLflow run_id.

    Logs (per this phase's requirement):
      - hyperparameters: model.get_params() (every XGBoost hyperparameter
        actually used, read back from the fitted estimator itself rather
        than retyped by hand, so logged params can never drift from what
        was really used)
      - metrics: baseline_* and model_* for every evaluation metric,
        prefixed so both are visible side by side in the MLflow UI
      - dataset version: hash + tickers + date range + row count
      - feature version: hash of ml/features/features.py's source
      - training timestamp (UTC, explicit -- separate from MLflow's own
        run start time, which is also recorded automatically)
      - the trained model artifact (via mlflow.xgboost, which also
        captures the model's input/output schema)
    """
    _init_mlflow()

    dataset_version = compute_dataset_version(price_history)
    feature_version = compute_feature_version()
    model = train_result["model"]

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())

        for name, value in train_result["baseline_metrics"].items():
            mlflow.log_metric(f"baseline_{name}", value)
        for name, value in train_result["model_metrics"].items():
            mlflow.log_metric(f"model_{name}", value)
        for name, value in train_result["comparison"]["deltas"].items():
            mlflow.log_metric(f"delta_{name}", value)

        mlflow.log_params(dataset_version.as_dict())
        mlflow.log_param("feature_version_hash", feature_version)
        mlflow.log_param("training_timestamp_utc", datetime.now(timezone.utc).isoformat())
        mlflow.log_param("train_size", train_result["train_size"])
        mlflow.log_param("val_size", train_result["val_size"])
        mlflow.log_param("test_size", train_result["test_size"])

        for split, rate in train_result["label_rate_by_split"].items():
            mlflow.log_metric(f"label_rate_{split}", rate)

        for feature_name, importance in train_result["feature_importances"].items():
            mlflow.log_metric(f"feature_importance_{feature_name}", float(importance))

        mlflow.xgboost.log_model(model, name="model")

        return run.info.run_id
