"""Model registry: registration, the production promotion gate, and
retrieval of everything needed to reproduce what a registered model saw.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

from mlops.config import MODEL_NAME, TRACKING_URI
from mlops.tracking import _init_mlflow

PRODUCTION_ALIAS = "production"
MIN_F1_IMPROVEMENT_OVER_BASELINE = 0.05


@dataclass
class PromotionResult:
    promoted: bool
    reason: str
    candidate_version: str
    candidate_f1_improvement: float


def register_model(run_id: str, model_name: str = MODEL_NAME) -> str:
    """Registers the model artifact logged under `run_id` (see
    mlops.tracking.log_training_run) as a new version of `model_name`
    in the MLflow Model Registry. Returns the new version number
    (as a string, matching MLflow's own convention)."""
    _init_mlflow()
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    return result.version


def promote_if_beats_baseline(
    model_name: str,
    version: str,
    min_f1_improvement: float = MIN_F1_IMPROVEMENT_OVER_BASELINE,
) -> PromotionResult:
    """THE PROMOTION GATE.

    Definition: a candidate model version is promoted to the
    "production" alias if and only if
        model_f1 - baseline_f1 >= min_f1_improvement
    on the held-out chronological test set, reading both metrics back
    from the run that produced this model version (never recomputed
    here -- the gate trusts exactly what mlops.tracking logged, so it
    cannot silently diverge from Phase 7's own evaluation).

    If the candidate does NOT clear the bar, the current "production"
    alias (if any) is left untouched -- this function never moves the
    alias backward or clears it, only forward past a version that
    actually clears the bar. The rejection reason is written as a tag
    on the REJECTED version, so `mlflow ui` / the registry shows why a
    given version never became production, rather than that fact
    silently disappearing.
    """
    _init_mlflow()
    client = MlflowClient(tracking_uri=TRACKING_URI)

    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)
    metrics = run.data.metrics

    if "delta_f1" not in metrics:
        raise KeyError(
            f"run {mv.run_id} has no 'delta_f1' metric -- was it logged via "
            "mlops.tracking.log_training_run?"
        )
    f1_improvement = metrics["delta_f1"]

    if f1_improvement >= min_f1_improvement:
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version)
        reason = (
            f"promoted: model F1 beat baseline F1 by {f1_improvement:.4f}, "
            f">= required margin {min_f1_improvement:.4f}"
        )
        client.set_model_version_tag(model_name, version, "promotion_status", "promoted")
        client.set_model_version_tag(model_name, version, "promotion_reason", reason)
        return PromotionResult(True, reason, version, f1_improvement)

    reason = (
        f"rejected: model F1 beat baseline F1 by only {f1_improvement:.4f}, "
        f"below required margin {min_f1_improvement:.4f} -- previous production "
        f"model (if any) remains active"
    )
    client.set_model_version_tag(model_name, version, "promotion_status", "rejected")
    client.set_model_version_tag(model_name, version, "promotion_reason", reason)
    return PromotionResult(False, reason, version, f1_improvement)


def get_model_info(model_name: str = MODEL_NAME, alias_or_version: str = PRODUCTION_ALIAS) -> dict:
    """Retrieval function for Phase 10's Streamlit MLOps page: given a
    model version number OR an alias (default: "production"), returns
    everything needed to fully reproduce what that model saw --
    hyperparameters, dataset version, feature version, and metrics --
    plus the loadable model artifact itself.
    """
    _init_mlflow()
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # A version is always numeric-string ("1", "2", ...); anything else
    # is treated as an alias. MLflow's alias lookup raises a raw
    # TypeError (not MlflowException) when handed a non-string/numeric
    # value, so the two cases are disambiguated up front rather than by
    # catching the wrong exception type.
    if str(alias_or_version).isdigit():
        mv = client.get_model_version(model_name, str(alias_or_version))
    else:
        mv = client.get_model_version_by_alias(model_name, alias_or_version)

    run = client.get_run(mv.run_id)
    params = run.data.params
    metrics = run.data.metrics

    model_uri = f"models:/{model_name}/{mv.version}"
    model = mlflow.xgboost.load_model(model_uri)

    return {
        "model_name": model_name,
        "version": mv.version,
        "run_id": mv.run_id,
        "model": model,
        "hyperparameters": {
            k: v for k, v in params.items()
            if not k.startswith("dataset_") and k != "feature_version_hash"
        },
        "dataset_version": {
            "hash": params.get("dataset_version_hash"),
            "tickers": params.get("dataset_tickers"),
            "start_date": params.get("dataset_start_date"),
            "end_date": params.get("dataset_end_date"),
            "row_count": params.get("dataset_row_count"),
        },
        "feature_version_hash": params.get("feature_version_hash"),
        "training_timestamp_utc": params.get("training_timestamp_utc"),
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("feature_importance_")},
        "promotion_status": mv.tags.get("promotion_status"),
        "promotion_reason": mv.tags.get("promotion_reason"),
    }
