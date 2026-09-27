"""Model status endpoint: thin wrapper over Phase 8's real retrieval
function. Reports the truth -- including "no production model exists"
when that is actually the case (this project's real trained model was
correctly rejected by the promotion gate; see Phase 8's own findings),
rather than fabricating a fake production status.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlops.config import MODEL_NAME, TRACKING_URI
from mlops.registry import PRODUCTION_ALIAS, get_model_info
from mlops.tracking import _init_mlflow

from api.schemas.model_status import DatasetVersionInfo, ModelStatusResponse, WalkForwardFold

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    try:
        info = get_model_info(MODEL_NAME, alias_or_version=PRODUCTION_ALIAS)
    except MlflowException:
        # No version currently holds the "production" alias -- report
        # the latest registered version's real status instead of
        # inventing a production state that doesn't exist.
        _init_mlflow()
        client = MlflowClient(tracking_uri=TRACKING_URI)
        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        except MlflowException as exc:
            raise HTTPException(status_code=404, detail=f"model '{MODEL_NAME}' is not registered yet") from exc
        if not versions:
            raise HTTPException(status_code=404, detail=f"model '{MODEL_NAME}' has no registered versions")
        latest = max(versions, key=lambda v: int(v.version))
        info = get_model_info(MODEL_NAME, alias_or_version=latest.version)
        info["promotion_status"] = info["promotion_status"] or "not yet evaluated"

    return ModelStatusResponse(
        model_name=info["model_name"],
        version=str(info["version"]),
        run_id=info["run_id"],
        dataset_version=DatasetVersionInfo(**info["dataset_version"]),
        feature_version_hash=info["feature_version_hash"],
        training_timestamp_utc=info["training_timestamp_utc"],
        metrics=info["metrics"],
        feature_importances=info["feature_importances"],
        walk_forward_folds=[WalkForwardFold(**f) for f in info["walk_forward_folds"]],
        promotion_status=info["promotion_status"],
        promotion_reason=info.get("promotion_reason"),
    )
