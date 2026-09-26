from pydantic import BaseModel


class DatasetVersionInfo(BaseModel):
    hash: str | None
    tickers: str | None
    start_date: str | None
    end_date: str | None
    row_count: str | None


class ModelStatusResponse(BaseModel):
    """Mirrors mlops.registry.get_model_info's return shape (minus the
    loaded model object itself, which isn't serializable)."""

    model_name: str
    version: str
    run_id: str
    dataset_version: DatasetVersionInfo
    feature_version_hash: str | None
    training_timestamp_utc: str | None
    metrics: dict[str, float]
    feature_importances: dict[str, float]
    promotion_status: str | None
    promotion_reason: str | None
