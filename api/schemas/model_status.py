from pydantic import BaseModel


class DatasetVersionInfo(BaseModel):
    hash: str | None
    tickers: str | None
    start_date: str | None
    end_date: str | None
    row_count: str | None


class WalkForwardFold(BaseModel):
    """Mirrors mlops.registry._parse_walk_forward_folds's per-fold
    dict. Supplementary diagnostic only -- the promotion gate never
    reads this, it only gates on the primary single-split delta_f1
    (see ModelStatusResponse.metrics['delta_f1'])."""

    fold: int
    train_start: str | None
    train_end: str | None
    test_start: str | None
    test_end: str | None
    train_size: int
    test_size: int
    label_rate_train: float | None
    label_rate_test: float | None
    baseline_f1: float | None
    model_f1: float | None
    delta_f1: float | None
    model_beats_baseline: bool


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
    walk_forward_folds: list[WalkForwardFold]
    promotion_status: str | None
    promotion_reason: str | None
