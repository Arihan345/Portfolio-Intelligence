from datetime import datetime

from pydantic import BaseModel


class PipelineStatusResponse(BaseModel):
    """Mirrors the real pipeline_runs table's columns (Phase 3 DDL)."""

    run_id: str
    dag_id: str
    status: str
    rows_processed: int | None
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None
    error_message: str | None
