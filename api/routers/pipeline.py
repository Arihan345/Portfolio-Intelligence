"""Pipeline status endpoint: queries the real pipeline_runs table
(Phase 3's warehouse design) directly -- no separate pipeline-tracking
state invented for the API."""
from __future__ import annotations

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException

from analytics.data_access import engine

from api.schemas.pipeline_status import PipelineStatusResponse

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/status", response_model=PipelineStatusResponse)
def pipeline_status() -> PipelineStatusResponse:
    with engine.begin() as conn:
        row = conn.execute(
            sa.text(
                """
                SELECT run_id, dag_id, status, rows_processed, started_at,
                       completed_at, duration_seconds, error_message
                FROM pipeline_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
        ).mappings().fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="no pipeline runs recorded yet")

    return PipelineStatusResponse(**dict(row))


@router.get("/runs", response_model=list[PipelineStatusResponse])
def pipeline_runs(limit: int = 20) -> list[PipelineStatusResponse]:
    """History list, additive to /status (integration-discovered gap:
    the Phase 10 Pipeline Monitoring page needs a run history table,
    which a single-latest-row response can't provide). Same table,
    same columns, just more rows -- /status is untouched."""
    with engine.begin() as conn:
        rows = conn.execute(
            sa.text(
                """
                SELECT run_id, dag_id, status, rows_processed, started_at,
                       completed_at, duration_seconds, error_message
                FROM pipeline_runs
                ORDER BY started_at DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().fetchall()

    return [PipelineStatusResponse(**dict(row)) for row in rows]
