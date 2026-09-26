"""Shared FastAPI dependencies: DB access and the portfolio-exists
check every /portfolio/{id}/* endpoint needs before calling into
analytics/monte_carlo (those modules assume a valid portfolio_id and
would otherwise just return empty DataFrames rather than a clear 404).
"""
from __future__ import annotations

import sqlalchemy as sa
from fastapi import HTTPException

from analytics.data_access import engine

# Matches the real fetched data window used throughout Phase 5/6's own
# demo scripts (warehouse OHLCV was loaded for Jan-Jun 2024 only -- see
# warehouse/load/run_load.py) -- analytics endpoints reuse this same
# window rather than inventing a different one.
ANALYSIS_START = "2024-01-01"
ANALYSIS_END = "2024-06-30"


def get_portfolio_or_404(portfolio_id: int) -> None:
    with engine.begin() as conn:
        exists = conn.execute(
            sa.text("SELECT 1 FROM dim_portfolio WHERE portfolio_id = :pid"),
            {"pid": portfolio_id},
        ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail=f"portfolio {portfolio_id} not found")
