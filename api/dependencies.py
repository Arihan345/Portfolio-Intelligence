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
# DEMO scripts (warehouse OHLCV was loaded for Jan-Jun 2024 only -- see
# warehouse/load/run_load.py) -- kept here only for those demo scripts
# and for api/tests/test_api.py's assertions against that same demo
# fixture's own known, real, naturally-bounded date range.
#
# NOT used by the live /portfolio/{id}/* endpoints in api/routers/
# portfolio.py or api/routers/monte_carlo.py -- those used to hard-cap
# "current" at 2024-06-30 for every portfolio regardless of how much
# real data existed beyond it (confirmed real bug: a real portfolio
# uploaded with transactions through 2026 still reported
# as_of_date=2024-06-30 and a stale market value, making the platform's
# own numbers impossible to compare against the real Groww ground
# truth this project's real-upload testing depends on). Those endpoints
# now use the mart's own natural latest date / full available history
# instead of this fixed window.
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
