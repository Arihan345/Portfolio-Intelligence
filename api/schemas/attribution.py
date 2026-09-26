from datetime import date

from pydantic import BaseModel


class AttributionResponse(BaseModel):
    portfolio_id: int
    """Attribution window: the one stable-composition sub-period (no
    BUY/SELL) within the loaded example -- see
    analytics/run_analytics_demo.py's ATTRIBUTION section for why a
    single fixed window is required for the weight*return identity to
    reconcile, and analytics.attribution.attribution's module docstring
    for the underlying definition."""
    window_start: date
    window_end: date
    asset_contribution: dict[str, float]
    sector_contribution: dict[str, float]
    total_return_from_contributions: float
    actual_holdings_return: float
