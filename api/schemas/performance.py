from pydantic import BaseModel


class CapitalSummary(BaseModel):
    """Mirrors analytics.performance.returns.capital_summary's return dict."""

    invested_capital: float
    market_value: float
    absolute_return: float
    pct_return: float


class PerformanceResponse(BaseModel):
    portfolio_id: int
    capital_summary: CapitalSummary
    cagr: float
    monthly_returns: dict[str, float]
    latest_30d_rolling_return: float
    time_weighted_return: float
    xirr: float
