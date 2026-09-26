from datetime import date

from pydantic import BaseModel


class MaxDrawdown(BaseModel):
    """Mirrors analytics.risk.risk.max_drawdown's return dict."""

    max_drawdown: float
    peak_date: date
    trough_date: date
    recovery_date: date | None
    drawdown_duration_days: int


class BetaAlpha(BaseModel):
    beta: float
    alpha_annualized: float


class RiskResponse(BaseModel):
    portfolio_id: int
    risk_free_rate_annual: float
    var_confidence: float
    annualized_volatility: float
    downside_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta_alpha: BetaAlpha
    max_drawdown: MaxDrawdown
    historical_var: float
    parametric_var: float
    conditional_var: float
    correlation_matrix: dict[str, dict[str, float]]
