from pydantic import BaseModel


class PredictionInterval(BaseModel):
    confidence: float
    lower: float
    upper: float


class BacktestOrigin(BaseModel):
    origin_date: str
    actual_date: str
    order: list[int]
    actual: float
    point_forecast: float
    forecast_error: float
    intervals: list[PredictionInterval]
    within_interval: dict[str, bool]


class IntervalCoverage(BaseModel):
    confidence: float
    observed_coverage: float


class BacktestSummary(BaseModel):
    n_origins: int
    horizon_days: int
    mae: float
    rmse: float
    interval_coverage: list[IntervalCoverage]
    origins: list[BacktestOrigin]


class ArimaForecastResponse(BaseModel):
    portfolio_id: int
    horizon_days: int
    initial_value: float
    order: list[int]
    point_forecast: float
    intervals: list[PredictionInterval]
    disclaimer: str
    backtest: BacktestSummary


class ForecastComparisonResponse(BaseModel):
    portfolio_id: int
    horizon_days: int
    initial_value: float

    monte_carlo_percentile_bands: dict[str, float]
    monte_carlo_cagr: float
    monte_carlo_sigma_annual: float
    monte_carlo_disclaimer: str

    arima_point_forecast: float
    arima_intervals: list[PredictionInterval]
    arima_order: list[int]
    arima_disclaimer: str

    agreement_note: str
