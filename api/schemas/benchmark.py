from pydantic import BaseModel


class BenchmarkResponse(BaseModel):
    """Mirrors analytics.benchmark.benchmark.benchmark_comparison's
    return dict exactly."""

    portfolio_id: int
    benchmark_ticker: str
    portfolio_cumulative_return: float
    benchmark_cumulative_return: float
    excess_cumulative_return: float
    tracking_error_annualized: float
    beta: float
    alpha_annualized: float
    information_ratio: float
    correlation: float
