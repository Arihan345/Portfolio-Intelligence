from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MonteCarloRequest(BaseModel):
    model: Literal["baseline", "asset_level"] = "baseline"
    n_simulations: int = Field(10_000, gt=0, le=200_000)
    horizon_days: int = Field(252, gt=0, le=252 * 10)
    var_confidence: float = Field(0.95, gt=0, lt=1)
    target_value: float = Field(65_000.0, gt=0)
    seed: int | None = None

    # Manual overrides for stress-testing (Phase 6 requirement): default
    # None means "use Phase 5/6's historical estimate", exactly as
    # monte_carlo.params derives it -- never recomputed differently here.
    mu_annual_override: float | None = None
    sigma_annual_override: float | None = Field(None, ge=0)

    @field_validator("n_simulations")
    @classmethod
    def _positive_sim_count(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_simulations must be positive")
        return v


class PercentileBands(BaseModel):
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


class SimulatedVaRES(BaseModel):
    var: float
    expected_shortfall: float


class DrawdownStatistics(BaseModel):
    mean_max_drawdown: float
    median_max_drawdown: float
    worst_5pct_max_drawdown: float
    best_max_drawdown: float


class MonteCarloResponse(BaseModel):
    run_id: str
    portfolio_id: int
    model: str
    n_simulations: int
    horizon_days: int
    initial_value: float
    disclaimer: str
    created_at: datetime

    percentile_bands: PercentileBands
    probability_of_loss: float
    probability_of_exceeding_target: float
    target_value: float
    simulated_var_es: SimulatedVaRES
    drawdown_statistics: DrawdownStatistics
