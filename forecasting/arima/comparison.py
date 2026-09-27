"""Combines a Monte Carlo run with an ARIMA forecast on the same
portfolio-value scale and the same forecast horizon, so the two methods
can be read side by side -- two different lenses on the future, not
merged into one number (see forecasting/__init__.py's module docstring
for the conceptual distinction).

ANCHOR-DATE FAIRNESS: Phase 6's standalone /monte-carlo endpoint derives
its drift/volatility from mart_portfolio_performance -- the real NAV
series, but only covering the original demo transaction window ending
2024-06-30 (see api/dependencies.py's ANALYSIS_END). ARIMA here forecasts
forward from the LATEST date in the extended 5-year price cache (today,
per forecasting.arima.series). Comparing Phase 6's endpoint directly to
ARIMA would silently compare forecasts anchored at two different "todays"
using two different information sets -- not a fair comparison. So the
comparison built here instead derives Monte Carlo's own drift/volatility
parameters FROM THE SAME extended series ARIMA uses, anchored at the
same latest date, before simulating with monte_carlo's own (unmodified)
GBM math. The standalone Phase 6 endpoint is untouched and still serves
its own page using the original short-window estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.performance.returns import cagr
from analytics.risk.risk import annualized_volatility
from monte_carlo.outputs import percentile_bands
from monte_carlo.simulate import simulate_portfolio_gbm

DEFAULT_N_SIMULATIONS = 10_000


def _to_mu_gbm(historical_cagr: float, sigma_annual: float) -> float:
    """Same conversion as monte_carlo.params._to_mu_gbm (see that
    module's docstring for the full CAGR-vs-continuous-compounding
    derivation) -- reproduced here rather than imported since it is a
    private helper of that module and this is a genuinely separate
    parameter-estimation path (see module docstring above)."""
    return np.log1p(historical_cagr) + 0.5 * sigma_annual ** 2


@dataclass
class MonteCarloForComparison:
    s0: float
    cagr: float
    sigma_annual: float
    percentile_bands: dict[str, float]


def monte_carlo_from_extended_series(
    series: pd.Series, horizon_days: int, n_simulations: int = DEFAULT_N_SIMULATIONS, seed: int | None = None
) -> MonteCarloForComparison:
    daily_returns = series.pct_change().dropna()
    sigma = annualized_volatility(daily_returns)
    s0 = float(series.iloc[-1])
    start_date = series.index[0]
    end_date = series.index[-1]
    start_date = start_date.date() if hasattr(start_date, "date") else start_date
    end_date = end_date.date() if hasattr(end_date, "date") else end_date
    historical_cagr = cagr(float(series.iloc[0]), s0, start_date, end_date)
    mu_gbm = _to_mu_gbm(historical_cagr, sigma)

    paths = simulate_portfolio_gbm(s0, mu_gbm, sigma, horizon_days, n_simulations, seed=seed)
    terminal = paths[:, -1]

    return MonteCarloForComparison(
        s0=s0,
        cagr=historical_cagr,
        sigma_annual=sigma,
        percentile_bands=percentile_bands(terminal),
    )


@dataclass
class ForecastComparison:
    horizon_days: int
    initial_value: float
    monte_carlo_percentile_bands: dict[str, float]
    monte_carlo_cagr: float
    monte_carlo_sigma_annual: float
    arima_point_forecast: float
    arima_intervals: dict[float, tuple[float, float]]
    arima_order: tuple[int, int, int]
    agreement_note: str


def compare(
    mc: MonteCarloForComparison, arima_point_forecast: float, arima_intervals: dict[float, tuple[float, float]],
    arima_order: tuple[int, int, int], horizon_days: int,
) -> ForecastComparison:
    """Reports where the two methods agree/diverge rather than
    reconciling them -- a real disagreement is itself informative about
    forecast uncertainty (same spirit as the baseline-vs-asset-level
    Monte Carlo divergence finding from Phase 6)."""
    bands = mc.percentile_bands
    if bands["p25"] <= arima_point_forecast <= bands["p75"]:
        agreement = (
            f"ARIMA's point forecast ({arima_point_forecast:,.0f}) falls within Monte Carlo's "
            f"25th-75th percentile band ({bands['p25']:,.0f}-{bands['p75']:,.0f}) -- the two "
            "methods broadly agree on the central outcome, despite using genuinely different "
            "techniques (pattern-fit extrapolation vs. distributional simulation)."
        )
    elif arima_point_forecast < bands["p5"] or arima_point_forecast > bands["p95"]:
        agreement = (
            f"ARIMA's point forecast ({arima_point_forecast:,.0f}) falls OUTSIDE Monte Carlo's "
            f"5th-95th percentile range ({bands['p5']:,.0f}-{bands['p95']:,.0f}) -- the two methods "
            "meaningfully disagree here. This is a real finding about forecast uncertainty, not "
            "an error to reconcile: ARIMA extrapolates the fitted historical pattern directly, "
            "while Monte Carlo simulates around a drift/volatility estimate -- a trending series "
            "with a random-walk-like ARIMA fit and a GBM drift estimated from the same series can "
            "legitimately diverge like this."
        )
    else:
        agreement = (
            f"ARIMA's point forecast ({arima_point_forecast:,.0f}) falls outside Monte Carlo's "
            f"central 25th-75th band but within its 5th-95th range ({bands['p5']:,.0f}-{bands['p95']:,.0f}) "
            "-- a moderate difference between the two methods' central estimates."
        )

    return ForecastComparison(
        horizon_days=horizon_days,
        initial_value=mc.s0,
        monte_carlo_percentile_bands=bands,
        monte_carlo_cagr=mc.cagr,
        monte_carlo_sigma_annual=mc.sigma_annual,
        arima_point_forecast=arima_point_forecast,
        arima_intervals=arima_intervals,
        arima_order=arima_order,
        agreement_note=agreement,
    )
