"""Fetches historical return/volatility/correlation estimates FROM
Phase 5's analytics engine -- this module never recomputes those
statistics independently, so it cannot silently drift out of sync with
analytics/'s numbers.

GBM drift convention used throughout this module: Phase 5 surfaces
CAGR as its headline annualized-return figure (analytics.performance.
returns.cagr). CAGR is defined by ANNUAL compounding: end = start *
(1 + CAGR) ** years. GBM's median path instead grows by CONTINUOUS
compounding: median_terminal = s0 * exp((mu - 0.5*sigma^2) * T) (see
simulate.py's module docstring for the Ito-correction reasoning).
These are different compounding conventions, and exp(x) != 1 + x
except for small x -- for a large rate like a 69% CAGR, exp(0.69) =
1.998 vs 1 + 0.69 = 1.69, an 18% gap. Naively setting the GBM drift's
"mu - 0.5*sigma^2" term equal to CAGR itself (an earlier version of
this module did exactly that) reproduces the WRONG median by that same
~18% -- caught by this phase's own required sanity check (median
terminal vs. s0*(1+CAGR)) coming up meaningfully off, exactly as the
brief warned it might.

The correct conversion is the continuously-compounded rate equivalent
to the given CAGR: r_continuous = ln(1 + CAGR), since
    (1 + CAGR) ** T == exp(ln(1 + CAGR) * T)
So the GBM arithmetic drift parameter fed into simulate.py is:
    mu_gbm = ln(1 + historical_CAGR) + 0.5 * sigma^2
which makes
    median_terminal = s0 * exp((mu_gbm - 0.5*sigma^2) * T)
                     = s0 * exp(ln(1 + CAGR) * T)
                     = s0 * (1 + CAGR) ** T_years
                     = s0 * (1 + CAGR)   [at T = 1 year, exactly]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics import data_access as da
from analytics.allocation.allocation import asset_allocation
from analytics.performance.returns import cagr
from analytics.risk.risk import annualized_volatility


@dataclass
class PortfolioParams:
    s0: float
    cagr: float
    sigma_annual: float
    mu_gbm: float  # arithmetic drift for simulate_portfolio_gbm


@dataclass
class AssetLevelParams:
    tickers: list[str]
    weights: np.ndarray
    s0_total: float
    cagr: np.ndarray
    sigma_annual: np.ndarray
    mu_gbm: np.ndarray
    correlation: np.ndarray


def _to_mu_gbm(historical_cagr: float, sigma_annual: float) -> float:
    return np.log1p(historical_cagr) + 0.5 * sigma_annual ** 2


def get_portfolio_params(
    portfolio_id: int, analysis_start: str, analysis_end: str
) -> PortfolioParams:
    """Baseline-model parameters, straight from Phase 5's own
    portfolio-level NAV series (mart_portfolio_performance)."""
    pp = da.get_portfolio_performance(portfolio_id)
    pp = pp[(pp["value_date"] >= analysis_start) & (pp["value_date"] <= analysis_end)]
    pp = pp.set_index("value_date")

    daily_returns = pp["daily_return"].dropna()
    sigma = annualized_volatility(daily_returns)

    start_nav = float(pp["total_nav_inr"].iloc[0])
    end_nav = float(pp["total_nav_inr"].iloc[-1])
    historical_cagr = cagr(start_nav, end_nav, pp.index[0].date(), pp.index[-1].date())

    return PortfolioParams(
        s0=end_nav,
        cagr=historical_cagr,
        sigma_annual=sigma,
        mu_gbm=_to_mu_gbm(historical_cagr, sigma),
    )


def get_asset_level_params(
    portfolio_id: int, analysis_start: str, analysis_end: str
) -> AssetLevelParams:
    """Asset-level model parameters: per-asset CAGR/volatility from
    real historical prices, current weights from mart_allocation, and
    the real correlation matrix -- all pulled from Phase 5/analytics,
    not recomputed independently."""
    ap = da.get_asset_performance(portfolio_id)
    ap = ap[(ap["as_of_date"] >= analysis_start) & (ap["as_of_date"] <= analysis_end)]

    wide_prices = ap.pivot(index="as_of_date", columns="ticker", values="last_close_inr")
    tickers = list(wide_prices.columns)
    daily_returns = wide_prices.pct_change().dropna()

    sigma = np.array([annualized_volatility(daily_returns[t]) for t in tickers])

    cagrs = []
    for t in tickers:
        series = wide_prices[t].dropna()
        c = cagr(
            float(series.iloc[0]), float(series.iloc[-1]),
            series.index[0].date(), series.index[-1].date(),
        )
        cagrs.append(c)
    cagrs = np.array(cagrs)

    correlation = daily_returns[tickers].corr().to_numpy()

    alloc = da.get_allocation(portfolio_id)
    snapshot = alloc[alloc["as_of_date"] == analysis_end]
    weights_series = asset_allocation(snapshot).reindex(tickers)
    weights = weights_series.to_numpy()
    s0_total = float(snapshot["market_value_inr"].sum())

    mu_gbm = np.array([_to_mu_gbm(c, s) for c, s in zip(cagrs, sigma)])

    return AssetLevelParams(
        tickers=tickers,
        weights=weights,
        s0_total=s0_total,
        cagr=cagrs,
        sigma_annual=sigma,
        mu_gbm=mu_gbm,
        correlation=correlation,
    )
