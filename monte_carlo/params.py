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
    """Baseline-model parameters, from a real per-ticker weighted value
    series (forecasting.arima.series.build_extended_portfolio_series)
    rather than mart_portfolio_performance's raw NAV series directly.

    Why not the raw NAV series: mart_portfolio_performance's cash model
    (int_cash_flow.sql) treats every BUY as "externally funded" and
    every SELL's proceeds as cash that sits untouched forever (its own
    documented assumption, correct for avoiding a fake loss on a sale
    day) -- but for a real portfolio with many sells whose proceeds
    were actually reinvested into later purchases (never recorded as
    separate DEPOSIT/WITHDRAWAL transactions, since a Groww order-
    history export doesn't include bank transfers), that tracked cash
    balance only ever grows, never nets back down when it's really
    spent again. Confirmed for real: a real portfolio's tracked cash
    balance reached +23,356 against a real market value of 17,371 --
    s0 (from total_nav_inr) came out at 40,727, more than double the
    real value, and the resulting CAGR/volatility were consequently
    absurd (525% CAGR, 245% annualized volatility) -- Monte Carlo
    percentile bands centered on ~13x the real current value, nowhere
    near plausible. build_extended_portfolio_series instead builds a
    value series purely from each currently-held ticker's own real
    price history and TODAY's real weights/value -- anchored exactly to
    the real market value, with no cash-tracking assumption to distort
    it (the same real data ARIMA already uses successfully).

    analysis_start/analysis_end are accepted for signature
    compatibility with existing callers but no longer used to filter a
    date range here -- the extended series always uses each held
    ticker's own full available real price history.
    """
    from forecasting.arima.series import build_extended_portfolio_series

    series = build_extended_portfolio_series(portfolio_id)
    daily_returns = series.pct_change().dropna()
    sigma = annualized_volatility(daily_returns)

    start_value = float(series.iloc[0])
    end_value = float(series.iloc[-1])
    historical_cagr = cagr(start_value, end_value, series.index[0].date(), series.index[-1].date())

    return PortfolioParams(
        s0=end_value,
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
