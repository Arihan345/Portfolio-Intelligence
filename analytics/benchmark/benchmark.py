"""Benchmark comparison: portfolio vs. NIFTY 50 (mart_benchmark).

Reuses analytics.risk.beta_alpha for beta/alpha (same CAPM regression,
benchmark = NIFTY 50 here specifically) rather than re-deriving it, to
avoid two implementations of the same formula silently drifting apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.risk.risk import TRADING_DAYS_PER_YEAR, beta_alpha


def excess_return(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """Definition: excess_return_t = r_portfolio_t - r_benchmark_t (per day)."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["p", "b"]
    return aligned["p"] - aligned["b"]


def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Definition:
        tracking_error = std(excess_return) * sqrt(252)

    Annualized standard deviation of the day-to-day return difference
    between portfolio and benchmark -- how much the portfolio deviates
    from the benchmark, regardless of direction.
    """
    er = excess_return(portfolio_returns, benchmark_returns)
    return float(er.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Definition:
        IR = mean(excess_return) * 252 / tracking_error

    Annualized excess return per unit of tracking error: how much
    benchmark-relative return the portfolio earns for the deviation it
    takes on to earn it.
    """
    er = excess_return(portfolio_returns, benchmark_returns)
    te = tracking_error(portfolio_returns, benchmark_returns)
    if te == 0:
        return float("nan")
    return float(er.mean() * TRADING_DAYS_PER_YEAR / te)


def benchmark_comparison(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    portfolio_cumulative_return_override: float | None = None,
) -> dict:
    """Bundles portfolio/benchmark cumulative return, excess return,
    tracking error, beta, alpha, information ratio and correlation into
    one summary dict, all derived from the same aligned return pair.

    portfolio_cumulative_return_override (optional): compounding
    portfolio_returns day-by-day (the default, and still what beta/
    tracking-error/correlation/information-ratio below always use --
    those genuinely need a real daily series) silently assumes those
    returns came only from market moves. For a real portfolio built up
    through many separate BUYs over time rather than one lump-sum
    investment, a day with a large purchase relative to the then-small
    NAV reads as a huge "daily return" in this series -- a real,
    confirmed case: a portfolio that grew via 51 small trades over two
    years showed a single day at +553% (a large relative buy, not a
    553% market move), which compounded into a fabricated ~9,724%
    "cumulative return" here while the portfolio's real total return on
    invested capital was 10.67%. Passing that real, cost-basis total
    return here overrides ONLY the headline cumulative-return figure
    (and excess_cumulative_return, derived from it) with the correct
    number; the underlying statistical series-based numbers keep using
    the real daily series, unchanged.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["p", "b"]

    if portfolio_cumulative_return_override is not None:
        portfolio_cum_return = float(portfolio_cumulative_return_override)
    else:
        portfolio_cum_return = float((1 + aligned["p"]).prod() - 1)
    benchmark_cum_return = float((1 + aligned["b"]).prod() - 1)
    ba = beta_alpha(aligned["p"], aligned["b"], risk_free_rate_annual)

    return {
        "portfolio_cumulative_return": portfolio_cum_return,
        "benchmark_cumulative_return": benchmark_cum_return,
        "excess_cumulative_return": portfolio_cum_return - benchmark_cum_return,
        "tracking_error_annualized": tracking_error(aligned["p"], aligned["b"]),
        "beta": ba["beta"],
        "alpha_annualized": ba["alpha_annualized"],
        "information_ratio": information_ratio(aligned["p"], aligned["b"]),
        "correlation": float(aligned["p"].corr(aligned["b"])),
    }
