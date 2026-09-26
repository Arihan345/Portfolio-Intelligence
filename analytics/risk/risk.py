"""Risk metrics: volatility, downside volatility, Sharpe/Sortino ratio,
beta/alpha vs. a benchmark, max drawdown and its duration, VaR/CVaR,
and a correlation matrix across held assets + benchmark.

Risk-free rate and confidence level are always explicit parameters
(never hard-coded) -- see analytics/__init__.py's dbt-vs-Python
boundary note for why: a single materialized SQL table cannot serve
every caller's choice of risk-free rate or VaR confidence level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_volatility(daily_returns: pd.Series) -> float:
    """
    Definition:
        sigma_annual = std(daily_returns) * sqrt(252)

    Sample standard deviation (ddof=1) of daily simple returns, scaled
    to an annual figure under the standard random-walk assumption.
    """
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def downside_volatility(daily_returns: pd.Series, mar: float = 0.0) -> float:
    """
    Definition:
        downside_dev = sqrt(mean(min(r_i - mar, 0) ** 2)) * sqrt(252)

    Only penalizes returns below the minimum acceptable return `mar`
    (default 0 = any loss). This is the denominator of the Sortino
    ratio, distinguishing "bad" (downside) variance from ordinary
    two-sided volatility.
    """
    downside = np.minimum(daily_returns - mar, 0)
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate_annual: float) -> float:
    """
    Definition:
        Sharpe = (annualized_mean_return - risk_free_rate_annual) / annualized_volatility
        annualized_mean_return = mean(daily_returns) * 252

    risk_free_rate_annual is a required parameter (e.g. the current
    India 91-day T-Bill or 10Y G-Sec yield as a decimal, e.g. 0.07 for
    7%) -- never hard-coded, since the "correct" risk-free rate changes
    over time and by jurisdiction.
    """
    annual_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    vol = annualized_volatility(daily_returns)
    if vol == 0:
        return float("nan")
    return float((annual_return - risk_free_rate_annual) / vol)


def sortino_ratio(
    daily_returns: pd.Series, risk_free_rate_annual: float, mar: float = 0.0
) -> float:
    """
    Definition:
        Sortino = (annualized_mean_return - risk_free_rate_annual) / downside_volatility

    Same numerator as Sharpe, but penalizes only downside deviation
    below `mar`, so a strategy with large upside swings and no losses
    scores better on Sortino than on Sharpe.
    """
    annual_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    dvol = downside_volatility(daily_returns, mar)
    if dvol == 0:
        return float("nan")
    return float((annual_return - risk_free_rate_annual) / dvol)


def beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
) -> dict:
    """
    Definition (CAPM single-factor regression):
        beta  = cov(r_p, r_b) / var(r_b)
        alpha = mean(r_p) - beta * mean(r_b)          (per-period alpha)
        alpha_annualized = alpha * 252

    beta measures sensitivity to benchmark moves (beta=1: moves with
    the market; beta>1: amplifies it). alpha is the return unexplained
    by that benchmark exposure, i.e. Jensen's alpha when a risk-free
    rate is supplied: alpha = mean(r_p - rf) - beta * mean(r_b - rf).
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["p", "b"]
    if len(aligned) < 2:
        return {"beta": float("nan"), "alpha_annualized": float("nan")}

    rf_daily = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    excess_p = aligned["p"] - rf_daily
    excess_b = aligned["b"] - rf_daily

    cov = np.cov(excess_p, excess_b, ddof=1)[0, 1]
    var_b = np.var(excess_b, ddof=1)
    if var_b == 0:
        return {"beta": float("nan"), "alpha_annualized": float("nan")}

    beta = cov / var_b
    alpha_daily = excess_p.mean() - beta * excess_b.mean()
    return {"beta": float(beta), "alpha_annualized": float(alpha_daily * TRADING_DAYS_PER_YEAR)}


def max_drawdown(nav: pd.Series) -> dict:
    """
    Definition:
        running_peak_t = max(nav_0 .. nav_t)
        drawdown_t     = (nav_t - running_peak_t) / running_peak_t
        max_drawdown   = min(drawdown_t) over the whole series

    drawdown_duration_days is the length of the longest stretch, in
    calendar days between the series' own index values, from a peak to
    its subsequent recovery back above that peak (or to the end of the
    series if never recovered).
    """
    nav = nav.sort_index()
    running_peak = nav.cummax()
    drawdown = (nav - running_peak) / running_peak
    mdd = float(drawdown.min())

    trough_idx = drawdown.idxmin()
    peak_value = running_peak.loc[trough_idx]
    # The peak is the most recent date at/before the trough where NAV
    # itself set that running peak (i.e. drawdown was exactly 0 there).
    peak_candidates = nav.loc[:trough_idx][nav.loc[:trough_idx] == peak_value]
    peak_idx = peak_candidates.index[-1]

    recovery = nav[(nav.index > trough_idx) & (nav >= peak_value)]
    end_idx = recovery.index[0] if not recovery.empty else nav.index[-1]

    duration_days = (end_idx - peak_idx).days
    return {"max_drawdown": mdd, "peak_date": peak_idx, "trough_date": trough_idx,
            "recovery_date": recovery.index[0] if not recovery.empty else None,
            "drawdown_duration_days": duration_days}


def historical_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Definition:
        VaR_hist(c) = -percentile(daily_returns, 100 * (1 - c))

    The loss (positive number) such that (1-c) of historical daily
    returns were worse than -VaR. Purely empirical: no distributional
    assumption, but sensitive to the specific historical sample.
    """
    return float(-np.percentile(daily_returns.dropna(), 100 * (1 - confidence)))


def parametric_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Definition (Gaussian / variance-covariance VaR):
        VaR_param(c) = -(mu + z_c * sigma)
        z_c = inverse CDF of the standard normal at (1 - c)

    Assumes daily returns are normally distributed with mean mu and
    std sigma; z_c is negative for c > 0.5 (e.g. z_0.95 ~= -1.645), so
    -(mu + z_c*sigma) is a positive loss figure for typical inputs.
    """
    from scipy.stats import norm

    mu = daily_returns.mean()
    sigma = daily_returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def conditional_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Definition (historical CVaR / Expected Shortfall):
        CVaR_hist(c) = -mean(r_i for r_i <= percentile(r, 100*(1-c)))

    The average loss in the worst (1-c) fraction of historical days --
    always at least as large as historical VaR at the same confidence,
    since it looks past the cutoff into the tail rather than stopping
    at it.
    """
    threshold = np.percentile(daily_returns.dropna(), 100 * (1 - confidence))
    tail = daily_returns[daily_returns <= threshold]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def correlation_matrix(returns_by_asset: pd.DataFrame) -> pd.DataFrame:
    """
    Definition:
        corr[i,j] = cov(r_i, r_j) / (std(r_i) * std(r_j))

    `returns_by_asset` is a wide DataFrame: one column per asset (plus
    optionally the benchmark), one row per date, of daily returns.
    Pandas' pairwise-complete-observations correlation is used, so
    assets with different available date ranges still get a value from
    their overlapping days.
    """
    return returns_by_asset.corr()
