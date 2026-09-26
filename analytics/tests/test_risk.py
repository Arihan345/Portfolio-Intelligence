import statistics

import numpy as np
import pandas as pd
import pytest

from analytics.risk.risk import (
    TRADING_DAYS_PER_YEAR,
    conditional_var,
    historical_var,
    max_drawdown,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)


def test_sharpe_ratio_hand_case():
    # Independently computed via the stdlib `statistics` module (a
    # different code path than analytics.risk's numpy/pandas math) to
    # verify the formula, not just that the function runs.
    returns = [0.02, 0.01, -0.01]
    rf_annual = 0.05
    mean = statistics.mean(returns)
    stdev = statistics.stdev(returns)  # sample stdev, ddof=1 by default
    expected = (mean * TRADING_DAYS_PER_YEAR - rf_annual) / (stdev * (TRADING_DAYS_PER_YEAR ** 0.5))

    result = sharpe_ratio(pd.Series(returns), risk_free_rate_annual=rf_annual)
    assert result == pytest.approx(expected)


def test_sortino_ratio_only_penalizes_losses():
    # All-positive returns: downside deviation from mar=0 is zero
    # (nothing below the minimum acceptable return), so Sortino must be
    # nan (division by zero), matching the function's documented
    # behavior rather than silently returning inf.
    returns = pd.Series([0.01, 0.02, 0.015])
    result = sortino_ratio(returns, risk_free_rate_annual=0.0)
    assert pd.isna(result)


def test_max_drawdown_hand_case():
    # NAV path: 100 -> 120 (new peak) -> 90 (trough, -25% from 120)
    # -> 110 -> 130 (recovers past 120 on the last day).
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    nav = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0], index=idx)
    result = max_drawdown(nav)
    assert result["max_drawdown"] == pytest.approx(-0.25)
    assert result["peak_date"] == idx[1]
    assert result["trough_date"] == idx[2]
    assert result["recovery_date"] == idx[4]
    assert result["drawdown_duration_days"] == (idx[4] - idx[1]).days


def test_var_and_cvar_hand_case():
    # Chosen so the 25th percentile (confidence=0.75) lands exactly on
    # a data point under linear interpolation: n=5, position =
    # (n-1)*0.25 = 1 -> the 2nd-smallest value, no interpolation
    # ambiguity to reason through by hand.
    returns = pd.Series([-0.05, -0.03, -0.01, 0.02, 0.04])
    var_75 = historical_var(returns, confidence=0.75)
    assert var_75 == pytest.approx(0.03)  # -(-0.03)

    cvar_75 = conditional_var(returns, confidence=0.75)
    # tail = values <= -0.03 -> [-0.05, -0.03], mean = -0.04
    assert cvar_75 == pytest.approx(0.04)

    # CVaR must never be smaller than VaR at the same confidence: it
    # looks past the cutoff into the tail rather than stopping at it.
    assert cvar_75 >= var_75


def test_parametric_var_matches_normal_formula():
    from scipy.stats import norm

    returns = pd.Series([-0.05, -0.03, -0.01, 0.02, 0.04])
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(0.25)
    expected = -(mu + z * sigma)

    result = parametric_var(returns, confidence=0.75)
    assert result == pytest.approx(expected)
