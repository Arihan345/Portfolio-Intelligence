import math
from datetime import date

import pandas as pd
import pytest

from analytics.performance.returns import (
    cagr,
    capital_summary,
    periodic_returns,
    rolling_returns,
    time_weighted_return,
    xirr,
)


def test_capital_summary_hand_case():
    result = capital_summary(invested_capital=1000.0, market_value=1200.0)
    assert result["absolute_return"] == 200.0
    assert result["pct_return"] == pytest.approx(0.2)


def test_cagr_hand_case():
    # 730 calendar days (2022 and 2023 are both non-leap) is 730/365.25
    # years, not exactly 2 -- compute the expected value the same way
    # by hand rather than guessing a rounded literal.
    start, end = date(2022, 1, 1), date(2024, 1, 1)
    days = (end - start).days
    assert days == 730
    expected = (121.0 / 100.0) ** (1 / (days / 365.25)) - 1
    result = cagr(100.0, 121.0, start, end)
    assert result == pytest.approx(expected)


def test_cagr_exact_four_years():
    # 2020-01-01 to 2024-01-01 spans exactly 1461 days (2020 and 2024
    # are leap years but only 2020 falls inside the range), and
    # 1461 / 365.25 = 4.0 exactly -- the one calendar span short enough
    # to reason about by hand that divides evenly into the 365.25
    # days-per-year convention. 100 growing at 10%/year for 4 years
    # compounds to 100 * 1.1**4 = 146.41.
    start, end = date(2020, 1, 1), date(2024, 1, 1)
    days = (end - start).days
    assert days == 1461
    assert days / 365.25 == 4.0
    result = cagr(100.0, 146.41, start, end)
    assert result == pytest.approx(0.10, abs=1e-9)


def test_periodic_returns_monthly_compounding():
    # Two days of +10% each within the same month must compound to
    # 1.1 * 1.1 - 1 = 0.21, not 0.20 (simple sum).
    idx = pd.to_datetime(["2024-01-05", "2024-01-10"])
    r = pd.Series([0.10, 0.10], index=idx)
    monthly = periodic_returns(r, freq="ME")
    assert monthly.iloc[0] == pytest.approx(0.21)


def test_rolling_returns_hand_case():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    r = pd.Series([0.10, 0.10, 0.10], index=idx)
    rolling = rolling_returns(r, window=2)
    assert math.isnan(rolling.iloc[0])
    assert rolling.iloc[1] == pytest.approx(0.21)
    assert rolling.iloc[2] == pytest.approx(0.21)


def test_twr_no_external_flows_equals_simple_return():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    nav = pd.Series([100.0, 110.0, 121.0], index=idx)
    assert time_weighted_return(nav) == pytest.approx(0.21)


def test_twr_removes_deposit_distortion():
    # Day0: NAV 100. Day1: a $100 deposit arrives AND the market is
    # flat, so NAV becomes 200 purely from the deposit, not performance.
    # Day2: market grows 10% on the (now) 200 NAV -> NAV 220.
    # Naive return (200->220 ignoring the deposit) would look like a
    # tiny gain from 100->220 (=120%), which is wrong -- that's the
    # deposit, not performance. TWR must show a much smaller number:
    # HPR1 = (200-100-100)/100 = 0%, HPR2 = (220-200)/200 = 10%
    # TWR = 1.00 * 1.10 - 1 = 10%
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    nav = pd.Series([100.0, 200.0, 220.0], index=idx)
    flows = pd.Series([100.0], index=[idx[1]])
    result = time_weighted_return(nav, flows)
    assert result == pytest.approx(0.10)


def test_xirr_hand_case_single_period():
    # Invest 1000 today, receive exactly 1100 in exactly 365 days.
    # XIRR must be 10%: -1000 + 1100/(1.10)**(365/365) = 0.
    cash_flows = [(date(2024, 1, 1), -1000.0), (date(2025, 1, 1), 1100.0)]
    result = xirr(cash_flows)
    assert result == pytest.approx(0.10, abs=1e-3)
