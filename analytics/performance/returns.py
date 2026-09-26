"""Portfolio performance metrics: capital/value/return summary, CAGR,
periodic returns, rolling returns, time-weighted return (TWR), and
money-weighted return (XIRR).

All functions are pure: they take pandas Series/DataFrames or plain
numbers and return plain results. No database access here (see
analytics/data_access.py).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def capital_summary(invested_capital: float, market_value: float) -> dict:
    """
    Definitions:
        absolute_return = market_value - invested_capital
        pct_return       = absolute_return / invested_capital

    invested_capital here is the running remaining cost basis of
    currently-held positions (sum of mart_asset_performance.cost_basis_inr),
    NOT total historical capital ever deployed -- shares already sold
    have left the cost-basis pool by design (see int_daily_holdings).
    """
    absolute_return = market_value - invested_capital
    pct_return = absolute_return / invested_capital if invested_capital else float("nan")
    return {
        "invested_capital": invested_capital,
        "market_value": market_value,
        "absolute_return": absolute_return,
        "pct_return": pct_return,
    }


def cagr(start_value: float, end_value: float, start_date: date, end_date: date) -> float:
    """
    Definition:
        CAGR = (end_value / start_value) ** (1 / years) - 1
        years = (end_date - start_date).days / 365.25

    Compound annual growth rate between two NAV points. Undefined (nan)
    if start_value <= 0 or the period is shorter than one day.
    """
    days = (end_date - start_date).days
    if start_value <= 0 or days <= 0:
        return float("nan")
    years = days / 365.25
    return (end_value / start_value) ** (1 / years) - 1


def periodic_returns(daily_returns: pd.Series, freq: str) -> pd.Series:
    """
    Definition:
        period_return = product(1 + r_i for r_i in period) - 1

    Compounds a daily simple-return series (indexed by date) up to a
    coarser frequency. freq follows pandas offset aliases: 'ME' (month
    end), 'YE' (year end), 'W' (week).
    """
    r = daily_returns.fillna(0.0)
    return r.resample(freq).apply(lambda x: (1 + x).prod() - 1)


def rolling_returns(daily_returns: pd.Series, window: int) -> pd.Series:
    """
    Definition:
        rolling_return_t = product(1 + r_i for i in [t-window+1, t]) - 1

    A trailing `window`-day compounded return ending on each day. The
    first (window - 1) entries are NaN (insufficient history).
    """
    r = daily_returns.fillna(0.0)
    return r.rolling(window).apply(lambda x: np.prod(1 + x) - 1, raw=True)


def time_weighted_return(
    nav: pd.Series, external_cash_flows: pd.Series | None = None
) -> float:
    """
    Definition (time-weighted return, geometrically linked):
        For each sub-period between consecutive EXTERNAL cash flows
        (contributions/withdrawals the investor made, NOT internal
        trading activity like buying/selling holdings):
            HPR_i = (V_end_i - V_start_i - CF_i) / V_start_i
        TWR = product(1 + HPR_i) - 1

    TWR isolates investment performance from the size and timing of the
    investor's own contributions/withdrawals -- it answers "how did the
    strategy perform," not "how did this particular investor's
    contributions perform" (that is XIRR's job, below).

    `nav` is a date-indexed Series of total portfolio value. In this
    codebase, internal activity (BUY/SELL/DIVIDEND) is already excluded
    from being treated as an external flow at the dbt layer (BUY is
    zero-net-effect by convention in int_cash_flow; SELL/DIVIDEND are
    real performance, not investor contributions) -- so only genuine
    DEPOSIT/WITHDRAWAL transactions belong in `external_cash_flows`.
    With no such transactions (the current example data has none), TWR
    reduces exactly to the single-period compounded return over `nav`.
    """
    nav = nav.sort_index()
    if external_cash_flows is None or external_cash_flows.empty:
        v0, v1 = nav.iloc[0], nav.iloc[-1]
        if v0 == 0:
            return float("nan")
        return (v1 - v0) / v0

    flow_dates = sorted(external_cash_flows.index)
    boundaries = [nav.index[0]] + flow_dates + [nav.index[-1]]
    boundaries = sorted(set(boundaries))

    twr = 1.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        v_start = nav.loc[start]
        v_end = nav.loc[end]
        cf = external_cash_flows.get(end, 0.0)
        if v_start == 0:
            continue
        hpr = (v_end - v_start - cf) / v_start
        twr *= 1 + hpr
    return twr - 1


def xirr(cash_flows: list[tuple[date, float]]) -> float:
    """
    Definition (money-weighted return / XIRR):
        Find r such that NPV(r) = sum(CF_i / (1 + r) ** (t_i / 365)) = 0
        where t_i is the number of days from the first cash flow to
        cash flow i.

    Unlike TWR, XIRR IS sensitive to the size and timing of cash flows
    -- it answers "what annualized rate of return did this specific
    sequence of contributions and withdrawals actually earn."
    cash_flows is a list of (date, amount) with outflows (money leaving
    the investor's pocket: BUYs, fees) as negative and inflows (money
    returned: SELLs, dividends, and a final terminal liquidation value)
    as positive. Solved numerically via Brent's method.
    """
    if len(cash_flows) < 2:
        return float("nan")
    cash_flows = sorted(cash_flows, key=lambda cf: cf[0])
    t0 = cash_flows[0][0]

    def npv(rate: float) -> float:
        return sum(
            amount / (1 + rate) ** ((d - t0).days / 365.0) for d, amount in cash_flows
        )

    try:
        return brentq(npv, -0.9999, 10)
    except ValueError:
        return float("nan")
