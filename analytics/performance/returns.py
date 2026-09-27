"""Portfolio performance metrics: capital/value/return summary, CAGR,
periodic returns, rolling returns, time-weighted return (TWR), and
money-weighted return (XIRR).

All functions are pure: they take pandas Series/DataFrames or plain
numbers and return plain results. No database access here (see
analytics/data_access.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def capital_summary(
    invested_capital: float, market_value: float, realized_summary: "RealizedPnlSummary | None" = None
) -> dict:
    """
    Definitions:
        absolute_return = market_value - invested_capital
        pct_return       = absolute_return / invested_capital

    invested_capital here is the running remaining cost basis of
    currently-held positions (sum of mart_asset_performance.cost_basis_inr,
    now correctly a weighted-average remaining basis -- see
    int_daily_holdings.sql), NOT total historical capital ever deployed
    -- shares already sold have left the cost-basis pool by design.

    That default formula is undefined once invested_capital is exactly
    0 -- which is the CORRECT value for a fully-exited portfolio (zero
    current holdings), not a bug to work around. But
    market_value - invested_capital would then also collapse to 0,
    silently discarding a real, historical realized profit/loss that a
    fully-exited portfolio can genuinely have (confirmed for real: a
    synthetic fully-exited portfolio -- bought for 2,165 total, sold
    for 2,690 total -- has a real 525 realized gain even though nothing
    is held today). Passing `realized_summary` (see
    realized_pnl_summary below) makes that real number the headline
    Absolute Return/% Return instead, in exactly this one degenerate
    case; whenever invested_capital > 0 this function's original,
    already-validated-against-real-ground-truth behavior is completely
    unchanged.
    """
    absolute_return = market_value - invested_capital
    pct_return = absolute_return / invested_capital if invested_capital else float("nan")

    if abs(invested_capital) < 1e-9 and realized_summary is not None and realized_summary.total_capital_deployed > 0:
        absolute_return = realized_summary.realized_pnl
        pct_return = realized_summary.realized_pnl / realized_summary.total_capital_deployed

    return {
        "invested_capital": invested_capital,
        "market_value": market_value,
        "absolute_return": absolute_return,
        "pct_return": pct_return,
    }


@dataclass
class RealizedPnlSummary:
    realized_pnl: float
    total_capital_deployed: float


def realized_pnl_summary(transactions: pd.DataFrame) -> RealizedPnlSummary:
    """
    Standard weighted-average-cost inventory accounting, walked
    chronologically per base symbol (ticker with the exchange suffix
    stripped, so a cross-exchange BUY/SELL pair on the same underlying
    company nets correctly -- same convention as int_daily_holdings.sql's
    canonical_asset): a BUY adds its own cost to the running total and
    increases quantity; a SELL realizes (proceeds - cost removed
    proportional to quantity sold) and decreases quantity/cost by that
    same proportion.

    realized_pnl is the sum of every SELL's realized gain/loss across
    the portfolio's full history (independent of whether the position
    is still held today); total_capital_deployed is the sum of every
    BUY's cost ever, across that same history. Used by capital_summary
    as the fallback numerator/denominator for a fully-exited portfolio,
    where dividing by (correctly zero) current invested capital is
    undefined -- see its docstring.
    """
    if transactions.empty:
        return RealizedPnlSummary(0.0, 0.0)

    realized_pnl = 0.0
    total_capital_deployed = 0.0
    base_symbol = transactions["ticker"].str.split(".").str[0]
    for _, group in transactions.assign(base_symbol=base_symbol).sort_values("txn_date").groupby("base_symbol"):
        qty = 0.0
        cost = 0.0
        for _, r in group.iterrows():
            if r["transaction_type"] == "BUY":
                buy_cost = r["price_inr"] * r["quantity"] + r["fees"]
                qty += r["quantity"]
                cost += buy_cost
                total_capital_deployed += buy_cost
            elif r["transaction_type"] == "SELL":
                cost_removed = cost * (r["quantity"] / qty) if qty > 0 else 0.0
                proceeds = r["price_inr"] * r["quantity"] - r["fees"] - r["tax"]
                realized_pnl += proceeds - cost_removed
                cost -= cost_removed
                qty -= r["quantity"]

    return RealizedPnlSummary(realized_pnl=realized_pnl, total_capital_deployed=total_capital_deployed)


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


def daily_returns_from_value_and_flows(
    value: pd.Series, external_cash_flows: pd.Series | None = None
) -> pd.Series:
    """
    Definition (Modified-Dietz-style single-day holding period return):
        r_t = (V_t - V_(t-1) - CF_t) / V_(t-1)

    `value` is a date-indexed Series of portfolio (or holdings) value.
    `external_cash_flows` is a date-indexed Series of net capital moved
    into (positive) or out of (negative) that value on each date -- a
    BUY's cost is a positive flow (new capital converted into shares,
    not gain), a SELL's net proceeds are a negative flow (shares
    leaving the book, not a loss). Without this adjustment, a single
    day with a trade whose SIZE is large relative to the then-small
    portfolio value reads as an enormous fake return -- confirmed for
    real: a portfolio's own mart_portfolio_performance.daily_return
    showed +443% on a day a new BUY was recorded, because that model's
    naive value(t)/value(t-1) ratio has no way to tell "new capital
    arrived" apart from "the market moved." Chain-linking that one
    corrupted day compounds into a multi-hundred-percent CAGR/TWR for
    the whole history even though every other day is normal (see
    time_weighted_return below).

    Returns NaN for any day whose prior value is <= 0 (the series
    hasn't started yet / no meaningful base to divide by).
    """
    value = value.sort_index()
    cf = pd.Series(0.0, index=value.index)
    if external_cash_flows is not None and not external_cash_flows.empty:
        cf = cf.add(external_cash_flows.reindex(value.index, fill_value=0.0), fill_value=0.0)

    prev = value.shift(1)
    r = (value - prev - cf) / prev
    return r.where(prev > 0)


def net_external_cash_flows_by_day(transactions: pd.DataFrame) -> pd.Series:
    """
    Builds the date-indexed net external cash flow used by
    daily_returns_from_value_and_flows/time_weighted_return, from raw
    BUY/SELL transactions.

    BUY: cost (price*quantity + fees) is a positive inflow into the
    holdings value being measured -- new capital committed, not a
    gain on that day.
    SELL: net proceeds (price*quantity - fees - tax) are a negative
    inflow -- shares leaving the holdings book, not a loss on that day.

    DIVIDEND is real investment income (correctly left as part of
    performance, not backed out here) and genuine DEPOSIT/WITHDRAWAL
    transactions don't touch holdings value directly, so neither
    belongs in this series -- only BUY/SELL move the holdings-value
    number this function's caller measures returns against.

    `transactions` needs txn_date/transaction_type/quantity/price_inr/
    fees/tax columns (analytics.data_access.get_transactions's shape).
    """
    if transactions.empty:
        return pd.Series(dtype=float)
    tx = transactions[transactions["transaction_type"].isin(["BUY", "SELL"])]
    if tx.empty:
        return pd.Series(dtype=float)

    flow = tx.apply(
        lambda r: (r["price_inr"] * r["quantity"] + r["fees"])
        if r["transaction_type"] == "BUY"
        else -(r["price_inr"] * r["quantity"] - r["fees"] - r["tax"]),
        axis=1,
    )
    return flow.groupby(tx["txn_date"]).sum()


def implied_market_value_cash_flows_by_day(asset_performance: pd.DataFrame) -> pd.Series:
    """
    Derives the daily net external cash flow from quantity CHANGES,
    valued on the SAME real market-price basis as the holdings-value
    series it will be subtracted against in daily_returns_from_value_
    and_flows -- not the raw transaction's own recorded price, which
    net_external_cash_flows_by_day above uses.

    Why this exists, not just the transaction-price version above: for
    a REAL portfolio, a trade's recorded price and the ticker's real
    market close on that day are approximately the same number (a real
    trade executes near the real market price), so which basis is used
    barely matters. But confirmed for real: a synthetic test portfolio
    used a ticker string ("BETA.NS") that coincidentally has REAL
    yfinance price data of its own -- a real close of roughly INR 705/
    share -- utterly unrelated to that fixture's fictional INR 58/share
    sale price. holdings_value (market_value_inr) is priced from the
    real INR 705/share close; net_external_cash_flows_by_day priced the
    same sale at the fictional INR 58/share. Subtracting a cash flow
    computed in one price universe from a value series computed in a
    completely different one produced a fake -92% single-day "return"
    on the exit day alone (the transaction-price-based flow was ~13x
    too small to offset the real-price-based value drop) -- not a
    tiny-denominator issue, and not a chain-linking bug: the
    compounding formula itself is correct (see
    test_twr_cagr_unaffected_by_tiny_denominator_extreme_period, which
    verifies exactly this transition pattern using a self-consistent
    synthetic value series). The fix is standard Modified-Dietz
    practice: a cash flow must be valued on the SAME basis as the NAV
    it is measured against, so this derives it directly from the
    mart's own quantity_held/market_value_inr instead of ever touching
    the transaction's recorded price.

    For each asset, on a day its quantity_held changes by qty_delta:
        cf = qty_delta * (today's implied real price if still held
             after the trade, else yesterday's implied real price for
             a full exit -- there's no "today" price for a quantity of
             zero to divide by, so the last known real valuation IS the
             exit's cash flow: exiting fully removes exactly yesterday's
             real market value, a clean, zero-price-move assumption for
             that specific day's own trade).
    Summed across assets per day.

    `asset_performance` needs ticker/as_of_date/quantity_held/
    market_value_inr columns (analytics.data_access.get_asset_
    performance's shape, gap-filled or not -- pass whichever version
    of market_value_inr the caller's holdings_value series itself uses,
    so both stay on the same basis).
    """
    if asset_performance.empty:
        return pd.Series(dtype=float)

    ap = asset_performance.sort_values(["ticker", "as_of_date"]).copy()
    # fillna(0), not left as NaN: a ticker's OWN first row in this
    # DataFrame is its real first-ever transaction date (int_daily_
    # holdings only generates a calendar from each asset's own start
    # date onward, never "before it existed") -- shift(1) correctly has
    # no prior row there, but that means "held zero before this,"
    # never "unknown." Left as NaN, qty_delta below would also be NaN
    # on every brand-new position's first day, and NaN.abs() > 1e-9 is
    # always False -- silently treating a real, sometimes-large first
    # BUY as a zero cash flow. Confirmed for real: a portfolio's very
    # first purchase of a new ticker (e.g. a single INR 15,117 DIXON.NS
    # share) got credited as INR 0 cash flow instead of its real cost,
    # making that day's holdings-value jump read as pure, uncosted
    # gain -- inflating TWR from a real ~38% to a nonsensical ~11,659%
    # across the whole portfolio (every ticker's first buy hit this).
    qty_prev = ap.groupby("ticker")["quantity_held"].shift(1).fillna(0.0)
    mv_prev = ap.groupby("ticker")["market_value_inr"].shift(1).fillna(0.0)

    qty_delta = ap["quantity_held"] - qty_prev
    price_now = ap["market_value_inr"] / ap["quantity_held"].where(ap["quantity_held"] > 1e-9)
    price_prev = mv_prev / qty_prev.where(qty_prev > 1e-9)
    implied_price = price_now.where(ap["quantity_held"] > 1e-9, price_prev)

    cf = (qty_delta * implied_price).where(qty_delta.abs() > 1e-9, 0.0).fillna(0.0)
    return cf.groupby(ap["as_of_date"]).sum()


def time_weighted_return(
    nav: pd.Series, external_cash_flows: pd.Series | None = None
) -> float:
    """
    Definition (time-weighted return, geometrically linked):
        For each day t:
            r_t = (V_t - V_(t-1) - CF_t) / V_(t-1)
        TWR = product(1 + r_t) - 1

    TWR isolates investment performance from the size and timing of the
    investor's own contributions/withdrawals -- it answers "how did the
    strategy perform," not "how did this particular investor's
    contributions perform" (that is XIRR's job, below).

    `nav` is a date-indexed Series of portfolio value, chain-linked at
    DAILY granularity (not just between flow dates) so a flow landing
    between two widely-spaced NAV observations still gets attributed
    to the exact day it happened, and a normal day right next to a
    trade day is never contaminated by it. See
    daily_returns_from_value_and_flows's docstring for why this
    per-day adjustment matters -- without it, a single trade whose
    size is large relative to that day's small portfolio value
    compounds into a wildly wrong multi-year TWR. With no external
    cash flows, this reduces exactly to the single-period compounded
    return over `nav`.
    """
    r = daily_returns_from_value_and_flows(nav, external_cash_flows).dropna()
    if r.empty:
        return float("nan")
    return float((1 + r).prod() - 1)


def annualize_return(total_return: float, days: int) -> float:
    """
    Definition:
        (1 + total_return) ** (365.25 / days) - 1

    Annualizes an already-computed holding-period return (e.g. a
    cash-flow-adjusted TWR from time_weighted_return) over `days`
    calendar days. This is NOT interchangeable with cagr() above: cagr()
    takes two raw values and implicitly assumes nothing but market
    movement happened between them, which is false for any real
    portfolio with BUY/SELL activity in between -- annualize_return
    instead takes a return that has ALREADY had contribution/withdrawal
    timing effects removed, so it's the correct way to annualize TWR
    into a real portfolio's CAGR.
    """
    if days <= 0 or total_return <= -1:
        return float("nan")
    years = days / 365.25
    return (1 + total_return) ** (1 / years) - 1


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
