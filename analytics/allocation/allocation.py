"""Allocation breakdowns: what % of the portfolio's value sits in each
asset, sector, industry, currency, or cash, as of a given day.

These are grouping/aggregation operations kept in Python rather than
dbt because mart_allocation already gives per-asset weight; this module
exists to re-aggregate that to coarser dimensions (sector, industry,
currency) on demand, without materializing a separate mart per
dimension for a breakdown that's a two-line groupby.
"""
from __future__ import annotations

import pandas as pd


def _weighted_breakdown(allocation: pd.DataFrame, group_col: str) -> pd.Series:
    """
    Definition:
        weight(group) = sum(market_value_inr for assets in group) / total_value

    `allocation` must already be filtered to a single as_of_date.
    """
    total = allocation["market_value_inr"].sum()
    if total == 0:
        return pd.Series(dtype=float)
    return (
        allocation.groupby(group_col)["market_value_inr"].sum() / total
    ).sort_values(ascending=False)


def asset_allocation(allocation_on_date: pd.DataFrame) -> pd.Series:
    """% of portfolio value per ticker, on the given day."""
    return _weighted_breakdown(allocation_on_date, "ticker")


def sector_allocation(allocation_on_date: pd.DataFrame) -> pd.Series:
    """% of portfolio value per sector, on the given day."""
    return _weighted_breakdown(allocation_on_date, "sector")


def industry_allocation(allocation_on_date: pd.DataFrame) -> pd.Series:
    """% of portfolio value per industry, on the given day."""
    return _weighted_breakdown(allocation_on_date, "industry")


def currency_allocation(allocation_on_date: pd.DataFrame) -> pd.Series:
    """% of portfolio value per currency, on the given day. Requires a
    `currency_code` column (mart_allocation itself doesn't carry
    currency -- the caller joins it in from stg_assets; see
    data_access.get_allocation)."""
    return _weighted_breakdown(allocation_on_date, "currency_code")


def cash_allocation(holdings_market_value: float, cash_balance: float) -> dict:
    """
    Definition:
        total = holdings_market_value + cash_balance (cash_balance can
        be negative in this codebase's convention -- see
        int_cash_flow's header comment -- so this is a signed split,
        not a guaranteed [0,1] percentage pair, unless cash_balance >= 0.
        weight_holdings = holdings_market_value / total
        weight_cash     = cash_balance / total
    """
    total = holdings_market_value + cash_balance
    if total == 0:
        return {"weight_holdings": float("nan"), "weight_cash": float("nan")}
    return {
        "weight_holdings": holdings_market_value / total,
        "weight_cash": cash_balance / total,
    }
