"""Data access layer: pulls DataFrames from the dbt marts.

This is the ONLY module in the analytics package that talks to the
database. Every function in performance/, allocation/, risk/,
benchmark/, attribution/ takes plain pandas/numpy inputs -- it is the
caller's job (typically a script like run_analytics_demo.py) to fetch
data here and hand it to those pure functions, so the formulas stay
testable without a live database.
"""
from __future__ import annotations

import pandas as pd
import sqlalchemy as sa

DB_URL = "postgresql+psycopg://arihan@localhost:5432/portfolio_analytics"

engine = sa.create_engine(DB_URL)


def get_portfolio_performance(portfolio_id: int) -> pd.DataFrame:
    """mart_portfolio_performance: one row per day (value_date, ordered)."""
    return pd.read_sql(
        sa.text(
            """
            select value_date, holdings_market_value_inr, cash_balance_inr,
                   total_nav_inr, daily_return, cumulative_return
            from public_marts.mart_portfolio_performance
            where portfolio_id = :pid
            order by value_date
            """
        ),
        engine,
        params={"pid": portfolio_id},
        parse_dates=["value_date"],
    )


def get_asset_performance(portfolio_id: int) -> pd.DataFrame:
    """mart_asset_performance: one row per (asset, day)."""
    return pd.read_sql(
        sa.text(
            """
            select asset_key, ticker, sector, industry, as_of_date,
                   quantity_held, cost_basis_inr, last_close_inr,
                   market_value_inr, unrealized_gain_inr
            from public_marts.mart_asset_performance
            where portfolio_id = :pid
            order by as_of_date, ticker
            """
        ),
        engine,
        params={"pid": portfolio_id},
        parse_dates=["as_of_date"],
    )


def get_allocation(portfolio_id: int) -> pd.DataFrame:
    """mart_allocation joined to stg_assets.currency_code (the mart
    itself doesn't carry currency -- see allocation.currency_allocation)."""
    return pd.read_sql(
        sa.text(
            """
            select m.asset_key, m.ticker, m.sector, m.industry, m.as_of_date,
                   m.market_value_inr, m.portfolio_total_value_inr,
                   m.weight_in_portfolio, a.currency_code
            from public_marts.mart_allocation m
            join public_staging.stg_assets a
                on a.asset_key = m.asset_key
               and m.as_of_date between a.effective_from and a.effective_to
            where m.portfolio_id = :pid
            order by m.as_of_date, m.ticker
            """
        ),
        engine,
        params={"pid": portfolio_id},
        parse_dates=["as_of_date"],
    )


def get_benchmark(portfolio_id: int) -> pd.DataFrame:
    """mart_benchmark: one row per day comparing portfolio vs. NIFTY 50."""
    return pd.read_sql(
        sa.text(
            """
            select value_date, portfolio_daily_return, benchmark_close,
                   benchmark_daily_return
            from public_marts.mart_benchmark
            where portfolio_id = :pid
            order by value_date
            """
        ),
        engine,
        params={"pid": portfolio_id},
        parse_dates=["value_date"],
    )


def get_price_history(tickers: list[str]) -> pd.DataFrame:
    """fact_daily_prices for the given tickers, across their full real
    warehouse history -- prices are asset-level, not portfolio-scoped,
    so this isn't parameterized by portfolio_id.

    Used by forecasting/arima/series.py to build ARIMA's extended value
    series from REAL price data already in the warehouse for whatever
    tickers a real upload actually holds, instead of a separate, fixed
    ml.data.fetch_price_history cache that was only ever built for the
    ML model's own training universe (TCS.NS/RELIANCE.NS) and had no
    real data at all for any other real portfolio's tickers -- the
    exact real cause of a confirmed 422 ("missing tickers") on every
    real upload's Forecasting/ARIMA and ML Insights pages. The
    warehouse's own price history already covers every currently-held
    ticker back to its first real transaction (fetched during upload,
    the same mechanism Monte Carlo's baseline/asset-level params
    already use successfully) -- unifying on it here means ARIMA never
    depends on a second, separately-populated cache again.
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "date", "close"])
    return pd.read_sql(
        sa.text(
            """
            select distinct on (da.ticker, dd.full_date)
                da.ticker, dd.full_date as date, fp.close
            from fact_daily_prices fp
            join dim_asset da on da.asset_key = fp.asset_key
            join dim_date dd on dd.date_key = fp.date_key
            where da.ticker in :tickers
            order by da.ticker, dd.full_date, da.is_current desc, fp.asset_key asc
            """
        ).bindparams(sa.bindparam("tickers", expanding=True)),
        engine,
        params={"tickers": tickers},
        parse_dates=["date"],
    )


def get_transactions(portfolio_id: int) -> pd.DataFrame:
    """stg_transactions: raw transaction cash flows, for XIRR."""
    return pd.read_sql(
        sa.text(
            """
            select txn_date, transaction_type, quantity, price_inr, fees, tax, ticker
            from public_staging.stg_transactions t
            join public_staging.stg_assets a on a.asset_key = t.asset_key
            where t.portfolio_id = :pid
            order by txn_date
            """
        ),
        engine,
        params={"pid": portfolio_id},
        parse_dates=["txn_date"],
    )
