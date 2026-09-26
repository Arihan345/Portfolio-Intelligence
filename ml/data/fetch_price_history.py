"""Fetches extended price history for ML training via the existing
MarketDataProvider abstraction (Phase 2) -- no new fetch logic, reused
as-is.

Why this is separate from the warehouse's fact_daily_prices: that
table was loaded (Phase 3) only for Jan-Jun 2024, matching the example
PORTFOLIO's actual transaction window. Reusing it for ML would give a
dangerously short training history and repeat exactly the mistake
Phase 6 surfaced (a short window producing an unstable, misleading
statistic -- there, portfolio CAGR; here, it would be volatility-regime
labels with almost no regime changes to learn from). The ML feature
table is also architecturally required to be a separate lineage from
the analytics marts per this project's frozen Phase 1 decision. So ML
training data is fetched and cached independently, at least 3-5 years
back, without touching the transactional warehouse at all.

Run with: python -m ml.data.fetch_price_history
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ingestion.providers.yfinance_provider import YFinanceProvider

TICKERS = ["TCS.NS", "RELIANCE.NS"]
YEARS_OF_HISTORY = 5
CACHE_PATH = Path(__file__).parent / "price_history_cache.parquet"


def fetch_and_cache(
    tickers: list[str] = TICKERS, years: int = YEARS_OF_HISTORY, end: date | None = None
) -> pd.DataFrame:
    end = end or date.today()
    start = date(end.year - years, end.month, end.day)

    provider = YFinanceProvider()
    result = provider.fetch_ohlcv(tickers, start, end)
    if result.failed_tickers:
        raise RuntimeError(f"failed to fetch tickers: {result.failed_tickers}")

    df = result.data.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df


def load_cached() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{CACHE_PATH} not found -- run `python -m ml.data.fetch_price_history` first"
        )
    return pd.read_parquet(CACHE_PATH)


if __name__ == "__main__":
    df = fetch_and_cache()
    print(f"Fetched {len(df)} rows for {df['ticker'].nunique()} tickers")
    print(df.groupby("ticker")["date"].agg(["min", "max", "count"]))
