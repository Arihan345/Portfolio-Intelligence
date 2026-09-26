"""Loads the NIFTY 50 index (^NSEI) into the warehouse as a benchmark.

Phase 4's mart_benchmark model joins on dim_asset.asset_type = 'INDEX',
but no such row existed, so the mart correctly built with zero rows
rather than fabricating benchmark numbers. This script closes that gap
for real: fetches ^NSEI OHLCV through the existing MarketDataProvider
abstraction (no new provider code needed -- the abstraction already
supports any ticker) and inserts it as an INDEX-type dim_asset row plus
its fact_daily_prices, over the same Jan-Jun 2024 window as the
portfolio's other price data so the two are comparable day-for-day.

Run with: python -m warehouse.load.load_benchmark
"""
from __future__ import annotations

import uuid
from datetime import date

import sqlalchemy as sa

from ingestion.providers.yfinance_provider import YFinanceProvider
from warehouse.load.load_warehouse import engine, load_fact_daily_prices

BENCHMARK_TICKER = "^NSEI"
SOURCE_SYSTEM = "yfinance"


def upsert_benchmark_asset(as_of: date, run_id: str) -> int:
    with engine.begin() as conn:
        existing = conn.execute(
            sa.text(
                "SELECT asset_key FROM dim_asset WHERE ticker = :ticker AND is_current"
            ),
            {"ticker": BENCHMARK_TICKER},
        ).fetchone()
        if existing:
            return existing.asset_key

        result = conn.execute(
            sa.text(
                """
                INSERT INTO dim_asset
                    (ticker, asset_name, sector, industry, exchange,
                     asset_type, currency_code, effective_from,
                     source_system, pipeline_run_id)
                VALUES
                    (:ticker, 'NIFTY 50', NULL, NULL, 'NSE',
                     'INDEX', 'INR', :as_of, :source_system, :run_id)
                RETURNING asset_key
                """
            ),
            {
                "ticker": BENCHMARK_TICKER,
                "as_of": as_of,
                "source_system": SOURCE_SYSTEM,
                "run_id": run_id,
            },
        )
        return result.scalar_one()


def main() -> None:
    run_id = str(uuid.uuid4())
    print(f"pipeline_run_id = {run_id}")

    asset_key = upsert_benchmark_asset(date(2024, 1, 1), run_id)
    print(f"dim_asset (INDEX) upserted: {BENCHMARK_TICKER} -> asset_key {asset_key}")

    provider = YFinanceProvider()
    ohlcv = provider.fetch_ohlcv([BENCHMARK_TICKER], date(2024, 1, 1), date(2024, 6, 30))
    print(f"OHLCV fetched: {len(ohlcv.data)} rows, failed: {ohlcv.failed_tickers}")

    n = load_fact_daily_prices(ohlcv.data, SOURCE_SYSTEM, run_id)
    print(f"fact_daily_prices rows loaded: {n}")


if __name__ == "__main__":
    main()
