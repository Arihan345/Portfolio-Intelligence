"""End-to-end load: Phase 2 pipeline output -> Phase 3 warehouse.

Run with: python -m warehouse.load.run_load
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ingestion.cleaning import clean_and_standardize
from ingestion.portfolio_validation import validate_csv
from ingestion.providers.yfinance_provider import YFinanceProvider
from warehouse.load.load_warehouse import (
    derive_fact_holdings,
    derive_fact_portfolio_returns,
    derive_fact_portfolio_value,
    finish_pipeline_run,
    load_fact_daily_prices,
    load_fact_transactions,
    start_pipeline_run,
    upsert_dim_asset,
)

PORTFOLIO_CSV = "ingestion/tests/test_portfolio.csv"
PORTFOLIO_ID = 1
TICKERS = ["TCS.NS", "RELIANCE.NS"]
SOURCE_SYSTEM = "yfinance"


def main() -> None:
    run_id = start_pipeline_run(dag_id="portfolio_load_demo")
    print(f"pipeline_run_id = {run_id}")
    total_rows = 0

    try:
        # --- dim_asset: seed with basic metadata (sector/industry via
        # fetch_sector_industry would hit yfinance again; kept minimal
        # here since Phase 3's focus is the warehouse, not re-testing
        # the provider abstraction already proven in Phase 2). ---
        asset_meta = {
            "TCS.NS": {"name": "Tata Consultancy Services", "sector": "Technology",
                       "industry": "IT Services", "exchange": "NSE", "currency": "INR"},
            "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy",
                             "industry": "Conglomerate", "exchange": "NSE", "currency": "INR"},
        }
        asset_keys = upsert_dim_asset(asset_meta, date(2024, 1, 1), SOURCE_SYSTEM, run_id)
        print(f"dim_asset upserted: {asset_keys}")

        # --- validate + clean the example portfolio CSV ---
        vresult = validate_csv(PORTFOLIO_CSV, today=date(2026, 9, 26))
        cleaned, dq_report = clean_and_standardize(vresult.valid_rows)
        # test_portfolio.csv also carries Phase 2's INFY.NS dirty-data test
        # rows (missing-currency / anomaly cases); Phase 3's warehouse load
        # is scoped to the worked TCS.NS/RELIANCE.NS example only.
        cleaned = cleaned[cleaned["ticker"].isin(TICKERS)].reset_index(drop=True)
        print(f"cleaned rows ready to load: {len(cleaned)}")

        n = load_fact_transactions(cleaned, PORTFOLIO_ID, SOURCE_SYSTEM, run_id)
        print(f"fact_transactions rows loaded: {n}")
        total_rows += n

        # --- fetch + load real OHLCV for both tickers, Jan-Jun 2024 ---
        provider = YFinanceProvider()
        ohlcv_result = provider.fetch_ohlcv(TICKERS, date(2024, 1, 1), date(2024, 6, 30))
        print(f"OHLCV fetched: {len(ohlcv_result.data)} rows, failed: {ohlcv_result.failed_tickers}")
        n = load_fact_daily_prices(ohlcv_result.data, SOURCE_SYSTEM, run_id)
        print(f"fact_daily_prices rows loaded: {n}")
        total_rows += n

        # --- derive downstream facts ---
        n = derive_fact_holdings(PORTFOLIO_ID, SOURCE_SYSTEM, run_id)
        print(f"fact_holdings rows derived: {n}")
        n = derive_fact_portfolio_value(PORTFOLIO_ID, SOURCE_SYSTEM, run_id)
        print(f"fact_portfolio_value rows derived: {n}")
        n = derive_fact_portfolio_returns(PORTFOLIO_ID, SOURCE_SYSTEM, run_id)
        print(f"fact_portfolio_returns rows derived: {n}")

        finish_pipeline_run(run_id, "SUCCESS", total_rows)
        print("pipeline run SUCCESS")
    except Exception:
        finish_pipeline_run(run_id, "FAILED", total_rows)
        raise


if __name__ == "__main__":
    main()
