"""End-to-end demo: fetch -> validate -> clean/standardize.

Run with: python -m ingestion.run_pipeline_demo
"""
from __future__ import annotations

import json
from datetime import date

from ingestion.cleaning import clean_and_standardize
from ingestion.portfolio_validation import validate_csv
from ingestion.providers.yfinance_provider import YFinanceProvider

TEST_CSV = "ingestion/tests/test_portfolio.csv"
EXAMPLE_TICKERS = ["TCS.NS", "RELIANCE.NS", "NOTAREALTICKERXYZ"]


def main() -> None:
    print("### STAGE 1: MARKET DATA FETCH ###")
    provider = YFinanceProvider()
    ohlcv = provider.fetch_ohlcv(EXAMPLE_TICKERS, date(2024, 1, 1), date(2024, 1, 10))
    print(f"fetched {len(ohlcv.data)} OHLCV rows for {ohlcv.data['ticker'].nunique()} tickers")
    print(f"failed_tickers: {ohlcv.failed_tickers}")
    print()

    print("### STAGE 2: PORTFOLIO CSV VALIDATION ###")
    vresult = validate_csv(TEST_CSV, today=date(2026, 9, 26))
    print(f"valid_rows: {len(vresult.valid_rows)}, rejected_rows: {len(vresult.rejected_rows)}")
    print("\n-- rejected rows --")
    print(
        vresult.rejected_rows[
            ["date", "ticker", "transaction_type", "rejection_reason"]
        ].to_string(index=False)
    )
    print()

    print("### STAGE 3: CLEANING + STANDARDIZATION ###")
    cleaned, report = clean_and_standardize(vresult.valid_rows)
    print("\n-- cleaned rows --")
    print(
        cleaned[
            [
                "date",
                "ticker",
                "transaction_type",
                "price",
                "currency",
                "price_inr",
                "near_duplicate_flag",
                "price_anomaly_flag",
            ]
        ].to_string(index=False)
    )
    print("\n-- data quality report --")
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
