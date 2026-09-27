"""Concrete MarketDataProvider backed by yfinance.

This is the ONLY module in the codebase allowed to `import yfinance`.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from ingestion.providers.base import (
    CorporateActionsResult,
    MarketDataProvider,
    OHLCVResult,
    SectorIndustry,
    SectorIndustryResult,
    SplitEvent,
)

SOURCE_SYSTEM = "yfinance"


class YFinanceProvider(MarketDataProvider):
    def fetch_ohlcv(
        self, tickers: list[str], start: date, end: date
    ) -> OHLCVResult:
        rows: list[pd.DataFrame] = []
        failed: list[tuple[str, str]] = []

        # yfinance's `end` is exclusive; the caller's `end` is inclusive.
        yf_end = end + timedelta(days=1)

        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(
                    start=start.isoformat(), end=yf_end.isoformat()
                )
            except Exception as exc:  # noqa: BLE001 - must never propagate
                failed.append((ticker, f"fetch error: {exc}"))
                continue

            if hist is None or hist.empty:
                failed.append((ticker, "no data returned for date range"))
                continue

            hist = hist.reset_index()
            hist["ticker"] = ticker
            hist["source_system"] = SOURCE_SYSTEM
            hist = hist.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            hist["date"] = pd.to_datetime(hist["date"]).dt.date
            rows.append(
                hist[
                    [
                        "ticker",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "source_system",
                    ]
                ]
            )

        data = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                columns=[
                    "ticker",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "source_system",
                ]
            )
        )
        return OHLCVResult(data=data, failed_tickers=failed)

    def fetch_sector_industry(
        self, tickers: list[str]
    ) -> SectorIndustryResult:
        result: dict[str, SectorIndustry] = {}
        failed: list[tuple[str, str]] = []

        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).get_info()
            except Exception as exc:  # noqa: BLE001 - must never propagate
                failed.append((ticker, f"fetch error: {exc}"))
                continue

            if not info:
                failed.append((ticker, "no info returned"))
                continue

            result[ticker] = SectorIndustry(
                ticker=ticker,
                sector=info.get("sector"),
                industry=info.get("industry"),
            )

        return SectorIndustryResult(data=result, failed_tickers=failed)

    def fetch_corporate_actions(
        self, tickers: list[str], start: date, end: date
    ) -> CorporateActionsResult:
        result: dict[str, list[SplitEvent]] = {}
        failed: list[tuple[str, str]] = []

        for ticker in tickers:
            try:
                splits = yf.Ticker(ticker).splits
            except Exception as exc:  # noqa: BLE001 - must never propagate
                failed.append((ticker, f"fetch error: {exc}"))
                continue

            events: list[SplitEvent] = []
            if splits is not None:
                for ts, ratio in splits.items():
                    split_date = ts.date() if hasattr(ts, "date") else ts
                    if start <= split_date <= end:
                        events.append(SplitEvent(ticker=ticker, split_date=split_date, ratio=float(ratio)))
            result[ticker] = events

        return CorporateActionsResult(data=result, failed_tickers=failed)
