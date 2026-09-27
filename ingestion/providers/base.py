"""Market data provider abstraction.

Hard rule: no module outside this package may import a concrete provider
library (yfinance, etc.) directly. All access to market data goes through
the MarketDataProvider interface so the backing provider can be swapped
with a one-line change (see providers/yfinance_provider.py for the only
place yfinance is imported).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class OHLCVResult:
    """Result of an OHLCV fetch across one or more tickers.

    data: long-format DataFrame with columns
        [ticker, date, open, high, low, close, volume, source_system]
        for every ticker that succeeded.
    failed_tickers: tickers that could not be fetched, each paired with
        the reason, so ingestion can continue for the rest instead of
        aborting the whole batch.
    """

    data: pd.DataFrame
    failed_tickers: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SectorIndustry:
    ticker: str
    sector: str | None
    industry: str | None


@dataclass
class SectorIndustryResult:
    data: dict[str, SectorIndustry]
    failed_tickers: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SplitEvent:
    """A single stock-split / bonus-issue event. ratio follows the
    yfinance convention: 10.0 means a 1-for-10 forward split (1 old
    share becomes 10 new shares) -- quantity multiplies by ratio, price
    divides by ratio. A ratio < 1 (e.g. 0.5) would be a reverse split;
    the same arithmetic still applies."""

    ticker: str
    split_date: date
    ratio: float


@dataclass
class CorporateActionsResult:
    data: dict[str, list[SplitEvent]]
    failed_tickers: list[tuple[str, str]] = field(default_factory=list)


class MarketDataProvider(ABC):
    """Abstract interface for any market data source."""

    @abstractmethod
    def fetch_ohlcv(
        self, tickers: list[str], start: date, end: date
    ) -> OHLCVResult:
        """Fetch daily OHLCV bars for `tickers` between start and end
        (inclusive). Must never raise on a single bad ticker: partial
        failures are recorded in the result's failed_tickers list and
        the rest of the batch still returns data.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_sector_industry(
        self, tickers: list[str]
    ) -> SectorIndustryResult:
        """Fetch sector/industry classification for `tickers`. Same
        partial-failure contract as fetch_ohlcv.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_corporate_actions(
        self, tickers: list[str], start: date, end: date
    ) -> CorporateActionsResult:
        """Fetch stock-split/bonus-issue events for `tickers` with an ex-
        date between start and end (inclusive). Same partial-failure
        contract as fetch_ohlcv/fetch_sector_industry: a bad or unknown
        ticker is recorded in failed_tickers, never raised, so the rest
        of the batch still returns data. A ticker with no split history
        in range gets an empty list, not an entry in failed_tickers.
        """
        raise NotImplementedError
