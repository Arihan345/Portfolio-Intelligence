"""Feature engineering: pure functions on a single ticker's price
DataFrame (columns: date, close, volume, sorted ascending by date).

LEAKAGE RULE (see ml/tests/test_leakage.py for the enforced check):
every function here uses only pandas .shift(n) (n >= 0, pulls from n
rows BEFORE the current row) and .rolling(window) with the default
right-aligned, backward-looking window (never center=True, never a
negative shift). Every value at row t is therefore computable using
only data at or before date t -- nothing here ever reads a future row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def lagged_returns(prices: pd.DataFrame, lags: tuple[int, ...] = (1, 5, 21)) -> pd.DataFrame:
    """
    Definition: return_lag_N(t) = close(t) / close(t-N) - 1

    The trailing N-day simple return ending at (and including) date t.
    Uses only close prices at or before t.
    """
    out = pd.DataFrame(index=prices.index)
    for lag in lags:
        out[f"return_lag_{lag}"] = prices["close"] / prices["close"].shift(lag) - 1
    return out


def rolling_volatility(prices: pd.DataFrame, windows: tuple[int, ...] = (10, 21, 63)) -> pd.DataFrame:
    """
    Definition: vol_N(t) = std(daily_return(t-N+1) .. daily_return(t)) * sqrt(252)

    Trailing N-day annualized realized volatility ending at t, computed
    from the same daily-return series used everywhere else in this
    codebase (analytics.risk.annualized_volatility uses the identical
    std*sqrt(252) formula; this is the same math applied to a rolling
    window instead of a fixed one).
    """
    daily_return = prices["close"].pct_change()
    out = pd.DataFrame(index=prices.index)
    for window in windows:
        out[f"volatility_{window}d"] = (
            daily_return.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    return out


def momentum(prices: pd.DataFrame, windows: tuple[int, ...] = (10, 21)) -> pd.DataFrame:
    """
    Definition: momentum_N(t) = close(t) / close(t-N) - 1

    Rate-of-change over N days -- mathematically the same construction
    as a lagged return, but reported as its own named indicator since
    it is conventionally read as a trend-strength signal (medium-term
    10/21-day horizons) rather than a short-horizon lag feature (the
    1-day lag in particular is a distinct, much shorter-horizon signal).
    """
    out = pd.DataFrame(index=prices.index)
    for window in windows:
        out[f"momentum_{window}d"] = prices["close"] / prices["close"].shift(window) - 1
    return out


def moving_averages(prices: pd.DataFrame, windows: tuple[int, ...] = (20, 50)) -> pd.DataFrame:
    """
    Definition:
        MA_N(t)              = mean(close(t-N+1) .. close(t))
        price_to_MA_N_ratio(t) = close(t) / MA_N(t)

    The ratio > 1 means price is trading above its N-day moving
    average (a common trend-following signal).
    """
    out = pd.DataFrame(index=prices.index)
    for window in windows:
        ma = prices["close"].rolling(window).mean()
        out[f"ma_{window}d"] = ma
        out[f"price_to_ma_{window}d_ratio"] = prices["close"] / ma
    return out


def volume_zscore(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Definition: volume_zscore(t) = (volume(t) - rolling_mean(volume, N)) / rolling_std(volume, N)

    How many standard deviations today's volume is from its trailing
    N-day average -- a spike/dry-up signal independent of the raw
    volume level, which varies a lot by ticker.
    """
    rolling_mean = prices["volume"].rolling(window).mean()
    rolling_std = prices["volume"].rolling(window).std(ddof=1)
    out = pd.DataFrame(index=prices.index)
    out[f"volume_zscore_{window}d"] = (prices["volume"] - rolling_mean) / rolling_std
    return out


def drawdown_depth(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Definition: drawdown_depth(t) = (close(t) - rolling_max(close, N)) / rolling_max(close, N)

    How far below its trailing N-day peak the price currently sits (<=
    0; 0 means today IS the N-day peak). Same drawdown definition as
    analytics.risk.max_drawdown, applied to a rolling trailing window
    instead of the whole series.
    """
    rolling_peak = prices["close"].rolling(window).max()
    out = pd.DataFrame(index=prices.index)
    out[f"drawdown_depth_{window}d"] = (prices["close"] - rolling_peak) / rolling_peak
    return out


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Assembles every feature above for one ticker's price history
    (prices must be sorted ascending by date, columns: date, close,
    volume). Returns a DataFrame indexed the same as `prices`, with
    `date` carried through as a column."""
    parts = [
        lagged_returns(prices),
        rolling_volatility(prices),
        momentum(prices),
        moving_averages(prices),
        volume_zscore(prices),
        drawdown_depth(prices),
    ]
    features = pd.concat(parts, axis=1)
    features.insert(0, "date", prices["date"].values)
    return features
