"""Volatility-regime label definition.

LABELING RULE (documented before implementation, per this phase's dev
discipline):

    For each (ticker, date t):
        realized_vol_forward(t) = std(daily_return(t+1) .. daily_return(t+21)) * sqrt(252)
        label(t) = 1  if realized_vol_forward(t) > threshold(ticker)
                   0  otherwise
        threshold(ticker) = 75th percentile of that ticker's full-history
                             trailing 21-day rolling volatility distribution
                             (ml.features.features.rolling_volatility's
                             "volatility_21d" column, computed once over
                             the ticker's entire available history)

realized_vol_forward(t) uses ONLY daily returns strictly AFTER date t
(days t+1 through t+21) -- it never includes day t's own return. This
is the forward-looking half of the leakage boundary: features (built
in ml/features) use only data at/before t, labels use only data after
t, and the two windows never overlap for the same row (verified by
ml/tests/test_leakage.py).

threshold(ticker) is a SINGLE fixed number per ticker, computed once
over the ticker's whole historical rolling-vol distribution -- not a
point-in-time / expanding percentile recomputed at each row. This is a
known, deliberate simplification (documented here rather than left
implicit): it means the "high volatility" bar for a ticker is defined
using its full history including dates that come after some of the
rows being labeled, which is a form of look-ahead in how the BAR ITSELF
is set (though never in which data produces any individual feature or
label value). A fully point-in-time-correct version would use an
expanding percentile of only past rolling-vol observations; that is
noted here as a known limitation, not hidden.

The last 21 rows of each ticker's series have no label (there is no
future data to compute realized_vol_forward from) and are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
FORWARD_WINDOW = 21
HIGH_VOL_PERCENTILE = 75


def realized_vol_forward(prices: pd.DataFrame, window: int = FORWARD_WINDOW) -> pd.Series:
    """realized_vol_forward(t) using daily returns from t+1 to t+window
    (inclusive), computed by shifting the rolling-volatility series
    backward by `window` rows: a trailing `window`-day rolling std
    computed as of row (t + window) covers returns (t+1 .. t+window)
    exactly, so shifting that series up by `window` rows aligns it back
    onto row t."""
    daily_return = prices["close"].pct_change()
    trailing_vol = daily_return.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return trailing_vol.shift(-window)


def historical_vol_threshold(
    trailing_vol_21d: pd.Series, percentile: float = HIGH_VOL_PERCENTILE
) -> float:
    """The fixed per-ticker threshold: percentile of the ticker's full
    trailing-21-day rolling volatility history (NaNs from the warm-up
    period excluded)."""
    return float(np.nanpercentile(trailing_vol_21d.dropna(), percentile))


def build_labels(prices: pd.DataFrame, trailing_vol_21d: pd.Series) -> pd.DataFrame:
    """Returns a DataFrame (indexed like `prices`) with columns:
    realized_vol_forward, vol_threshold (the fixed per-ticker constant,
    repeated on every row for traceability), label (0/1, NaN where
    forward data doesn't exist yet)."""
    fwd_vol = realized_vol_forward(prices)
    threshold = historical_vol_threshold(trailing_vol_21d)

    out = pd.DataFrame(index=prices.index)
    out["date"] = prices["date"].values
    out["realized_vol_forward"] = fwd_vol
    out["vol_threshold"] = threshold
    out["label"] = (fwd_vol > threshold).astype("Int64")
    out.loc[fwd_vol.isna(), "label"] = pd.NA
    return out
