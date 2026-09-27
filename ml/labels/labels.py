"""Volatility-regime label definition.

LABELING RULE (documented before implementation, per this phase's dev
discipline):

    For each (ticker, date t):
        realized_vol_forward(t) = std(daily_return(t+1) .. daily_return(t+21)) * sqrt(252)
        label(t) = 1  if realized_vol_forward(t) > threshold(ticker)
                   0  otherwise
        threshold(ticker) = 75th percentile of that ticker's TRAINING-PERIOD-ONLY
                             trailing 21-day rolling volatility distribution

realized_vol_forward(t) uses ONLY daily returns strictly AFTER date t
(days t+1 through t+21) -- it never includes day t's own return. This
is the forward-looking half of the leakage boundary: features (built
in ml/features) use only data at/before t, labels use only data after
t, and the two windows never overlap for the same row (verified by
ml/tests/test_leakage.py).

THRESHOLD LEAKAGE FIX (previously a bug, now fixed): an earlier version
of this module computed threshold(ticker) as a percentile over the
TICKER'S ENTIRE HISTORY (train+val+test combined), inside build_labels
itself. That meant the "high volatility" bar used to label EVERY row,
including training rows, was informed by volatility observed in the
future test period -- a form of look-ahead in how the label boundary
is set, distinct from (and in addition to) the per-row feature/label
window leakage this module was already checked against. Caught via
Pipeline & Model Monitoring surfacing a suspicious train/test label
rate skew (22% vs 51%) alongside a badly-negative promotion margin.

Investigation (see ml/dataset.py's build_dataset for the real fix, and
the analysis run during that investigation) found the skew was mostly
NOT explained by this leakage bug: TCS.NS's realized volatility itself
genuinely rose in the chronological test window (this dataset's most
recent months) versus its training window -- a real regime shift, not
an artifact. The leakage bug was real and worth fixing on correctness
grounds, but recomputing the threshold from training-only data barely
moved TCS.NS's test label rate (measured at ~77% under the old global
threshold vs ~80% under the corrected training-only threshold) -- if
anything, the corrected calculation is closer to what a model deployed
after only seeing the training period would have actually used.

`build_labels` therefore takes the threshold as an EXPLICIT parameter
now rather than computing it from whatever series happens to be handed
in -- this makes it structurally impossible for a caller to
accidentally pass a leakage-prone full-history series in its place;
the caller must decide up front (and document) what data the threshold
comes from. See ml.dataset.build_dataset for that decision: the
training-period-only slice, using ml.splits.train_cutoff_date so the
threshold's training window matches the same boundary that later
determines the model's actual train/val/test split.

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
    """Percentile of whatever trailing-21d volatility series is passed
    in (NaNs excluded). This function itself is agnostic to WHICH
    period that series covers -- see build_labels's docstring and
    ml.dataset.build_dataset for why the caller must pass only the
    training-period slice, never the full history."""
    return float(np.nanpercentile(trailing_vol_21d.dropna(), percentile))


def build_labels(prices: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Returns a DataFrame (indexed like `prices`) with columns:
    realized_vol_forward, vol_threshold (the given constant, repeated
    on every row for traceability), label (0/1, NaN where forward data
    doesn't exist yet).

    `threshold` must be precomputed by the caller from training-period
    data only (see this module's docstring) -- build_labels no longer
    computes it internally from `prices`, so there is no series this
    function could accidentally leak from.
    """
    fwd_vol = realized_vol_forward(prices)

    out = pd.DataFrame(index=prices.index)
    out["date"] = prices["date"].values
    out["realized_vol_forward"] = fwd_vol
    out["vol_threshold"] = threshold
    out["label"] = (fwd_vol > threshold).astype("Int64")
    out.loc[fwd_vol.isna(), "label"] = pd.NA
    return out
