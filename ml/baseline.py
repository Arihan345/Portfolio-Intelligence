"""Baseline model: persistence forecast.

Definition: predict that the NEXT 21-day realized-volatility regime is
the SAME as the CURRENT trailing 21-day volatility regime -- i.e.
baseline_prediction(t) = 1  if trailing_vol_21d(t) > threshold(ticker)
                          0  otherwise

This uses only trailing_vol_21d(t), which is itself computed from data
at/before t (see ml/features/features.py's rolling_volatility), so it
is a fair, no-leakage, callable baseline -- a real prediction any
downstream consumer could actually make in real time, not a comment
describing what a baseline would look like.

Any real ML model must beat this to be worth using: it is the "does
nothing clever, just assumes today's regime persists" reference point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_baseline(trailing_vol_21d: pd.Series, threshold: float | pd.Series) -> pd.Series:
    """Returns a 0/1 Series (NaN where trailing_vol_21d itself is NaN,
    i.e. the warm-up period with fewer than 21 days of history).

    `threshold` may be a single float (one ticker) or a per-row Series
    aligned to `trailing_vol_21d`'s index (multiple tickers, each with
    its own threshold -- pandas compares elementwise in that case, so
    each row is checked against its OWN ticker's threshold rather than
    one ticker's threshold being silently applied to every row)."""
    prediction = (trailing_vol_21d > threshold).astype("Int64")
    prediction[trailing_vol_21d.isna()] = pd.NA
    return prediction
