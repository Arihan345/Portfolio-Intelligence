"""Assembles the full row-per-(ticker, date) ML dataset: features +
labels, per ticker, concatenated across tickers.
"""
from __future__ import annotations

import pandas as pd

from ml.features.features import build_features, rolling_volatility
from ml.labels.labels import build_labels, historical_vol_threshold
from ml.splits import train_cutoff_date

FEATURE_COLUMNS = [
    "return_lag_1", "return_lag_5", "return_lag_21",
    "volatility_10d", "volatility_21d", "volatility_63d",
    "momentum_10d", "momentum_21d",
    "ma_20d", "ma_50d", "price_to_ma_20d_ratio", "price_to_ma_50d_ratio",
    "volume_zscore_21d",
    "drawdown_depth_60d",
]

DEFAULT_TRAIN_FRAC = 0.70


def build_dataset(price_history: pd.DataFrame, train_frac: float = DEFAULT_TRAIN_FRAC) -> pd.DataFrame:
    """price_history: long-format DataFrame with columns
    [ticker, date, close, volume, ...], one or more tickers.

    train_frac must match the fraction later used by
    ml.splits.chronological_split (default 0.70 for both) -- it decides
    where each ticker's OWN training period ends for the sole purpose
    of computing that ticker's label threshold from training-period
    volatility only (see ml.labels.labels's module docstring for why:
    computing it from the full history, as an earlier version of this
    function did, leaks future test-period volatility into the "high
    volatility" bar used to label every row, including training ones).

    Returns one row per (ticker, date) with all FEATURE_COLUMNS, the
    label, and bookkeeping columns (ticker, date, realized_vol_forward,
    vol_threshold), with warm-up and no-future-data rows dropped (any
    row with a NaN feature or NaN label is unusable for training).
    """
    frames = []
    for ticker, g in price_history.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        features = build_features(g)
        trailing_vol_21d = rolling_volatility(g)["volatility_21d"]

        cutoff = train_cutoff_date(g["date"], train_frac)
        train_mask = g["date"] <= cutoff
        threshold = historical_vol_threshold(trailing_vol_21d[train_mask])

        labels = build_labels(g, threshold)

        merged = features.merge(labels, on="date", suffixes=("", "_label"))
        merged.insert(0, "ticker", ticker)
        frames.append(merged)

    dataset = pd.concat(frames, ignore_index=True)
    required = FEATURE_COLUMNS + ["label"]
    dataset = dataset.dropna(subset=required).reset_index(drop=True)
    dataset["label"] = dataset["label"].astype(int)
    return dataset
