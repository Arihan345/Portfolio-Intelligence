import numpy as np
import pandas as pd
import pytest

from ml.labels.labels import build_labels, historical_vol_threshold, realized_vol_forward


def test_realized_vol_forward_hand_case():
    # 6 constant-return days (10% each) then noise: forward window from
    # row 0 covers returns at rows 1..3 (window=3), all exactly 0.10,
    # so realized_vol_forward(0) with a 3-day window must be exactly 0
    # (zero variance across identical returns) -- a clean, hand-checked
    # degenerate case.
    closes = [100.0, 110.0, 121.0, 133.1, 200.0]
    prices = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5), "close": closes})
    fwd = realized_vol_forward(prices, window=3)
    assert fwd.iloc[0] == pytest.approx(0.0, abs=1e-9)
    # Rows within `window` of the end have no forward data.
    assert np.isnan(fwd.iloc[-1])
    assert np.isnan(fwd.iloc[-2])
    assert np.isnan(fwd.iloc[-3])


def test_historical_vol_threshold_hand_case():
    # 75th percentile of [1, 2, 3, 4] (linear interpolation) = 3.25
    vol_series = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert historical_vol_threshold(vol_series, percentile=75) == pytest.approx(3.25)


def test_build_labels_assigns_1_above_threshold_0_below():
    rng = np.random.default_rng(0)
    n = 100
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    prices = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=n), "close": closes})

    trailing_vol = prices["close"].pct_change().rolling(21).std(ddof=1) * np.sqrt(252)
    labels = build_labels(prices, trailing_vol)

    valid = labels.dropna(subset=["label"])
    threshold = labels["vol_threshold"].iloc[0]
    assert (valid.loc[valid["realized_vol_forward"] > threshold, "label"] == 1).all()
    assert (valid.loc[valid["realized_vol_forward"] <= threshold, "label"] == 0).all()
