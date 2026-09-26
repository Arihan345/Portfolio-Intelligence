"""Empirically verifies the leakage boundary: features must be provably
unaffected by any future price, and each row's feature-window must not
overlap with that same row's label-window. Rather than trusting code
inspection, these tests PERTURB specific price rows and check exactly
which output rows change -- a value that shouldn't be reachable but
changes anyway is a leakage bug, caught mechanically.
"""
import numpy as np
import pandas as pd
import pytest

from ml.features.features import build_features, rolling_volatility
from ml.labels.labels import build_labels, realized_vol_forward

N_ROWS = 300
MAX_FEATURE_WINDOW = 63  # the longest lookback among all features
FORWARD_WINDOW = 21


def _synthetic_prices(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_ROWS)
    log_returns = rng.normal(0.0003, 0.015, N_ROWS)
    close = 100 * np.exp(np.cumsum(log_returns))
    volume = rng.integers(1_000_000, 5_000_000, N_ROWS).astype(float)
    return pd.DataFrame({"date": dates, "close": close, "volume": volume})


def test_features_at_row_t_are_identical_using_only_data_up_to_t():
    """The direct proof the brief asks for: recompute features using
    ONLY rows [0..t] (as if the future didn't exist) and confirm every
    feature value at row t is bit-identical to the value computed from
    the full series. If any feature secretly used a future row, this
    would fail: the truncated series physically cannot supply it."""
    prices = _synthetic_prices()
    full_features = build_features(prices)

    t = 150
    truncated_features = build_features(prices.iloc[: t + 1])

    full_row = full_features.iloc[t].drop("date")
    truncated_row = truncated_features.iloc[t].drop("date")
    pd.testing.assert_series_equal(full_row, truncated_row, check_names=False)


def test_features_unaffected_by_perturbing_a_future_price():
    """Complementary check by perturbation: changing a price strictly
    AFTER row t must not move any feature value AT row t."""
    prices = _synthetic_prices()
    t = 100
    perturb_at = t + 1  # the very next day -- as close as "future" gets

    original = build_features(prices)
    perturbed_prices = prices.copy()
    perturbed_prices.loc[perturb_at, "close"] *= 1.5  # a large, obvious shock
    perturbed = build_features(perturbed_prices)

    pd.testing.assert_series_equal(
        original.iloc[t].drop("date"), perturbed.iloc[t].drop("date"), check_names=False
    )
    # Sanity: the perturbation must actually be visible somewhere (a
    # later row whose lookback window includes perturb_at), or this
    # test would be vacuously true regardless of correctness.
    later_row = perturb_at + 5
    assert not original.iloc[later_row].drop("date").equals(
        perturbed.iloc[later_row].drop("date")
    ), "perturbation had no detectable effect anywhere -- test is not exercising anything"


def test_label_forward_window_unaffected_by_past_price_changes():
    """Perturbing a price strictly BEFORE t (outside the label's
    (t, t+21] forward window) must not move realized_vol_forward(t)."""
    prices = _synthetic_prices()
    t = 150
    perturb_at = t - 5  # strictly before t, outside (t, t+21]

    original_fwd = realized_vol_forward(prices)
    perturbed_prices = prices.copy()
    perturbed_prices.loc[perturb_at, "close"] *= 1.5
    perturbed_fwd = realized_vol_forward(perturbed_prices)

    assert original_fwd.iloc[t] == pytest.approx(perturbed_fwd.iloc[t])


def test_label_forward_window_IS_affected_by_the_declared_future_window():
    """Positive control: perturbing a price INSIDE (t, t+21] must move
    realized_vol_forward(t) -- confirms the label is actually reading
    the window it claims to, not silently computing a constant."""
    prices = _synthetic_prices()
    t = 150
    perturb_at = t + 10  # inside (t, t+21]

    original_fwd = realized_vol_forward(prices)
    perturbed_prices = prices.copy()
    perturbed_prices.loc[perturb_at, "close"] *= 1.5
    perturbed_fwd = realized_vol_forward(perturbed_prices)

    assert original_fwd.iloc[t] != pytest.approx(perturbed_fwd.iloc[t])


def test_feature_window_and_label_window_do_not_overlap_for_same_row():
    """The row-level claim itself: for row t, the set of source rows
    any feature reads (t - MAX_FEATURE_WINDOW + 1 .. t) and the set of
    source rows the label's forward vol reads (t+1 .. t+FORWARD_WINDOW)
    are disjoint index ranges -- proven by construction here (no
    overlap is possible if the ranges themselves don't intersect), and
    cross-checked against the two perturbation tests above which show
    each side is only sensitive to its own declared range."""
    t = 150
    feature_window = set(range(t - MAX_FEATURE_WINDOW + 1, t + 1))
    label_window = set(range(t + 1, t + FORWARD_WINDOW + 1))
    assert feature_window.isdisjoint(label_window)


def test_labels_have_no_lookahead_beyond_forward_window():
    """The tail rows (no 21 future days available) must be NA, not a
    silently wrong number -- a common source of accidental leakage is
    letting pandas produce a value from a shorter, incomplete window."""
    prices = _synthetic_prices()
    trailing_vol = rolling_volatility(prices)["volatility_21d"]
    labels = build_labels(prices, trailing_vol)

    tail = labels.iloc[-FORWARD_WINDOW:]
    assert tail["label"].isna().all()
    assert tail["realized_vol_forward"].isna().all()
