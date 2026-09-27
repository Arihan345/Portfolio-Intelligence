"""Proves ml.dataset.build_dataset's label threshold is derived ONLY
from training-period volatility -- the fix for the bug where an
earlier version computed it over a ticker's ENTIRE history (see
ml.labels.labels's module docstring for the full incident writeup).

Same perturbation methodology as ml/tests/test_leakage.py: if the
threshold secretly depended on test-period data, violently changing
only the test period's volatility would change the threshold. It must
not.
"""
import numpy as np
import pandas as pd

from ml.dataset import build_dataset

N_ROWS = 400
TRAIN_FRAC = 0.70


def _synthetic_price_history(test_period_shock: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_ROWS)
    sigma = np.full(N_ROWS, 0.01)

    if test_period_shock:
        # Only the test period (the last ~30% of rows, well past the
        # 0.70 train cutoff) gets a large volatility shock -- 20x the
        # daily sigma used everywhere else.
        cutoff = int(N_ROWS * TRAIN_FRAC)
        sigma[cutoff:] = 0.20

    log_returns = rng.normal(0.0, 1.0, N_ROWS) * sigma
    close = 100 * np.exp(np.cumsum(log_returns))
    volume = rng.integers(900_000, 1_100_000, N_ROWS).astype(float)
    return pd.DataFrame({"ticker": "TEST.NS", "date": dates, "close": close, "volume": volume})


def test_threshold_unaffected_by_test_period_volatility_shock():
    baseline_prices = _synthetic_price_history(test_period_shock=False, seed=1)
    shocked_prices = _synthetic_price_history(test_period_shock=True, seed=1)

    # Both series are IDENTICAL up through the train cutoff (same seed,
    # same sigma there) -- only the test-period tail differs.
    cutoff = int(N_ROWS * TRAIN_FRAC)
    assert np.allclose(
        baseline_prices["close"].iloc[:cutoff], shocked_prices["close"].iloc[:cutoff]
    )

    baseline_dataset = build_dataset(baseline_prices, train_frac=TRAIN_FRAC)
    shocked_dataset = build_dataset(shocked_prices, train_frac=TRAIN_FRAC)

    baseline_threshold = baseline_dataset["vol_threshold"].iloc[0]
    shocked_threshold = shocked_dataset["vol_threshold"].iloc[0]

    assert baseline_threshold == shocked_threshold, (
        f"threshold changed ({baseline_threshold} -> {shocked_threshold}) from a shock "
        "confined to the test period -- the threshold is leaking test-period data"
    )


def test_threshold_IS_affected_by_training_period_volatility():
    """Positive control: a volatility shock INSIDE the training period
    must change the threshold -- confirms the threshold is actually
    computed from training data, not a hard-coded constant."""
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=N_ROWS)

    low_vol_sigma = np.full(N_ROWS, 0.01)
    log_returns_low = rng.normal(0.0, 1.0, N_ROWS) * low_vol_sigma
    close_low = 100 * np.exp(np.cumsum(log_returns_low))
    low_prices = pd.DataFrame(
        {"ticker": "TEST.NS", "date": dates, "close": close_low, "volume": rng.integers(900_000, 1_100_000, N_ROWS).astype(float)}
    )

    rng2 = np.random.default_rng(2)
    high_vol_sigma = np.full(N_ROWS, 0.05)  # shock spans the WHOLE series, including training
    log_returns_high = rng2.normal(0.0, 1.0, N_ROWS) * high_vol_sigma
    close_high = 100 * np.exp(np.cumsum(log_returns_high))
    high_prices = pd.DataFrame(
        {"ticker": "TEST.NS", "date": dates, "close": close_high, "volume": rng.integers(900_000, 1_100_000, N_ROWS).astype(float)}
    )

    low_threshold = build_dataset(low_prices, train_frac=TRAIN_FRAC)["vol_threshold"].iloc[0]
    high_threshold = build_dataset(high_prices, train_frac=TRAIN_FRAC)["vol_threshold"].iloc[0]

    assert high_threshold > low_threshold
