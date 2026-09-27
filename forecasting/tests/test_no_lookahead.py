"""Explicit proof that walk_forward_backtest never fits on data from
after its own forecast origin. A spy fit_fn (injected via backtest.py's
fit_fn seam) records exactly what each fit call actually saw; the test
fails loudly if any call's training data extends past its origin date,
or if the "actual" value it's later compared against isn't strictly in
the future relative to that origin.

Uses a fake, near-instant ARIMA stand-in (not the real pmdarima fit) so
this test is about the BACKTEST HARNESS's slicing discipline, not about
auto_arima's own behavior -- that's covered separately by
test_flat_series_sanity.py and by exercising the real model end-to-end
in forecasting/run_forecast_demo.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.arima.backtest import walk_forward_backtest

HORIZON_DAYS = 30
N_ORIGINS = 4
MIN_TRAIN_SIZE = 150


class FakeArimaModel:
    """Minimal stand-in matching the subset of pmdarima's ARIMA
    interface forecast() actually calls: .order and .predict(...)."""

    def __init__(self, last_value: float, order: tuple[int, int, int] = (0, 1, 0)):
        self.last_value = last_value
        self.order = order

    def predict(self, n_periods: int, return_conf_int: bool = False, alpha: float = 0.05):
        point = np.full(n_periods, self.last_value)
        if not return_conf_int:
            return point
        width = self.last_value * 0.1
        conf_int = np.column_stack([point - width, point + width])
        return point, conf_int


def _make_series(n: int = 400) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    values = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.Series(values, index=idx)


def test_backtest_never_fits_on_data_after_its_own_origin():
    series = _make_series(400)
    seen: list[tuple[pd.Timestamp, int]] = []

    def spy_fit(train_series: pd.Series) -> FakeArimaModel:
        seen.append((train_series.index.max(), len(train_series)))
        return FakeArimaModel(last_value=float(train_series.iloc[-1]))

    bt = walk_forward_backtest(
        series, horizon_days=HORIZON_DAYS, n_origins=N_ORIGINS, min_train_size=MIN_TRAIN_SIZE, fit_fn=spy_fit,
    )

    assert len(seen) == len(bt.origins) == N_ORIGINS

    for (seen_max_date, seen_len), origin in zip(seen, bt.origins):
        origin_date = pd.Timestamp(origin.origin_date)
        actual_date = pd.Timestamp(origin.actual_date)

        # The fit call's own training data ends exactly at (never past) its origin.
        assert seen_max_date == origin_date

        # The training slice contains exactly the rows up to and including the
        # origin -- not more (which would smuggle in future rows) and not fewer.
        origin_idx = series.index.get_loc(origin_date)
        assert seen_len == origin_idx + 1

        # The value compared against is strictly in the future relative to the
        # origin, exactly horizon_days of index steps ahead.
        actual_idx = series.index.get_loc(actual_date)
        assert actual_idx > origin_idx
        assert actual_idx - origin_idx == HORIZON_DAYS
