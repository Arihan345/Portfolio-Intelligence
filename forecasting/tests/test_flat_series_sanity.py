"""Basic correctness check before trusting auto_arima on real volatile
data: a perfectly flat/constant historical series (zero variance, no
trend, no autocorrelation to find) should produce a flat point forecast
at that same constant level, with a narrow prediction interval -- there
is no historical evidence of any variability for the model to widen the
interval around.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.arima.model import fit_arima, forecast

CONSTANT_LEVEL = 1000.0


def test_flat_series_produces_flat_forecast_with_narrow_interval():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    series = pd.Series(np.full(300, CONSTANT_LEVEL), index=idx)

    model = fit_arima(series)
    fc = forecast(model, horizon_days=30)

    assert np.allclose(fc.point_forecast, CONSTANT_LEVEL, atol=1.0)

    lower_95, upper_95 = fc.intervals[0.95]
    width_95 = upper_95[-1] - lower_95[-1]
    # Narrow relative to the series' own level -- there is zero historical
    # variance to justify anything close to Monte Carlo/real-market-style
    # interval widths here.
    assert width_95 < CONSTANT_LEVEL * 0.05
