"""Walk-forward backtesting for ARIMA -- the same no-lookahead discipline
ml.walk_forward applies to the classifier, applied here to a forecasting
model instead.

Definition: pick n_origins dates spread across the available history
(after leaving room for both a minimum training window and a full
horizon_days of REALIZED future to compare against). For each origin T:
  1. Fit ARIMA using ONLY series.loc[:T] -- nothing after T is ever
     passed into fit_fn.
  2. Forecast horizon_days trading days forward from T.
  3. Compare the forecast to the REALIZED value at T + horizon_days
     (already-known historical data, since T + horizon_days <= today).
Report MAE/RMSE of the point forecast across origins, and -- the real
test of whether the model's uncertainty estimates are honest -- what
fraction of origins had their realized value actually fall inside the
model's own stated prediction interval, compared to that interval's
stated confidence level. If observed coverage is meaningfully below the
stated confidence (e.g. the 95% interval only contains the realized
value 60% of the time across backtests), the intervals are overconfident
-- reported as INTERVAL_COVERAGE below and surfaced honestly in the API/
UI, not hidden or explained away.

fit_fn is an injectable seam (defaults to arima.model.fit_arima) purely
so tests can wrap it with a spy asserting no call ever receives data
past its own origin -- see forecasting/tests/test_no_lookahead.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from forecasting.arima.model import DEFAULT_CONFIDENCE_LEVELS, fit_arima, forecast

DEFAULT_N_ORIGINS = 5
DEFAULT_MIN_TRAIN_SIZE = 600


@dataclass
class BacktestOrigin:
    origin_date: str
    actual_date: str
    order: tuple[int, int, int]
    actual: float
    point_forecast: float
    forecast_error: float  # point_forecast - actual
    interval_lower: dict[float, float] = field(default_factory=dict)
    interval_upper: dict[float, float] = field(default_factory=dict)
    within_interval: dict[float, bool] = field(default_factory=dict)


@dataclass
class BacktestSummary:
    n_origins: int
    horizon_days: int
    mae: float
    rmse: float
    interval_coverage: dict[float, float]  # confidence level -> observed fraction within interval
    origins: list[BacktestOrigin]


def walk_forward_backtest(
    series: pd.Series,
    horizon_days: int = 252,
    n_origins: int = DEFAULT_N_ORIGINS,
    confidence_levels: tuple[float, ...] = DEFAULT_CONFIDENCE_LEVELS,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    fit_fn: Callable[[pd.Series], object] = fit_arima,
) -> BacktestSummary:
    series = series.sort_index()
    n = len(series)
    last_origin_idx = n - horizon_days - 1
    if last_origin_idx <= min_train_size:
        raise ValueError(
            f"not enough history for a {horizon_days}-day walk-forward backtest with "
            f"min_train_size={min_train_size}: have {n} rows, need > {min_train_size + horizon_days + 1}"
        )

    origin_idxs = np.linspace(min_train_size, last_origin_idx, n_origins, dtype=int)

    origins: list[BacktestOrigin] = []
    for idx in origin_idxs:
        train = series.iloc[: idx + 1]  # inclusive of the origin day itself, nothing after
        origin_date = series.index[idx]
        actual_idx = idx + horizon_days
        actual_date = series.index[actual_idx]
        actual = float(series.iloc[actual_idx])

        model = fit_fn(train)
        fc = forecast(model, horizon_days, confidence_levels)
        point = float(fc.point_forecast[-1])

        interval_lower, interval_upper, within = {}, {}, {}
        for cl in confidence_levels:
            lo, hi = fc.intervals[cl]
            interval_lower[cl] = float(lo[-1])
            interval_upper[cl] = float(hi[-1])
            within[cl] = interval_lower[cl] <= actual <= interval_upper[cl]

        origins.append(
            BacktestOrigin(
                origin_date=str(origin_date.date()) if hasattr(origin_date, "date") else str(origin_date),
                actual_date=str(actual_date.date()) if hasattr(actual_date, "date") else str(actual_date),
                order=fc.order,
                actual=actual,
                point_forecast=point,
                forecast_error=point - actual,
                interval_lower=interval_lower,
                interval_upper=interval_upper,
                within_interval=within,
            )
        )

    errors = np.array([o.forecast_error for o in origins])
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    interval_coverage = {
        cl: float(np.mean([o.within_interval[cl] for o in origins])) for cl in confidence_levels
    }

    return BacktestSummary(
        n_origins=len(origins),
        horizon_days=horizon_days,
        mae=mae,
        rmse=rmse,
        interval_coverage=interval_coverage,
        origins=origins,
    )
