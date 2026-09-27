"""Pure(ish) ARIMA fit/forecast wrapper. Takes a plain pandas Series (no
database access), so it is testable with hand-picked series independent
of the warehouse being up -- mirrors monte_carlo/simulate.py's split
between pure math and the impure data-fetching layer (series.py here).

WHY auto_arima (pmdarima) instead of a hand-picked (p,d,q) order: a
manually guessed order is a modeling choice with no principled basis for
this specific series, and would invite exactly the kind of "tuned to
look good" result this project's dev discipline explicitly forbids for
the ML promotion gate -- the same discipline applies here. auto_arima
performs a stepwise search over candidate orders, selecting the one
that minimizes AIC (a proper penalized-likelihood criterion trading fit
quality against model complexity) on the same training data every time,
so the order is a reproducible function of the data, not a guess tuned
by eye.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pmdarima as pm

DEFAULT_CONFIDENCE_LEVELS: tuple[float, ...] = (0.80, 0.95)


@dataclass
class ArimaForecast:
    order: tuple[int, int, int]
    horizon_days: int
    point_forecast: np.ndarray  # shape (horizon_days,)
    intervals: dict[float, tuple[np.ndarray, np.ndarray]]  # confidence -> (lower, upper), each shape (horizon_days,)


def fit_arima(series: pd.Series, seasonal: bool = False) -> pm.arima.ARIMA:
    """Fits auto_arima on the series' values (order only -- no exogenous
    regressors). seasonal=False since portfolio value has no known fixed
    seasonal period; a stepwise (p,d,q) search over non-seasonal orders
    is what auto_arima performs by default.
    """
    values = series.to_numpy(dtype=float)
    return pm.auto_arima(
        values,
        seasonal=seasonal,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
        # Explicit, not pmdarima's "auto" default: for a portfolio-value
        # series (never meaningfully mean-zero), auto_arima's default
        # intercept handling drops the intercept entirely in a degenerate
        # edge case (a fully constant series -- see
        # forecasting/tests/test_flat_series_sanity.py), forecasting 0
        # instead of the series' own level. Forcing an intercept fixes
        # that case and is also the right choice for real, non-degenerate
        # portfolio-value data.
        with_intercept=True,
    )


def forecast(
    model: pm.arima.ARIMA,
    horizon_days: int,
    confidence_levels: tuple[float, ...] = DEFAULT_CONFIDENCE_LEVELS,
) -> ArimaForecast:
    """Point forecast is identical across confidence levels (it's the
    same fitted model); only the interval width changes with alpha, so
    predict() is called once per confidence level purely to get that
    level's interval, discarding the (identical) point forecast on
    repeat calls.
    """
    point_forecast, _ = model.predict(horizon_days, return_conf_int=True, alpha=1 - confidence_levels[0])
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for cl in confidence_levels:
        pf, conf_int = model.predict(horizon_days, return_conf_int=True, alpha=1 - cl)
        intervals[cl] = (conf_int[:, 0], conf_int[:, 1])
        point_forecast = pf  # identical every call; harmless to reassign

    order = model.order if hasattr(model, "order") else tuple(model.get_params()["order"])
    return ArimaForecast(
        order=tuple(order),
        horizon_days=horizon_days,
        point_forecast=point_forecast,
        intervals=intervals,
    )
