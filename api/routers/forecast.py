"""ARIMA forecasting endpoints, complementary to monte_carlo.py -- see
forecasting/__init__.py's module docstring for the conceptual
distinction this router's two endpoints are built to preserve.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from forecasting import DISCLAIMER as ARIMA_DISCLAIMER
from forecasting.arima.backtest import DEFAULT_MIN_TRAIN_SIZE, DEFAULT_N_ORIGINS, BacktestSummary, walk_forward_backtest
from forecasting.arima.comparison import compare, monte_carlo_from_extended_series
from forecasting.arima.model import DEFAULT_CONFIDENCE_LEVELS, fit_arima, forecast
from forecasting.arima.series import build_extended_portfolio_series
from monte_carlo import DISCLAIMER as MC_DISCLAIMER

from api.dependencies import get_portfolio_or_404
from api.schemas.forecast import (
    ArimaForecastResponse,
    BacktestOrigin,
    BacktestSummary,
    ForecastComparisonResponse,
    IntervalCoverage,
    PredictionInterval,
)

router = APIRouter(prefix="/portfolio", tags=["forecast"])


def _intervals_at_horizon(fc, horizon_days: int) -> list[PredictionInterval]:
    return [
        PredictionInterval(confidence=cl, lower=float(lo[-1]), upper=float(hi[-1]))
        for cl, (lo, hi) in fc.intervals.items()
    ]


def _build_series_or_422(portfolio_id: int):
    """build_extended_portfolio_series now reads real price history
    straight from the warehouse (see its own module docstring) instead
    of a separate ML-training-only cache, so this mainly guards against
    a portfolio with no currently-held positions or a genuine data gap.
    A ValueError here still deserves an honest, specific error instead
    of a raw 500.
    """
    try:
        return build_extended_portfolio_series(portfolio_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"ARIMA forecasting isn't available for this portfolio right now: {exc}",
        ) from exc


def _adaptive_backtest(series, horizon_days: int) -> BacktestSummary:
    """walk_forward_backtest's default min_train_size (600 days) and
    n_origins (5) assume a long series -- fine for the original demo's
    ~5-year extended series, but a real, honest confirmed case (a
    portfolio whose largest holding only IPO'd in late 2024) can have
    meaningfully less real price history than that, and the rigid
    default raised an unhandled ValueError that crashed the WHOLE
    endpoint -- including the point forecast above, which had already
    succeeded and needed no backtest to be valid on its own.

    Scales min_train_size/n_origins down to what this series can
    actually support, and returns an honest n_origins=0 summary (never
    a crash) when even a single origin isn't feasible -- the point
    forecast is still returned either way; a missing backtest is a
    real, visible "not enough history yet" state, not a hidden failure.
    """
    n = len(series)
    max_feasible_min_train = n - horizon_days - 2
    if max_feasible_min_train < 30:
        return BacktestSummary(n_origins=0, horizon_days=horizon_days, mae=0.0, rmse=0.0, interval_coverage={}, origins=[])

    min_train_size = min(DEFAULT_MIN_TRAIN_SIZE, max_feasible_min_train)
    span = (n - horizon_days - 1) - min_train_size
    n_origins = max(1, min(DEFAULT_N_ORIGINS, span + 1))
    return walk_forward_backtest(series, horizon_days=horizon_days, min_train_size=min_train_size, n_origins=n_origins)


def _backtest_schema(bt) -> BacktestSummary:
    return BacktestSummary(
        n_origins=bt.n_origins,
        horizon_days=bt.horizon_days,
        mae=bt.mae,
        rmse=bt.rmse,
        interval_coverage=[
            IntervalCoverage(confidence=cl, observed_coverage=cov) for cl, cov in bt.interval_coverage.items()
        ],
        origins=[
            BacktestOrigin(
                origin_date=o.origin_date,
                actual_date=o.actual_date,
                order=list(o.order),
                actual=o.actual,
                point_forecast=o.point_forecast,
                forecast_error=o.forecast_error,
                intervals=[
                    PredictionInterval(confidence=cl, lower=o.interval_lower[cl], upper=o.interval_upper[cl])
                    for cl in o.interval_lower
                ],
                within_interval={str(cl): within for cl, within in o.within_interval.items()},
            )
            for o in bt.origins
        ],
    )


@router.get("/{portfolio_id}/forecast/arima", response_model=ArimaForecastResponse)
def get_arima_forecast(
    portfolio_id: int,
    horizon_days: int = Query(252, gt=0, le=252 * 5),
    _: None = Depends(get_portfolio_or_404),
) -> ArimaForecastResponse:
    series = _build_series_or_422(portfolio_id)
    model = fit_arima(series)
    fc = forecast(model, horizon_days, DEFAULT_CONFIDENCE_LEVELS)
    bt = _adaptive_backtest(series, horizon_days)

    return ArimaForecastResponse(
        portfolio_id=portfolio_id,
        horizon_days=horizon_days,
        initial_value=float(series.iloc[-1]),
        order=list(fc.order),
        point_forecast=float(fc.point_forecast[-1]),
        intervals=_intervals_at_horizon(fc, horizon_days),
        disclaimer=ARIMA_DISCLAIMER,
        backtest=_backtest_schema(bt),
    )


@router.get("/{portfolio_id}/forecast/comparison", response_model=ForecastComparisonResponse)
def get_forecast_comparison(
    portfolio_id: int,
    horizon_days: int = Query(252, gt=0, le=252 * 5),
    n_simulations: int = Query(10_000, gt=0, le=200_000),
    _: None = Depends(get_portfolio_or_404),
) -> ForecastComparisonResponse:
    series = _build_series_or_422(portfolio_id)

    model = fit_arima(series)
    fc = forecast(model, horizon_days, DEFAULT_CONFIDENCE_LEVELS)
    intervals = {cl: (float(lo[-1]), float(hi[-1])) for cl, (lo, hi) in fc.intervals.items()}
    arima_point = float(fc.point_forecast[-1])

    mc = monte_carlo_from_extended_series(series, horizon_days, n_simulations=n_simulations)
    result = compare(mc, arima_point, intervals, fc.order, horizon_days)

    return ForecastComparisonResponse(
        portfolio_id=portfolio_id,
        horizon_days=horizon_days,
        initial_value=result.initial_value,
        monte_carlo_percentile_bands=result.monte_carlo_percentile_bands,
        monte_carlo_cagr=result.monte_carlo_cagr,
        monte_carlo_sigma_annual=result.monte_carlo_sigma_annual,
        monte_carlo_disclaimer=MC_DISCLAIMER,
        arima_point_forecast=result.arima_point_forecast,
        arima_intervals=[
            PredictionInterval(confidence=cl, lower=lo, upper=hi)
            for cl, (lo, hi) in result.arima_intervals.items()
        ],
        arima_order=list(result.arima_order),
        arima_disclaimer=ARIMA_DISCLAIMER,
        agreement_note=result.agreement_note,
    )
