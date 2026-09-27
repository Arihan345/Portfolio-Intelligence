"""Real end-to-end run of the ARIMA forecasting module against this
repo's actual extended price history and live warehouse allocation --
same spirit as monte_carlo/run_monte_carlo_demo.py and
mlops/run_mlops_demo.py. Prints real fitted order, forecast, and
walk-forward backtest accuracy -- nothing here is fabricated.

Run with: python -m forecasting.run_forecast_demo
"""
from __future__ import annotations

from forecasting.arima.backtest import walk_forward_backtest
from forecasting.arima.comparison import compare, monte_carlo_from_extended_series
from forecasting.arima.model import fit_arima, forecast
from forecasting.arima.series import build_extended_portfolio_series

PORTFOLIO_ID = 1
HORIZON_DAYS = 252


def main() -> None:
    series = build_extended_portfolio_series(PORTFOLIO_ID)
    print(f"Extended portfolio-value series: {len(series)} rows, "
          f"{series.index[0]} -> {series.index[-1]}")
    print(f"Current value (last real point): {series.iloc[-1]:,.2f}\n")

    model = fit_arima(series)
    print(f"auto_arima selected order: {model.order}")

    fc = forecast(model, HORIZON_DAYS)
    print(f"\n{HORIZON_DAYS}-day point forecast: {fc.point_forecast[-1]:,.2f}")
    for cl, (lo, hi) in fc.intervals.items():
        print(f"  {int(cl*100)}% interval: [{lo[-1]:,.2f}, {hi[-1]:,.2f}]")

    print("\nWalk-forward backtest (no-lookahead, expanding-window origins):")
    bt = walk_forward_backtest(series, horizon_days=HORIZON_DAYS)
    print(f"  n_origins={bt.n_origins}  MAE={bt.mae:,.2f}  RMSE={bt.rmse:,.2f}")
    for cl, cov in bt.interval_coverage.items():
        flag = "" if cov >= cl - 0.15 else "  <-- BELOW stated confidence: intervals may be overconfident"
        print(f"  {int(cl*100)}% interval observed coverage: {cov*100:.0f}%{flag}")
    for o in bt.origins:
        print(f"    origin {o.origin_date} -> actual {o.actual_date}: "
              f"actual={o.actual:,.0f} forecast={o.point_forecast:,.0f} "
              f"error={o.forecast_error:+,.0f} within_95%={o.within_interval[0.95]}")

    print("\nComparison vs. Monte Carlo (parameters estimated from the SAME extended series, same anchor date):")
    mc = monte_carlo_from_extended_series(series, HORIZON_DAYS, seed=42)
    print(f"  Monte Carlo CAGR={mc.cagr:.2%}  sigma_annual={mc.sigma_annual:.2%}")
    print(f"  Monte Carlo percentile bands: {mc.percentile_bands}")
    intervals = {cl: (float(lo[-1]), float(hi[-1])) for cl, (lo, hi) in fc.intervals.items()}
    result = compare(mc, float(fc.point_forecast[-1]), intervals, fc.order, HORIZON_DAYS)
    print(f"\n  {result.agreement_note}")


if __name__ == "__main__":
    main()
