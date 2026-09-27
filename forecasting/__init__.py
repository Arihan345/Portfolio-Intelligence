"""Complementary time-series forecasting, alongside (not replacing)
Phase 6's Monte Carlo engine.

CONCEPTUAL DISTINCTION (kept explicit everywhere this module's output is
surfaced -- API responses, UI copy -- because it is easy to blur and the
two techniques answer different questions):

  - Monte Carlo (monte_carlo/) is DISTRIBUTION-BASED SIMULATION. It
    assumes returns follow a stated distribution (log-normal / GBM)
    parameterized by historical mean/volatility/correlation, then draws
    many random paths from that assumed distribution. Its output -- a
    spread of percentile bands -- describes "if returns behave like
    this distribution, here is the range of plausible outcomes."

  - ARIMA (forecasting/arima/) is PATTERN-BASED FORECASTING. It fits a
    model directly to the actual historical time series -- its trend
    and autocorrelation structure -- and extrapolates that fitted
    pattern forward. Its output is a single point forecast plus a
    STATISTICAL prediction interval derived from the fitted model's own
    residual variance, not from simulation.

Neither is more "correct" -- they make different assumptions and can
legitimately disagree. forecasting/arima/comparison.py reports
agreement/divergence between the two rather than reconciling them into
one number.

ARCHITECTURE (mirrors monte_carlo/'s pure-function / data-access split):
  - arima/series.py: builds the extended historical portfolio-value
    series ARIMA fits on (impure -- reads ml/'s cached multi-year price
    history plus the portfolio's current allocation weights).
  - arima/model.py: pure(ish) fit/forecast wrapper around pmdarima's
    auto_arima -- takes a plain pandas Series, no database access.
  - arima/backtest.py: walk-forward backtesting with an explicit
    no-lookahead guarantee (see its module docstring).
  - arima/comparison.py: combines a Monte Carlo run with an ARIMA
    forecast on the same value scale and horizon.
"""

DISCLAIMER = (
    "ARIMA statistical forecast, not a prediction of certainty. This model "
    "extrapolates the pattern (trend and autocorrelation) found in the "
    "portfolio's own historical value series. The prediction interval "
    "reflects the fitted model's own residual uncertainty, not a simulated "
    "distribution -- a genuinely different method from Monte Carlo's "
    "scenario simulation. Backtested interval coverage below shows how "
    "reliable these intervals have actually been historically; if coverage "
    "is below its stated confidence level, treat the intervals as "
    "overconfident."
)
