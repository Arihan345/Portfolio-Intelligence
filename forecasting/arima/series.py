"""Builds the extended historical portfolio-value series ARIMA fits and
backtests on.

WHY NOT the real portfolio NAV (mart_portfolio_performance) directly?
That series is a single portfolio-level number that already blends
every trade's cash-flow timing into it (see mart_portfolio_performance
.sql's own TODO about this not being cash-flow-weighted) -- fine for
"what's my NAV done," wrong for "what does each holding's OWN price
series look like," which is what ARIMA needs one ticker at a time.

WHERE THE PER-TICKER PRICES COME FROM: fact_daily_prices, via
analytics.data_access.get_price_history -- the SAME warehouse table
Monte Carlo's baseline/asset-level parameter estimation already reads
successfully for any real upload (get_portfolio_params/
get_asset_level_params in monte_carlo/params.py). This used to instead
require ml.data.fetch_price_history's separate, fixed cache file --
built once for the ML model's OWN training universe (TCS.NS/
RELIANCE.NS) and never extended to any other upload's real tickers.
Confirmed as a real, reproduced bug: every real portfolio upload with
different holdings hit a 422 here ("missing tickers"), even though the
warehouse already had real price history for those exact tickers the
whole time (fetched during upload, same as Monte Carlo uses) -- ARIMA
was pointed at the wrong data source, not missing data. Unifying on
the warehouse here means ARIMA can never again depend on a second,
separately-populated cache staying in sync with whatever's uploaded.

The real trade-off this doesn't remove: the warehouse's price history
for a given ticker only goes back to that ticker's first REAL
transaction (or its real listing date, if later -- e.g. HYUNDAI.NS
only has real prices from its actual 2024 NSE IPO onward, correctly,
since it did not trade before that). For a portfolio built up recently
this can be meaningfully shorter than 5 years, which does limit how
many independent walk-forward backtest origins are available -- a
real, honest limitation of the data itself, not something to paper
over, and the backtest's own n_origins is always reported alongside
its results for exactly this reason.

HOW THE VALUE SERIES IS CONSTRUCTED (an explicit, stated assumption --
not the portfolio's literal historical value, since these exact
holdings/weights did not necessarily exist for the full history):
  1. Take the portfolio's CURRENT weights (mart_allocation's latest
     snapshot, positions actually still held) and CURRENT total market
     value.
  2. For each ticker, compute its price relative to TODAY's price
     (price_t / price_today) across its own available real price
     history -- the ticker's own real historical return path, anchored
     at 1.0 on the most recent date.
  3. Multiply by (weight * current_total_value) per ticker and sum
     across tickers.
This reproduces the same buy-and-hold, fixed-weight, no-rebalancing
convention monte_carlo.simulate already uses for the asset-level model
(see that module's docstring) -- applied here to REAL historical prices
instead of simulated ones, and anchored so the series' last value
exactly equals today's real total market value (the one figure we know
precisely, from the real allocation mart) rather than an arbitrary
starting point.
"""
from __future__ import annotations

import pandas as pd

from analytics import data_access as da
from analytics.allocation.allocation import asset_allocation


def build_extended_portfolio_series(
    portfolio_id: int, price_history: pd.DataFrame | None = None
) -> pd.Series:
    """Returns a pd.Series of portfolio value, indexed by date
    (ascending, daily), spanning the currently-held tickers' own real
    available price history window.

    price_history (optional): inject a specific price DataFrame
    (columns ticker/date/close) -- tests use this to stay offline and
    deterministic. Omitted (the default): fetched fresh from the real
    warehouse via analytics.data_access.get_price_history for whichever
    tickers this portfolio actually holds right now.
    """
    alloc = da.get_allocation(portfolio_id)
    if alloc.empty:
        raise ValueError(f"no allocation data for portfolio {portfolio_id}")

    latest_date = alloc["as_of_date"].max()
    snapshot = alloc[alloc["as_of_date"] == latest_date]
    # Only currently-held (nonzero market value) positions need price
    # history -- a fully-exited position doesn't need forecasting, and
    # requiring history for every ticker EVER traded (most real
    # portfolios accumulate several closed-out positions) needlessly
    # widened the "missing tickers" list and the fetch cost below.
    snapshot = snapshot[snapshot["market_value_inr"] > 1e-9]
    if snapshot.empty:
        raise ValueError(f"no currently-held positions for portfolio {portfolio_id}")
    weights = asset_allocation(snapshot)
    current_total_value = float(snapshot["market_value_inr"].sum())

    held_tickers = list(weights.index)
    if price_history is None:
        price_history = da.get_price_history(held_tickers)

    wide = price_history.pivot(index="date", columns="ticker", values="close").sort_index()

    missing = [t for t in held_tickers if t not in wide.columns]
    if missing:
        raise ValueError(
            f"extended price history is missing tickers held by portfolio {portfolio_id}: {missing}"
        )

    wide = wide[held_tickers].dropna()
    w = weights[held_tickers]
    w = w / w.sum()  # renormalize in case any held ticker had to be dropped above

    unit_growth = wide / wide.iloc[-1]  # each ticker's price relative to its own latest cached price
    s0_per_asset = w * current_total_value
    portfolio_value = (unit_growth * s0_per_asset).sum(axis=1)
    portfolio_value.name = "portfolio_value"
    return portfolio_value
