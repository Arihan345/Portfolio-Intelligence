"""End-to-end demo: run every Phase 5 analytics metric against the real
example data loaded in the warehouse (TCS.NS/RELIANCE.NS, Jan-Jun 2024).

Run with: python -m analytics.run_analytics_demo
"""
from __future__ import annotations

import json

import pandas as pd

from analytics import data_access as da
from analytics.allocation.allocation import (
    asset_allocation,
    cash_allocation,
    currency_allocation,
    sector_allocation,
)
from analytics.allocation.concentration import (
    herfindahl_hirschman_index,
    largest_holding_pct,
    top_n_pct,
)
from analytics.attribution.attribution import asset_contribution, sector_contribution
from analytics.benchmark.benchmark import benchmark_comparison
from analytics.performance.returns import (
    cagr,
    capital_summary,
    periodic_returns,
    rolling_returns,
    time_weighted_return,
    xirr,
)
from analytics.risk.risk import (
    annualized_volatility,
    beta_alpha,
    conditional_var,
    correlation_matrix,
    downside_volatility,
    historical_var,
    max_drawdown,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)

PORTFOLIO_ID = 1
RISK_FREE_RATE_ANNUAL = 0.07  # illustrative: ~India 91-day T-Bill yield, configurable
VAR_CONFIDENCE = 0.95

# The example's real price data only covers Jan-Jun 2024 (Phase 3's
# load window); int_daily_holdings/mart_portfolio_performance extend
# the forward-filled price through today for point-in-time holdings
# lookups, which would flatten volatility/Sharpe/etc. with years of
# artificial zero-return days if included. Analytics here are computed
# over the real data window only.
ANALYSIS_START = "2024-01-01"
ANALYSIS_END = "2024-06-30"


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> None:
    pp = da.get_portfolio_performance(PORTFOLIO_ID)
    pp = pp[(pp["value_date"] >= ANALYSIS_START) & (pp["value_date"] <= ANALYSIS_END)]
    pp = pp.set_index("value_date")

    ap = da.get_asset_performance(PORTFOLIO_ID)
    ap = ap[(ap["as_of_date"] >= ANALYSIS_START) & (ap["as_of_date"] <= ANALYSIS_END)]

    alloc = da.get_allocation(PORTFOLIO_ID)
    alloc = alloc[(alloc["as_of_date"] >= ANALYSIS_START) & (alloc["as_of_date"] <= ANALYSIS_END)]

    bench = da.get_benchmark(PORTFOLIO_ID)
    bench = bench.set_index("value_date")

    tx = da.get_transactions(PORTFOLIO_ID)

    as_of = pd.Timestamp(ANALYSIS_END)

    # =========================== PERFORMANCE ===========================
    section("PERFORMANCE")

    latest_ap = ap[ap["as_of_date"] == as_of]
    invested_capital = float(latest_ap["cost_basis_inr"].sum())
    market_value = float(latest_ap["market_value_inr"].sum())
    summary = capital_summary(invested_capital, market_value)
    print("Capital summary (as of 2024-06-30):", json.dumps(summary, indent=2))

    start_nav = float(pp["total_nav_inr"].iloc[0])
    end_nav = float(pp["total_nav_inr"].iloc[-1])
    cagr_value = cagr(start_nav, end_nav, pp.index[0].date(), pp.index[-1].date())
    print(f"\nCAGR (start {pp.index[0].date()} -> end {pp.index[-1].date()}): {cagr_value:.4%}")

    daily_returns = pp["daily_return"].dropna()
    monthly = periodic_returns(pp["daily_return"], freq="ME")
    print("\nMonthly returns:")
    print(monthly.to_string())

    rolling_30d = rolling_returns(pp["daily_return"], window=30)
    print(f"\nLatest 30-day rolling return: {rolling_30d.iloc[-1]:.4%}")

    twr = time_weighted_return(pp["total_nav_inr"])
    print(f"\nTime-weighted return (no external DEPOSIT/WITHDRAWAL in this "
          f"dataset, so TWR = simple compounded NAV growth): {twr:.4%}")

    buy_sell_div = tx[tx["transaction_type"].isin(["BUY", "SELL", "DIVIDEND"])].copy()
    cash_flows = []
    for _, r in buy_sell_div.iterrows():
        if r["transaction_type"] == "BUY":
            amt = -(r["price_inr"] * r["quantity"] + r["fees"])
        elif r["transaction_type"] == "SELL":
            amt = r["price_inr"] * r["quantity"] - r["fees"] - r["tax"]
        else:  # DIVIDEND
            amt = r["price_inr"]
        cash_flows.append((r["txn_date"].date(), float(amt)))
    cash_flows.append((as_of.date(), market_value))  # terminal liquidation value
    xirr_value = xirr(cash_flows)
    print(f"\nXIRR (money-weighted, investor cash flows + terminal value): {xirr_value:.4%}")
    print("  cash flows used:", cash_flows)

    # =========================== ALLOCATION ===========================
    section("ALLOCATION")
    alloc_latest = alloc[alloc["as_of_date"] == as_of]

    print("Asset allocation:\n", asset_allocation(alloc_latest).to_string())
    print("\nSector allocation:\n", sector_allocation(alloc_latest).to_string())
    print("\nCurrency allocation:\n", currency_allocation(alloc_latest).to_string())

    holdings_mv = float(pp["holdings_market_value_inr"].iloc[-1])
    cash_bal = float(pp["cash_balance_inr"].iloc[-1])
    print("\nCash vs. holdings allocation:", cash_allocation(holdings_mv, cash_bal))

    # =========================== CONCENTRATION ===========================
    section("CONCENTRATION")
    weights = asset_allocation(alloc_latest)
    print(f"Largest holding %: {largest_holding_pct(weights):.2%}")
    print(f"Top-2 (all holdings) %: {top_n_pct(weights, 2):.2%}")
    hhi = herfindahl_hirschman_index(weights)
    n = len(weights)
    print(f"HHI: {hhi:.4f}  (bounds for n={n} holdings: [{1/n:.4f}, 1.0])")

    # =========================== RISK ===========================
    section("RISK")
    vol = annualized_volatility(daily_returns)
    dvol = downside_volatility(daily_returns)
    print(f"Annualized volatility: {vol:.2%}")
    print(f"Downside volatility: {dvol:.2%}")

    sharpe = sharpe_ratio(daily_returns, RISK_FREE_RATE_ANNUAL)
    sortino = sortino_ratio(daily_returns, RISK_FREE_RATE_ANNUAL)
    print(f"Sharpe ratio (rf={RISK_FREE_RATE_ANNUAL:.0%}): {sharpe:.3f}")
    print(f"Sortino ratio (rf={RISK_FREE_RATE_ANNUAL:.0%}): {sortino:.3f}")

    bench_returns = bench["benchmark_daily_return"].reindex(daily_returns.index)
    ba = beta_alpha(daily_returns, bench_returns, RISK_FREE_RATE_ANNUAL)
    print(f"Beta vs NIFTY 50: {ba['beta']:.3f}, Alpha (annualized): {ba['alpha_annualized']:.2%}")

    mdd = max_drawdown(pp["total_nav_inr"])
    print(f"Max drawdown: {mdd['max_drawdown']:.2%} "
          f"(peak {mdd['peak_date'].date()} -> trough {mdd['trough_date'].date()}, "
          f"duration {mdd['drawdown_duration_days']}d, "
          f"recovery {mdd['recovery_date'].date() if mdd['recovery_date'] is not None else 'not yet'})")

    hvar = historical_var(daily_returns, VAR_CONFIDENCE)
    pvar = parametric_var(daily_returns, VAR_CONFIDENCE)
    cvar = conditional_var(daily_returns, VAR_CONFIDENCE)
    print(f"Historical VaR ({VAR_CONFIDENCE:.0%}): {hvar:.2%}")
    print(f"Parametric VaR ({VAR_CONFIDENCE:.0%}): {pvar:.2%}")
    print(f"Conditional VaR / Expected Shortfall ({VAR_CONFIDENCE:.0%}): {cvar:.2%}")

    wide_returns = ap.pivot(index="as_of_date", columns="ticker", values="market_value_inr")
    wide_returns = wide_returns.pct_change().dropna(how="all")
    wide_returns["NIFTY50"] = bench_returns.reindex(wide_returns.index)
    corr = correlation_matrix(wide_returns)
    print("\nCorrelation matrix:\n", corr.round(3).to_string())

    # =========================== BENCHMARK ===========================
    section("BENCHMARK (NIFTY 50)")
    comparison = benchmark_comparison(daily_returns, bench_returns, RISK_FREE_RATE_ANNUAL)
    print(json.dumps(comparison, indent=2))

    # =========================== ATTRIBUTION ===========================
    section("ATTRIBUTION")
    # Attribution's contribution = weight * return identity only
    # reconciles to portfolio return over a window with NO rebalancing
    # (no BUY/SELL changing composition) -- otherwise the "weight" and
    # "return" are computed over a composition that didn't hold
    # throughout, which is not a well-defined thing to attribute.
    # The portfolio has two composition changes: RELIANCE.NS enters on
    # 2024-02-01, and TCS.NS is partially sold on 2024-06-01. The only
    # stable-composition window is [2024-02-01, 2024-05-31] (10 TCS.NS
    # + 5 RELIANCE.NS, zero transactions) -- attribution is computed
    # over that window. An initial attempt using the portfolio's very
    # first day (2024-01-01) as the baseline silently dropped
    # RELIANCE.NS (it did not exist yet that day) and reconciled only
    # against itself rather than the portfolio's real return -- caught
    # by checking against mart_portfolio_performance's actual holdings
    # growth for the window, which is the fix applied here.
    ATTRIB_START = "2024-02-01"
    ATTRIB_END = "2024-05-31"

    start_alloc = alloc[alloc["as_of_date"] == ATTRIB_START]
    start_weights = asset_allocation(start_alloc)
    print(f"Start weights ({ATTRIB_START}):\n{start_weights.to_string()}")

    end_prices = ap[ap["as_of_date"] == ATTRIB_END].set_index("ticker")["last_close_inr"]
    start_prices = ap[ap["as_of_date"] == ATTRIB_START].set_index("ticker")["last_close_inr"]
    asset_period_returns = (end_prices / start_prices - 1).dropna()
    print(f"\nAsset price returns ({ATTRIB_START} -> {ATTRIB_END}):\n{asset_period_returns.to_string()}")

    contrib = asset_contribution(start_weights, asset_period_returns)
    print("\nAsset contribution to return:\n", contrib.to_string())

    sector_map = ap.drop_duplicates("ticker").set_index("ticker")["sector"]
    sector_contrib = sector_contribution(start_weights, asset_period_returns, sector_map)
    print("\nSector contribution to return:\n", sector_contrib.to_string())

    total_from_contrib = float(contrib.sum())

    # Reconciliation target: the portfolio's ACTUAL holdings-value
    # return over the same window (not total_nav_inr, which also moves
    # from the 2024-03-15 dividend cash inflow -- a real cash event,
    # but not a price return any asset weight*return term can produce).
    holdings_start = float(pp.loc[ATTRIB_START, "holdings_market_value_inr"])
    holdings_end = float(pp.loc[ATTRIB_END, "holdings_market_value_inr"])
    actual_holdings_return = holdings_end / holdings_start - 1

    print(f"\nSum of contributions: {total_from_contrib:.6%}")
    print(f"Actual holdings return over the same window: {actual_holdings_return:.6%}")
    assert abs(total_from_contrib - actual_holdings_return) < 1e-9, "attribution does not reconcile!"
    print("Reconciliation holds exactly against the portfolio's actual holdings return.")


if __name__ == "__main__":
    main()
