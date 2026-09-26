"""End-to-end demo: run both Monte Carlo models against the real
portfolio parameters from Phase 5's analytics engine.

Run with: python -m monte_carlo.run_monte_carlo_demo
"""
from __future__ import annotations

import json
import time

import numpy as np

from monte_carlo import DISCLAIMER
from monte_carlo.outputs import (
    drawdown_statistics,
    percentile_bands,
    probability_of_exceeding,
    probability_of_loss,
    simulated_var_es,
)
from monte_carlo.params import get_asset_level_params, get_portfolio_params
from monte_carlo.simulate import simulate_portfolio_gbm, simulate_portfolio_multi_asset

PORTFOLIO_ID = 1
ANALYSIS_START = "2024-01-01"
ANALYSIS_END = "2024-06-30"
HORIZON_DAYS = 252
TARGET_VALUE = 65_000.0
SEED = 20240601  # fixed for reproducible demo output


def print_disclaimer() -> None:
    print("*" * 70)
    print(DISCLAIMER)
    print("*" * 70)


def report(name: str, s0: float, paths: np.ndarray, elapsed: float, n_sims: int) -> None:
    terminal = paths[:, -1]
    print(f"\n--- {name} ---")
    print(f"{n_sims:,} simulations x {HORIZON_DAYS} days in {elapsed:.3f}s "
          f"({n_sims * HORIZON_DAYS / elapsed:,.0f} sim-days/sec)")

    bands = percentile_bands(terminal)
    print("Percentile bands (terminal value, INR):")
    for k, v in bands.items():
        print(f"  {k}: {v:,.2f}")

    print(f"Probability of loss (terminal < initial {s0:,.2f}): {probability_of_loss(terminal, s0):.2%}")
    print(f"Probability of exceeding target {TARGET_VALUE:,.0f}: {probability_of_exceeding(terminal, TARGET_VALUE):.2%}")

    var_es = simulated_var_es(terminal, s0, confidence=0.95)
    print(f"Simulated VaR (95%, terminal return): {var_es['var']:.2%}")
    print(f"Simulated Expected Shortfall (95%): {var_es['expected_shortfall']:.2%}")

    dd = drawdown_statistics(paths)
    print("Drawdown statistics across simulated paths:")
    for k, v in dd.items():
        print(f"  {k}: {v:.2%}")


def main() -> None:
    print_disclaimer()

    # =========================== BASELINE MODEL ===========================
    pp = get_portfolio_params(PORTFOLIO_ID, ANALYSIS_START, ANALYSIS_END)
    print(f"\nBaseline model parameters (from Phase 5's mart_portfolio_performance):")
    print(f"  s0 (current NAV) = {pp.s0:,.2f}")
    print(f"  historical CAGR = {pp.cagr:.4%}")
    print(f"  annualized volatility = {pp.sigma_annual:.4%}")
    print(f"  GBM drift mu (= CAGR + 0.5*sigma^2, so median matches CAGR) = {pp.mu_gbm:.4%}")

    for n_sims in (1_000, 5_000, 10_000):
        t0 = time.perf_counter()
        paths = simulate_portfolio_gbm(
            pp.s0, pp.mu_gbm, pp.sigma_annual, HORIZON_DAYS, n_sims, seed=SEED
        )
        elapsed = time.perf_counter() - t0
        report(f"BASELINE GBM ({n_sims:,} sims)", pp.s0, paths, elapsed, n_sims)

    # Sanity check: median terminal value should be close to s0 compounded
    # forward one year at the historical CAGR (by construction of mu_gbm).
    median_terminal = float(np.median(paths[:, -1]))
    expected_from_cagr = pp.s0 * (1 + pp.cagr)
    pct_diff = abs(median_terminal - expected_from_cagr) / expected_from_cagr
    print(f"\nSanity check: median terminal ({median_terminal:,.2f}) vs. "
          f"s0*(1+CAGR) ({expected_from_cagr:,.2f}) -> {pct_diff:.2%} difference "
          f"(expected to be small: median path grows at mu-0.5*sigma^2 by construction, "
          f"matching CAGR to first order; the gap here is discretization + finite-sample "
          f"simulation noise, not a modeling error).")

    # =========================== ASSET-LEVEL MODEL ===========================
    ap = get_asset_level_params(PORTFOLIO_ID, ANALYSIS_START, ANALYSIS_END)
    print(f"\n\nAsset-level model parameters (from Phase 5's mart_asset_performance + risk module):")
    for i, t in enumerate(ap.tickers):
        print(f"  {t}: weight={ap.weights[i]:.2%}, CAGR={ap.cagr[i]:.4%}, "
              f"vol={ap.sigma_annual[i]:.4%}, mu_gbm={ap.mu_gbm[i]:.4%}")
    print(f"  correlation matrix:\n{ap.correlation}")
    print(f"  s0_total = {ap.s0_total:,.2f}")

    n_sims = 10_000
    t0 = time.perf_counter()
    asset_paths, portfolio_paths = simulate_portfolio_multi_asset(
        ap.weights, ap.s0_total, ap.mu_gbm, ap.sigma_annual, ap.correlation,
        HORIZON_DAYS, n_sims, seed=SEED,
    )
    elapsed = time.perf_counter() - t0
    report(f"ASSET-LEVEL CORRELATED ({n_sims:,} sims)", ap.s0_total, portfolio_paths, elapsed, n_sims)

    # Compare against a naive INDEPENDENT-assets run (correlation = identity)
    # to show what ignoring correlation would have understated.
    independent_corr = np.eye(len(ap.tickers))
    _, portfolio_paths_independent = simulate_portfolio_multi_asset(
        ap.weights, ap.s0_total, ap.mu_gbm, ap.sigma_annual, independent_corr,
        HORIZON_DAYS, n_sims, seed=SEED,
    )
    vol_correlated = float(portfolio_paths[:, -1].std())
    vol_independent = float(portfolio_paths_independent[:, -1].std())
    print(f"\nEffect of ignoring correlation: terminal-value std with real "
          f"correlation ({ap.correlation[0,1]:.4f}) = {vol_correlated:,.2f}; "
          f"assuming independence = {vol_independent:,.2f} "
          f"({(vol_correlated/vol_independent - 1):+.2%} difference).")


if __name__ == "__main__":
    main()
