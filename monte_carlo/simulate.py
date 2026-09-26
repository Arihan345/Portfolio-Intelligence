"""Pure GBM simulation math. No database access -- every function here
takes plain numpy/scalar inputs so it is testable with hand-picked
parameters, independent of the warehouse being up.

Both models discretize geometric Brownian motion in log-space:
    log(S_{t+dt}) - log(S_t) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    Z ~ N(0, 1)

The (mu - 0.5*sigma^2) drift adjustment (vs. plain mu) is the standard
Ito correction: it is what makes exp(...) of a sum of these increments
have the correct EXPECTED level E[S_T] = S_0 * exp(mu*T), even though
the MEDIAN path grows at the lower rate (mu - 0.5*sigma^2)*T. See
params.py for why mu is derived from historical CAGR plus a matching
0.5*sigma^2 add-back, so that the simulation's median path reproduces
the historical CAGR by construction.
"""
from __future__ import annotations

import numpy as np


def simulate_portfolio_gbm(
    s0: float,
    mu_annual: float,
    sigma_annual: float,
    horizon_days: int,
    n_sims: int,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> np.ndarray:
    """Baseline model: single geometric Brownian motion at the
    portfolio level (portfolio's own historical return/volatility, no
    per-asset detail).

    Returns an (n_sims, horizon_days + 1) array of portfolio value
    paths; column 0 is s0 for every simulation, column horizon_days is
    the terminal value.

    Vectorized: one (n_sims, horizon_days) normal draw, one cumsum, one
    exp -- no per-simulation or per-day Python loop.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, horizon_days))
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    diffusion = sigma_annual * np.sqrt(dt) * z
    log_returns = drift + diffusion
    cum_log_returns = np.cumsum(log_returns, axis=1)

    paths = np.empty((n_sims, horizon_days + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(cum_log_returns)
    return paths


def simulate_portfolio_multi_asset(
    weights: np.ndarray,
    s0_total: float,
    mu_annual: np.ndarray,
    sigma_annual: np.ndarray,
    correlation: np.ndarray,
    horizon_days: int,
    n_sims: int,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """More realistic model: simulates each held asset's own GBM path
    with CORRELATED shocks (via Cholesky decomposition of the
    correlation matrix), then sums the weighted asset values into a
    portfolio path each day.

    Why correlation matters: if TCS.NS and RELIANCE.NS were simulated
    as independent GBMs, the portfolio's simulated variance would be
    the naive weighted sum of each asset's variance
    (w1^2*sigma1^2 + w2^2*sigma2^2), which is only correct when their
    correlation is exactly 0. The true portfolio variance is
        w1^2*sigma1^2 + w2^2*sigma2^2 + 2*w1*w2*sigma1*sigma2*rho
    Since these two holdings are (per Phase 5's real correlation
    matrix) POSITIVELY correlated (not independent, not perfectly
    correlated), ignoring rho would misstate the portfolio's true risk
    in whichever direction rho actually points -- understating it here,
    since a positive rho adds to portfolio variance versus the
    independent-assets assumption. Simulating from the real correlation
    matrix reproduces the correct joint variance automatically, without
    hand-deriving the closed-form adjustment.

    Cholesky: for correlation matrix R (positive semi-definite), R =
    L @ L.T for a unique lower-triangular L. Given independent standard
    normal draws Z (one vector per asset), Y = L @ Z has covariance L @
    I @ L.T = R -- i.e. Y is a standard-normal vector with the target
    correlation structure. Applied per simulated day.

    Returns (asset_paths, portfolio_path):
      asset_paths shape (n_sims, horizon_days + 1, n_assets)
      portfolio_path shape (n_sims, horizon_days + 1)
    No daily rebalancing is modeled -- each asset's initial euro value
    is weights[i] * s0_total, and it then compounds on its own from
    there (buy-and-hold), matching how Phase 4/5 treat the real
    portfolio (no automatic rebalancing exists in this codebase).
    """
    n_assets = len(weights)
    rng = np.random.default_rng(seed)

    L = np.linalg.cholesky(correlation)

    z = rng.standard_normal((n_sims, horizon_days, n_assets))
    correlated_z = z @ L.T  # shape preserved: (n_sims, horizon_days, n_assets)

    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt  # shape (n_assets,)
    diffusion = sigma_annual * np.sqrt(dt) * correlated_z  # broadcasts over last axis
    log_returns = drift + diffusion
    cum_log_returns = np.cumsum(log_returns, axis=1)

    s0_per_asset = weights * s0_total  # shape (n_assets,)
    asset_paths = np.empty((n_sims, horizon_days + 1, n_assets))
    asset_paths[:, 0, :] = s0_per_asset
    asset_paths[:, 1:, :] = s0_per_asset * np.exp(cum_log_returns)

    portfolio_path = asset_paths.sum(axis=2)
    return asset_paths, portfolio_path
