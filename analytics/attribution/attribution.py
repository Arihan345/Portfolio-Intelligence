"""Return attribution: how much of the portfolio's total return each
asset (and each sector) contributed.

Definition:
    contribution_i = weight_i * return_i
    sum(contribution_i for all i) == total_portfolio_return

weight_i is each asset's share of TOTAL portfolio value at the START
of the period (not the end -- using end-of-period weights would let an
asset that grew a lot get credit proportional to its post-growth size,
double-counting its own return). return_i is that asset's simple
return over the period. Because weights are taken at the start and sum
to 1, the weighted sum of individual returns reconstructs the
portfolio's own weighted-average return exactly -- this is arithmetic
identity, not an approximation, and is checked by a dedicated
reconciliation test (analytics/tests/test_attribution.py).
"""
from __future__ import annotations

import pandas as pd


def asset_contribution(
    start_weights: pd.Series, asset_returns: pd.Series
) -> pd.Series:
    """
    Definition: contribution_i = start_weight_i * return_i

    Both inputs indexed by the same key (e.g. ticker). Assets missing
    from either input are dropped from the result (an asset with no
    starting weight cannot contribute, and one with no computable
    return cannot be attributed).
    """
    aligned = pd.concat([start_weights, asset_returns], axis=1).dropna()
    aligned.columns = ["weight", "return"]
    return aligned["weight"] * aligned["return"]


def sector_contribution(
    start_weights: pd.Series, asset_returns: pd.Series, sector_map: pd.Series
) -> pd.Series:
    """Definition: sector_contribution = sum(asset_contribution_i for
    assets i in that sector). sector_map indexes ticker -> sector."""
    contrib = asset_contribution(start_weights, asset_returns)
    sectors = sector_map.reindex(contrib.index)
    return contrib.groupby(sectors).sum()


def total_portfolio_return(start_weights: pd.Series, asset_returns: pd.Series) -> float:
    """Definition: sum(asset_contribution_i) -- the reconciliation
    target every contribution breakdown must sum back to exactly."""
    return float(asset_contribution(start_weights, asset_returns).sum())
