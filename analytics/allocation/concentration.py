"""Concentration metrics: how much of the portfolio sits in its largest
positions, and the Herfindahl-Hirschman Index (HHI) as a single-number
concentration score.
"""
from __future__ import annotations

import pandas as pd


def largest_holding_pct(weights: pd.Series) -> float:
    """Definition: max(w_i) across all holdings."""
    return float(weights.max()) if not weights.empty else float("nan")


def top_n_pct(weights: pd.Series, n: int) -> float:
    """Definition: sum of the n largest weights."""
    return float(weights.sort_values(ascending=False).head(n).sum())


def herfindahl_hirschman_index(weights: pd.Series) -> float:
    """
    Definition:
        HHI = sum(w_i ** 2) for i in holdings, where sum(w_i) = 1

    Ranges from 1/n (n equal-weighted holdings, most diversified) to 1
    (a single holding, maximally concentrated). Weights must sum to 1;
    this function does not renormalize, so a caller must pass a proper
    weight vector (e.g. from allocation.asset_allocation).
    """
    return float((weights ** 2).sum())
