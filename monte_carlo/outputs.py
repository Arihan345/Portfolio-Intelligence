"""Summarizes a simulation's terminal-value distribution and full paths
into reportable metrics. Pure functions: take numpy arrays, return
plain numbers/dicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.risk.risk import conditional_var, historical_var


def percentile_bands(
    terminal_values: np.ndarray, percentiles: tuple[float, ...] = (5, 25, 50, 75, 95)
) -> dict:
    """Definition: the requested percentiles of the terminal-value
    distribution (e.g. the 5th percentile is the value below which only
    5% of simulated outcomes fell)."""
    values = np.percentile(terminal_values, percentiles)
    return {f"p{int(p)}": float(v) for p, v in zip(percentiles, values)}


def probability_of_loss(terminal_values: np.ndarray, initial_value: float) -> float:
    """Definition: fraction of simulations ending below the starting value."""
    return float(np.mean(terminal_values < initial_value))


def probability_of_exceeding(terminal_values: np.ndarray, target_value: float) -> float:
    """Definition: fraction of simulations ending above a target value."""
    return float(np.mean(terminal_values > target_value))


def simulated_var_es(
    terminal_values: np.ndarray, initial_value: float, confidence: float = 0.95
) -> dict:
    """Simulated VaR/Expected Shortfall on TERMINAL returns.

    Reuses analytics.risk.historical_var / conditional_var rather than
    reimplementing the percentile/tail-mean formulas: those functions
    are defined purely in terms of a return sample (any sample -- there
    is nothing time-series-specific about a percentile or a tail mean),
    so applying them to the cross-sectional distribution of simulated
    terminal returns is the same math as applying them to a historical
    daily-return time series, just a different sample.
    """
    terminal_returns = pd.Series(terminal_values / initial_value - 1)
    return {
        "var": historical_var(terminal_returns, confidence),
        "expected_shortfall": conditional_var(terminal_returns, confidence),
    }


def drawdown_statistics(paths: np.ndarray) -> dict:
    """
    Definition (per simulated path, same formula as
    analytics.risk.max_drawdown, applied here with vectorized numpy
    instead of a pandas Series per path -- calling the pandas version
    10,000 times would dominate runtime with per-call overhead for a
    formula that is a three-line numpy reduction):
        running_peak_t = max(path_0 .. path_t)
        drawdown_t     = (path_t - running_peak_t) / running_peak_t
        path_max_drawdown = min(drawdown_t) over the path

    Returns summary statistics of path_max_drawdown across all
    simulated paths: mean, median, and worst-case (5th percentile, the
    deepest drawdowns across simulations).
    """
    running_peak = np.maximum.accumulate(paths, axis=1)
    drawdown = (paths - running_peak) / running_peak
    max_dd_per_path = drawdown.min(axis=1)

    return {
        "mean_max_drawdown": float(max_dd_per_path.mean()),
        "median_max_drawdown": float(np.median(max_dd_per_path)),
        "worst_5pct_max_drawdown": float(np.percentile(max_dd_per_path, 5)),
        "best_max_drawdown": float(max_dd_per_path.max()),
    }
