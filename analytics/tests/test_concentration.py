import pandas as pd
import pytest

from analytics.allocation.concentration import (
    herfindahl_hirschman_index,
    largest_holding_pct,
    top_n_pct,
)


def test_hhi_matches_brief_example():
    # 55% TCS, 45% Reliance -> HHI = 0.55^2 + 0.45^2 = 0.3025 + 0.2025 = 0.505
    weights = pd.Series({"TCS.NS": 0.55, "RELIANCE.NS": 0.45})
    assert herfindahl_hirschman_index(weights) == pytest.approx(0.505)


def test_hhi_bounds_equal_weight_and_single_holding():
    n = 4
    equal_weights = pd.Series([1 / n] * n)
    assert herfindahl_hirschman_index(equal_weights) == pytest.approx(1 / n)

    single_holding = pd.Series([1.0])
    assert herfindahl_hirschman_index(single_holding) == pytest.approx(1.0)


def test_largest_holding_and_top_n():
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05})
    assert largest_holding_pct(weights) == pytest.approx(0.5)
    assert top_n_pct(weights, 2) == pytest.approx(0.8)
    assert top_n_pct(weights, 3) == pytest.approx(0.95)
