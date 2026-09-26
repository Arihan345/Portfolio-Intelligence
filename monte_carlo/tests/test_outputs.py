import numpy as np
import pytest

from monte_carlo.outputs import (
    drawdown_statistics,
    percentile_bands,
    probability_of_exceeding,
    probability_of_loss,
    simulated_var_es,
)


def test_probability_of_loss_and_exceeding_hand_case():
    terminal_values = np.array([90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    assert probability_of_loss(terminal_values, initial_value=100.0) == pytest.approx(2 / 7)
    assert probability_of_exceeding(terminal_values, target_value=110.0) == pytest.approx(2 / 7)


def test_percentile_bands_median_hand_case():
    # 7 sorted values, indices 0..6: the median (50th percentile) is
    # the middle element at index 3 (the 4th value), which is 105.0 --
    # not 100.0 at index 2, a mistake caught by running this for real.
    terminal_values = np.array([90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    bands = percentile_bands(terminal_values)
    assert bands["p50"] == pytest.approx(105.0)
    assert set(bands.keys()) == {"p5", "p25", "p50", "p75", "p95"}


def test_simulated_var_es_matches_analytics_risk_directly():
    # Confirms outputs.py is a thin pass-through to analytics.risk, not
    # a divergent reimplementation: same inputs must give the same
    # numbers as calling analytics.risk directly.
    from analytics.risk.risk import conditional_var, historical_var
    import pandas as pd

    terminal_values = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    initial_value = 100.0
    result = simulated_var_es(terminal_values, initial_value, confidence=0.75)

    terminal_returns = pd.Series(terminal_values / initial_value - 1)
    assert result["var"] == pytest.approx(historical_var(terminal_returns, 0.75))
    assert result["expected_shortfall"] == pytest.approx(conditional_var(terminal_returns, 0.75))


def test_drawdown_statistics_hand_case():
    # Path 1 matches the exact -25% drawdown hand case from
    # analytics/tests/test_risk.py::test_max_drawdown_hand_case.
    # Path 2 is flat (never draws down).
    paths = np.array(
        [
            [100.0, 120.0, 90.0, 110.0, 130.0],
            [100.0, 100.0, 100.0, 100.0, 100.0],
        ]
    )
    result = drawdown_statistics(paths)
    assert result["mean_max_drawdown"] == pytest.approx((-0.25 + 0.0) / 2)
    assert result["median_max_drawdown"] == pytest.approx(-0.125)
    assert result["best_max_drawdown"] == pytest.approx(0.0)
    # Cross-check worst_5pct against numpy's own percentile on the same
    # 2-value array, rather than hand-deriving the interpolation.
    expected_worst = float(np.percentile([-0.25, 0.0], 5))
    assert result["worst_5pct_max_drawdown"] == pytest.approx(expected_worst)
