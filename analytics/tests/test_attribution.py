import pandas as pd
import pytest

from analytics.attribution.attribution import (
    asset_contribution,
    sector_contribution,
    total_portfolio_return,
)


def test_asset_contribution_hand_case():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    returns = pd.Series({"A": 0.10, "B": -0.05})
    contrib = asset_contribution(weights, returns)
    assert contrib["A"] == pytest.approx(0.06)
    assert contrib["B"] == pytest.approx(-0.02)


def test_contribution_reconciles_exactly_to_total_return():
    """This is the mandatory reconciliation check: contribution_i =
    weight_i * return_i summed over all assets MUST equal the
    portfolio's own weighted-average return -- not approximately, but
    exactly, because both sides are the same arithmetic expression."""
    weights = pd.Series({"A": 0.6, "B": 0.4})
    returns = pd.Series({"A": 0.10, "B": -0.05})

    total_from_contributions = total_portfolio_return(weights, returns)
    total_from_weighted_average = float((weights * returns).sum())

    assert total_from_contributions == pytest.approx(total_from_weighted_average, abs=1e-12)
    assert total_from_contributions == pytest.approx(0.04)


def test_contribution_reconciles_with_more_assets_and_sectors():
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    returns = pd.Series({"A": 0.08, "B": -0.10, "C": 0.15})
    sectors = pd.Series({"A": "Tech", "B": "Energy", "C": "Tech"})

    total = total_portfolio_return(weights, returns)
    expected_total = 0.5 * 0.08 + 0.3 * (-0.10) + 0.2 * 0.15
    assert total == pytest.approx(expected_total)

    by_sector = sector_contribution(weights, returns, sectors)
    # Tech = A + C contributions, Energy = B contribution
    assert by_sector["Tech"] == pytest.approx(0.5 * 0.08 + 0.2 * 0.15)
    assert by_sector["Energy"] == pytest.approx(0.3 * (-0.10))
    # Sector contributions must themselves reconcile to the same total.
    assert by_sector.sum() == pytest.approx(total)


def test_contribution_drops_assets_missing_from_either_input():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    returns = pd.Series({"A": 0.10})  # B has no computable return
    contrib = asset_contribution(weights, returns)
    assert list(contrib.index) == ["A"]
