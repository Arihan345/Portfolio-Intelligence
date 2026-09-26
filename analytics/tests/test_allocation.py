import pandas as pd
import pytest

from analytics.allocation.allocation import (
    asset_allocation,
    cash_allocation,
    sector_allocation,
)


def _sample_allocation():
    return pd.DataFrame(
        {
            "ticker": ["TCS.NS", "RELIANCE.NS"],
            "sector": ["Technology", "Energy"],
            "market_value_inr": [700.0, 300.0],
        }
    )


def test_asset_allocation_hand_case():
    result = asset_allocation(_sample_allocation())
    assert result["TCS.NS"] == pytest.approx(0.7)
    assert result["RELIANCE.NS"] == pytest.approx(0.3)


def test_sector_allocation_hand_case():
    result = sector_allocation(_sample_allocation())
    assert result["Technology"] == pytest.approx(0.7)
    assert result["Energy"] == pytest.approx(0.3)


def test_cash_allocation_hand_case():
    result = cash_allocation(holdings_market_value=800.0, cash_balance=200.0)
    assert result["weight_holdings"] == pytest.approx(0.8)
    assert result["weight_cash"] == pytest.approx(0.2)
