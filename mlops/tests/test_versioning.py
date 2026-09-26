import pandas as pd
import pytest

from mlops.versioning import compute_dataset_version, compute_feature_version


def _sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["TCS.NS", "TCS.NS", "RELIANCE.NS"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "close": [100.0, 101.0, 200.0],
            "volume": [1000, 1100, 2000],
        }
    )


def test_dataset_version_is_deterministic_for_identical_content():
    v1 = compute_dataset_version(_sample_prices())
    v2 = compute_dataset_version(_sample_prices())
    assert v1.version_hash == v2.version_hash


def test_dataset_version_changes_when_content_changes():
    original = compute_dataset_version(_sample_prices())
    changed_prices = _sample_prices()
    changed_prices.loc[0, "close"] = 999.0
    changed = compute_dataset_version(changed_prices)
    assert original.version_hash != changed.version_hash


def test_dataset_version_is_row_order_independent():
    prices = _sample_prices()
    shuffled = prices.iloc[::-1].reset_index(drop=True)
    v_original = compute_dataset_version(prices)
    v_shuffled = compute_dataset_version(shuffled)
    assert v_original.version_hash == v_shuffled.version_hash


def test_dataset_version_metadata_is_correct():
    v = compute_dataset_version(_sample_prices())
    assert set(v.tickers) == {"TCS.NS", "RELIANCE.NS"}
    assert v.row_count == 3
    assert v.start_date == "2024-01-01"
    assert v.end_date == "2024-01-02"


def test_feature_version_is_deterministic():
    v1 = compute_feature_version()
    v2 = compute_feature_version()
    assert v1 == v2
    assert len(v1) == 16  # truncated sha256 hex digest, per compute_feature_version
