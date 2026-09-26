import pandas as pd
import pytest

from ml.splits import chronological_split


def test_split_is_strictly_chronological_no_date_overlap():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=1000))
    masks = chronological_split(dates, train_frac=0.70, val_frac=0.15)

    train_dates = dates[masks["train"]]
    val_dates = dates[masks["val"]]
    test_dates = dates[masks["test"]]

    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()

    # No date is dropped or double-counted across the three splits.
    assert masks["train"].sum() + masks["val"].sum() + masks["test"].sum() == len(dates)
    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["val"] & masks["test"]).any()


def test_split_fractions_approximately_correct():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=1000))
    masks = chronological_split(dates, train_frac=0.70, val_frac=0.15)
    assert masks["train"].sum() == pytest.approx(700, abs=2)
    assert masks["val"].sum() == pytest.approx(150, abs=2)
    assert masks["test"].sum() == pytest.approx(150, abs=2)


def test_split_works_with_repeated_dates_multiple_tickers():
    # Two tickers sharing the same date index (the real multi-ticker
    # case): the split must still be a pure function of the DATE, so
    # both tickers' rows for a given date land in the same split.
    single_dates = pd.bdate_range("2020-01-01", periods=500)
    dates = pd.Series(list(single_dates) * 2)
    masks = chronological_split(dates, train_frac=0.70, val_frac=0.15)

    train_dates = set(dates[masks["train"]])
    val_dates = set(dates[masks["val"]])
    assert max(train_dates) < min(val_dates)
