import pandas as pd
import pytest

from ml.walk_forward import walk_forward_splits


def test_walk_forward_splits_are_expanding_and_contiguous():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=600))
    folds = walk_forward_splits(dates, n_folds=5)

    assert len(folds) == 5
    prev_train_end = None
    for i, fold in enumerate(folds):
        train_dates = dates[fold["train"]]
        test_dates = dates[fold["test"]]

        # Train is strictly before test (rolling-origin, no leakage).
        assert train_dates.max() < test_dates.min()

        # Expanding window: each fold's train set is a superset of the
        # previous fold's (same start, later end).
        if prev_train_end is not None:
            assert train_dates.max() > prev_train_end
        prev_train_end = train_dates.max()

        # No overlap between this fold's train and test.
        assert not (fold["train"] & fold["test"]).any()


def test_walk_forward_splits_test_blocks_are_sequential_and_non_overlapping():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=600))
    folds = walk_forward_splits(dates, n_folds=5)

    test_ranges = [(dates[f["test"]].min(), dates[f["test"]].max()) for f in folds]
    for (start1, end1), (start2, end2) in zip(test_ranges, test_ranges[1:]):
        assert end1 < start2, "consecutive fold test windows must not overlap"
