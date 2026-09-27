"""Strictly chronological train/validation/test split.

Definition: sort all distinct observation dates ascending, then cut at
the train_frac and (train_frac + val_frac) points. Train gets the
earliest dates, validation the next block, test the most recent block.
No shuffling, no random sampling -- a random split would let the model
train on dates AFTER some of its validation/test dates, which is a
leakage bug for any time-series problem (the model could learn
regime-specific patterns from the future and get credit for
"predicting" a past it was trained on in relative time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def train_cutoff_date(dates: pd.Series, train_frac: float = 0.70):
    """The last date belonging to the training split, at `train_frac`
    of the sorted distinct dates. Factored out from chronological_split
    so callers that need to know "where training ends" BEFORE the full
    train/val/test split exists (e.g. ml.dataset's label-threshold
    computation, which must only look at training-period data) share
    the exact same cutoff logic rather than risking a second,
    slightly-different definition of "the training period" drifting in
    alongside it.
    """
    unique_dates = np.sort(dates.unique())
    n = len(unique_dates)
    return unique_dates[int(n * train_frac) - 1]


def chronological_split(
    dates: pd.Series, train_frac: float = 0.70, val_frac: float = 0.15
) -> dict[str, np.ndarray]:
    """Returns boolean masks {'train', 'val', 'test'} aligned to `dates`."""
    unique_dates = np.sort(dates.unique())
    n = len(unique_dates)
    train_cutoff = train_cutoff_date(dates, train_frac)
    val_cutoff = unique_dates[int(n * (train_frac + val_frac)) - 1]

    train_mask = (dates <= train_cutoff).to_numpy()
    val_mask = ((dates > train_cutoff) & (dates <= val_cutoff)).to_numpy()
    test_mask = (dates > val_cutoff).to_numpy()

    return {"train": train_mask, "val": val_mask, "test": test_mask}
