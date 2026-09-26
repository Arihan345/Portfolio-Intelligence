"""Dataset and feature versioning: makes "what data trained model
vX.Y.Z" and "what feature logic trained model vX.Y.Z" reconstructible
later, per this phase's requirement.

DATASET VERSION: a hash of the price history's actual content
(ticker, date, close, volume values) plus explicit metadata (tickers,
date range, row count). The hash means two fetches that happen to
return byte-identical data get the SAME version tag (so re-running
training on an unchanged cache doesn't manufacture a fake new dataset
version), while any change to the underlying prices -- a re-fetch that
picks up a new trading day, a data correction -- changes the hash.

FEATURE VERSION: a hash of ml/features/features.py's own source code.
If the feature engineering logic changes (a new feature added, a
formula fixed), this hash changes, so a model trained against the old
feature definitions is never confused with one trained against new
ones, even if someone forgets to bump a manual version string.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetVersion:
    version_hash: str
    tickers: list[str]
    start_date: str
    end_date: str
    row_count: int

    def as_dict(self) -> dict:
        return {
            "dataset_version_hash": self.version_hash,
            "dataset_tickers": ",".join(sorted(self.tickers)),
            "dataset_start_date": self.start_date,
            "dataset_end_date": self.end_date,
            "dataset_row_count": self.row_count,
        }


def compute_dataset_version(price_history: pd.DataFrame) -> DatasetVersion:
    """price_history: the long-format DataFrame from
    ml.data.fetch_price_history (columns include ticker, date, close,
    volume). Hash is computed over a canonical (sorted, fixed-column)
    CSV serialization so row order never affects the hash, only content."""
    canonical = price_history[["ticker", "date", "close", "volume"]].sort_values(
        ["ticker", "date"]
    )
    payload = canonical.to_csv(index=False).encode("utf-8")
    version_hash = hashlib.sha256(payload).hexdigest()[:16]

    return DatasetVersion(
        version_hash=version_hash,
        tickers=sorted(price_history["ticker"].unique().tolist()),
        start_date=str(price_history["date"].min()),
        end_date=str(price_history["date"].max()),
        row_count=len(price_history),
    )


def compute_feature_version() -> str:
    """Hash of every function's source code in ml.features.features,
    concatenated in a fixed (alphabetical) order -- catches any edit to
    a feature's formula, not just to whichever function happens to be
    imported first."""
    from ml.features import features as features_module

    function_names = sorted(
        name
        for name, obj in inspect.getmembers(features_module, inspect.isfunction)
        if obj.__module__ == features_module.__name__
    )
    source = "".join(inspect.getsource(getattr(features_module, name)) for name in function_names)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
