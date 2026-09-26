import pandas as pd

from ml.baseline import predict_baseline


def test_baseline_predicts_persistence_of_current_regime():
    trailing_vol = pd.Series([0.10, 0.20, 0.30, 0.40])
    threshold = 0.25
    result = predict_baseline(trailing_vol, threshold)
    assert list(result) == [0, 0, 1, 1]


def test_baseline_propagates_nan_from_warmup_period():
    trailing_vol = pd.Series([float("nan"), float("nan"), 0.30])
    result = predict_baseline(trailing_vol, threshold=0.25)
    assert result.isna().iloc[0] and result.isna().iloc[1]
    assert result.iloc[2] == 1
