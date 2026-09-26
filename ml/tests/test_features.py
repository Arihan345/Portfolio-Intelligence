import numpy as np
import pandas as pd
import pytest

from ml.features.features import (
    drawdown_depth,
    lagged_returns,
    moving_averages,
    rolling_volatility,
    volume_zscore,
)


def _prices(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "close": closes,
            "volume": volumes or [1.0] * n,
        }
    )


def test_lagged_return_hand_case():
    # close = [100, 110, 121, 100]: 1-day return at row2 = 121/110-1=0.10
    prices = _prices([100.0, 110.0, 121.0, 100.0])
    result = lagged_returns(prices, lags=(1,))
    assert result["return_lag_1"].iloc[1] == pytest.approx(0.10)
    assert result["return_lag_1"].iloc[2] == pytest.approx(0.10)
    assert result["return_lag_1"].iloc[3] == pytest.approx(100 / 121 - 1)
    assert np.isnan(result["return_lag_1"].iloc[0])  # no prior day


def test_moving_average_hand_case():
    # 3-day MA at the last row of [10, 20, 30] = 20; price_to_ma = 30/20=1.5
    prices = _prices([10.0, 20.0, 30.0])
    result = moving_averages(prices, windows=(3,))
    assert result["ma_3d"].iloc[2] == pytest.approx(20.0)
    assert result["price_to_ma_3d_ratio"].iloc[2] == pytest.approx(1.5)
    assert np.isnan(result["ma_3d"].iloc[1])  # only 2 rows available, window needs 3


def test_rolling_volatility_hand_case():
    # Constant returns -> zero volatility (a clean, hand-verifiable
    # degenerate case): close doubling by the same 10% each day.
    closes = [100 * (1.10 ** i) for i in range(15)]
    prices = _prices(closes)
    result = rolling_volatility(prices, windows=(5,))
    assert result["volatility_5d"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_volume_zscore_hand_case():
    # 4 days of volume [10, 10, 10, 20], window=3: at the last row the
    # trailing 3 = [10, 10, 20], mean=13.333, std(ddof=1)=5.7735
    # z = (20 - 13.333) / 5.7735 = 1.1547
    prices = _prices([1.0, 1.0, 1.0, 1.0], volumes=[10.0, 10.0, 10.0, 20.0])
    result = volume_zscore(prices, window=3)
    window_vals = [10.0, 10.0, 20.0]
    expected_mean = sum(window_vals) / 3
    expected_std = pd.Series(window_vals).std(ddof=1)
    expected_z = (20.0 - expected_mean) / expected_std
    assert result["volume_zscore_3d"].iloc[-1] == pytest.approx(expected_z)


def test_drawdown_depth_hand_case():
    # Trailing 3-day peak at the last row of [100, 130, 90]: peak=130,
    # drawdown = (90-130)/130 = -0.30769...
    prices = _prices([100.0, 130.0, 90.0])
    result = drawdown_depth(prices, window=3)
    assert result["drawdown_depth_3d"].iloc[-1] == pytest.approx((90.0 - 130.0) / 130.0)
    # A new peak must show exactly 0 drawdown.
    prices2 = _prices([100.0, 90.0, 130.0])
    result2 = drawdown_depth(prices2, window=3)
    assert result2["drawdown_depth_3d"].iloc[-1] == pytest.approx(0.0)
