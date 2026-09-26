import math

import pandas as pd
import pytest

from analytics.benchmark.benchmark import (
    excess_return,
    information_ratio,
    tracking_error,
)
from analytics.risk.risk import TRADING_DAYS_PER_YEAR


def test_excess_return_hand_case():
    p = pd.Series([0.02, 0.01, 0.03])
    b = pd.Series([0.01, 0.01, 0.01])
    result = excess_return(p, b)
    assert list(result) == pytest.approx([0.01, 0.00, 0.02])


def test_tracking_error_and_information_ratio_hand_case():
    p = pd.Series([0.02, 0.01, 0.03])
    b = pd.Series([0.01, 0.01, 0.01])
    excess = [0.01, 0.00, 0.02]

    mean_excess = sum(excess) / 3
    variance = sum((x - mean_excess) ** 2 for x in excess) / (3 - 1)  # ddof=1
    expected_te = math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR)
    expected_ir = (mean_excess * TRADING_DAYS_PER_YEAR) / expected_te

    assert tracking_error(p, b) == pytest.approx(expected_te)
    assert information_ratio(p, b) == pytest.approx(expected_ir)
