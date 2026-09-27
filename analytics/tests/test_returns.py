import math
from datetime import date

import pandas as pd
import pytest

from analytics.performance.returns import (
    annualize_return,
    cagr,
    capital_summary,
    daily_returns_from_value_and_flows,
    implied_market_value_cash_flows_by_day,
    net_external_cash_flows_by_day,
    periodic_returns,
    realized_pnl_summary,
    rolling_returns,
    time_weighted_return,
    xirr,
)


def test_capital_summary_hand_case():
    result = capital_summary(invested_capital=1000.0, market_value=1200.0)
    assert result["absolute_return"] == 200.0
    assert result["pct_return"] == pytest.approx(0.2)


def test_cagr_hand_case():
    # 730 calendar days (2022 and 2023 are both non-leap) is 730/365.25
    # years, not exactly 2 -- compute the expected value the same way
    # by hand rather than guessing a rounded literal.
    start, end = date(2022, 1, 1), date(2024, 1, 1)
    days = (end - start).days
    assert days == 730
    expected = (121.0 / 100.0) ** (1 / (days / 365.25)) - 1
    result = cagr(100.0, 121.0, start, end)
    assert result == pytest.approx(expected)


def test_cagr_exact_four_years():
    # 2020-01-01 to 2024-01-01 spans exactly 1461 days (2020 and 2024
    # are leap years but only 2020 falls inside the range), and
    # 1461 / 365.25 = 4.0 exactly -- the one calendar span short enough
    # to reason about by hand that divides evenly into the 365.25
    # days-per-year convention. 100 growing at 10%/year for 4 years
    # compounds to 100 * 1.1**4 = 146.41.
    start, end = date(2020, 1, 1), date(2024, 1, 1)
    days = (end - start).days
    assert days == 1461
    assert days / 365.25 == 4.0
    result = cagr(100.0, 146.41, start, end)
    assert result == pytest.approx(0.10, abs=1e-9)


def test_periodic_returns_monthly_compounding():
    # Two days of +10% each within the same month must compound to
    # 1.1 * 1.1 - 1 = 0.21, not 0.20 (simple sum).
    idx = pd.to_datetime(["2024-01-05", "2024-01-10"])
    r = pd.Series([0.10, 0.10], index=idx)
    monthly = periodic_returns(r, freq="ME")
    assert monthly.iloc[0] == pytest.approx(0.21)


def test_rolling_returns_hand_case():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    r = pd.Series([0.10, 0.10, 0.10], index=idx)
    rolling = rolling_returns(r, window=2)
    assert math.isnan(rolling.iloc[0])
    assert rolling.iloc[1] == pytest.approx(0.21)
    assert rolling.iloc[2] == pytest.approx(0.21)


def test_twr_no_external_flows_equals_simple_return():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    nav = pd.Series([100.0, 110.0, 121.0], index=idx)
    assert time_weighted_return(nav) == pytest.approx(0.21)


def test_twr_removes_deposit_distortion():
    # Day0: NAV 100. Day1: a $100 deposit arrives AND the market is
    # flat, so NAV becomes 200 purely from the deposit, not performance.
    # Day2: market grows 10% on the (now) 200 NAV -> NAV 220.
    # Naive return (200->220 ignoring the deposit) would look like a
    # tiny gain from 100->220 (=120%), which is wrong -- that's the
    # deposit, not performance. TWR must show a much smaller number:
    # HPR1 = (200-100-100)/100 = 0%, HPR2 = (220-200)/200 = 10%
    # TWR = 1.00 * 1.10 - 1 = 10%
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    nav = pd.Series([100.0, 200.0, 220.0], index=idx)
    flows = pd.Series([100.0], index=[idx[1]])
    result = time_weighted_return(nav, flows)
    assert result == pytest.approx(0.10)


def test_twr_cagr_unaffected_by_tiny_denominator_extreme_period():
    # Regression for a real, confirmed bug: a portfolio's tiny-base
    # position (bought 2 @ INR 15, sold months later 2 @ INR 45 -- a
    # 200% move on a ~INR 30 cost basis, the synthetic fixture's
    # HDFCBANK case at tests/fixtures/synthetic_edge_case_portfolio.csv)
    # produced a portfolio-level CAGR of 524.95% and TWR of 8681.84%
    # when daily returns were computed as a naive value(t)/value(t-1)
    # ratio -- the BUY's and SELL's own SIZE got read as investment
    # return instead of capital moving in/out. A single extreme-
    # percentage period on a small denominator must never blow up the
    # portfolio-level CAGR/TWR once cash flows are correctly backed out.
    idx = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-06-01", "2024-06-02", "2026-01-01"]
    )
    # Day0: an established, normal-sized base (unrelated holdings).
    # Day1: BUY 2 HDFCBANK @ 15 = 30 (external inflow, no market move).
    # Day2 (~5 months later): SELL 2 HDFCBANK @ 45 = 90 net proceeds
    #   (external outflow) -- the other holdings alone are unchanged.
    # Day3: normal day, no flows.
    # Day4 (~2 years out): modest overall growth to 1200.
    value = pd.Series([1000.0, 1030.0, 1000.0, 1005.0, 1200.0], index=idx)
    flows = pd.Series([30.0, -90.0], index=[idx[1], idx[2]])

    daily = daily_returns_from_value_and_flows(value, flows).dropna()
    # Neither trade-adjacent period should be anywhere near the raw 200%
    # move on HDFCBANK's own tiny position -- each reflects only the
    # REST of the portfolio's real movement once the flow is backed out.
    assert daily.loc[idx[1]] == pytest.approx(0.0)  # BUY day: pure inflow, no fake gain
    assert abs(daily.loc[idx[2]]) < 0.10  # SELL day: no fake loss from cash leaving

    twr = time_weighted_return(value, flows)
    assert twr < 1.0  # nowhere near the old bug's 86.8184 (8681.84%)

    days = (idx[-1] - idx[0]).days
    cagr_value = annualize_return(twr, days)
    assert cagr_value < 1.0  # nowhere near the old bug's 5.2495 (524.95%)
    assert -0.5 < cagr_value < 1.0  # a plausible multi-year annualized rate


def test_net_external_cash_flows_by_day_buy_and_sell_signs():
    tx = pd.DataFrame(
        {
            "txn_date": pd.to_datetime(["2024-01-02", "2024-06-01"]),
            "transaction_type": ["BUY", "SELL"],
            "quantity": [2.0, 2.0],
            "price_inr": [15.0, 45.0],
            "fees": [0.0, 0.0],
            "tax": [0.0, 0.0],
        }
    )
    flows = net_external_cash_flows_by_day(tx)
    assert flows[pd.Timestamp("2024-01-02")] == pytest.approx(30.0)
    assert flows[pd.Timestamp("2024-06-01")] == pytest.approx(-90.0)


def test_capital_summary_fully_exited_portfolio_shows_realized_pnl():
    # Regression for a real, confirmed bug: a fully-exited synthetic
    # portfolio (3 tickers, all bought then fully sold: total buys
    # 2,165, total sells 2,690, a real 525 realized profit) showed
    # Invested Capital as -525 (net cash received, not a cost basis)
    # and % Return as a nonsensical -100.00% (absolute_return/0 falling
    # back to a hardcoded value). With invested_capital correctly at 0
    # (nothing is currently held) and a realized_summary reflecting the
    # real trading history, Absolute Return/% Return must reflect the
    # real 525 profit relative to real capital deployed, not collapse
    # to 0/NaN or a fake -100%.
    tx = pd.DataFrame(
        {
            "txn_date": pd.to_datetime(
                ["2023-01-05", "2023-03-05", "2023-02-01", "2023-04-01", "2023-02-15", "2023-05-15"]
            ),
            "ticker": ["TCS.NS", "TCS.NS", "INFY.NS", "INFY.NS", "WIPRO.NS", "WIPRO.NS"],
            "transaction_type": ["BUY", "SELL", "BUY", "SELL", "BUY", "SELL"],
            "quantity": [5.0, 5.0, 10.0, 10.0, 1.0, 1.0],
            "price_inr": [100.0, 140.0, 80.0, 100.0, 865.0, 990.0],
            "fees": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "tax": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    realized = realized_pnl_summary(tx)
    assert realized.realized_pnl == pytest.approx(525.0)
    assert realized.total_capital_deployed == pytest.approx(2165.0)

    result = capital_summary(invested_capital=0.0, market_value=0.0, realized_summary=realized)
    assert result["invested_capital"] == 0.0
    assert result["absolute_return"] == pytest.approx(525.0)
    assert result["pct_return"] == pytest.approx(525.0 / 2165.0)


def test_capital_summary_normal_case_unaffected_by_realized_summary_arg():
    # A portfolio with real current holdings (invested_capital > 0)
    # must behave EXACTLY as before -- the realized-P&L fallback only
    # ever applies in the fully-exited (invested_capital == 0) case, so
    # the already-validated real-portfolio ground truth (Invested
    # 15,696 / Market Value 17,372 / Absolute Return 1,675 / 10.67%)
    # must never be touched by this fix.
    tx = pd.DataFrame(
        {
            "txn_date": pd.to_datetime(["2023-01-05"]),
            "ticker": ["TCS.NS"],
            "transaction_type": ["BUY"],
            "quantity": [5.0],
            "price_inr": [100.0],
            "fees": [0.0],
            "tax": [0.0],
        }
    )
    realized = realized_pnl_summary(tx)
    result = capital_summary(invested_capital=1000.0, market_value=1200.0, realized_summary=realized)
    assert result["absolute_return"] == 200.0
    assert result["pct_return"] == pytest.approx(0.2)


def test_realized_pnl_summary_nets_cross_exchange_pair():
    # Same base-symbol folding convention as int_daily_holdings.sql's
    # canonical_asset: a BUY on one exchange and a SELL on the other
    # for the same underlying company must net as one position, not
    # read as an unmatched, impossible SELL.
    tx = pd.DataFrame(
        {
            "txn_date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "ticker": ["MOREPENLAB.BO", "MOREPENLAB.NS"],
            "transaction_type": ["BUY", "SELL"],
            "quantity": [10.0, 10.0],
            "price_inr": [50.0, 60.0],
            "fees": [0.0, 0.0],
            "tax": [0.0, 0.0],
        }
    )
    realized = realized_pnl_summary(tx)
    assert realized.realized_pnl == pytest.approx(100.0)
    assert realized.total_capital_deployed == pytest.approx(500.0)


def test_implied_market_value_cash_flows_credits_first_ever_buy():
    # Regression for a real, confirmed bug: groupby("ticker").shift(1)
    # correctly has no prior row on a ticker's own first-ever date (the
    # asset didn't exist before its first transaction), but leaving
    # that NaN unfilled made qty_delta also NaN there, and
    # NaN.abs() > 1e-9 is always False -- silently crediting a brand
    # new position's real first BUY as a ZERO cash flow instead of its
    # real cost. Confirmed for real: a portfolio's first-ever purchase
    # of a new ticker (a single INR 15,117 DIXON.NS share) was credited
    # as INR 0, making that day's holdings-value jump read as pure,
    # uncosted gain -- inflating a real portfolio's TWR from ~38% to a
    # nonsensical ~11,659%, since EVERY ticker's first buy hit this.
    ap = pd.DataFrame(
        {
            "ticker": ["X.NS"],
            "as_of_date": pd.to_datetime(["2024-01-01"]),
            "quantity_held": [10.0],
            "market_value_inr": [1000.0],
        }
    )
    cf = implied_market_value_cash_flows_by_day(ap)
    assert cf[pd.Timestamp("2024-01-01")] == pytest.approx(1000.0)


def test_implied_market_value_cash_flows_handles_price_basis_mismatch():
    # Regression for a real, confirmed bug: a synthetic test portfolio
    # used a ticker string ("BETA.NS") that coincidentally has its own
    # REAL yfinance listing at a real price (~INR 705/share) utterly
    # unrelated to the fixture's fictional ~INR 58/share sale price.
    # net_external_cash_flows_by_day (transaction-price-based) valued
    # the exit's cash flow at the fictional price while holdings_value
    # (market_value_inr) was priced at the real close -- a ~13x
    # mismatch that produced a fake -92% single-day "return" on the
    # exit day alone. implied_market_value_cash_flows_by_day instead
    # derives the flow from the SAME real-price basis as the value
    # series, so a full exit's cash flow exactly equals the prior day's
    # real market value regardless of what the original trade's own
    # recorded price was -- giving a clean ~0% return on the exit day.
    ap = pd.DataFrame(
        {
            "ticker": ["BETA.NS", "BETA.NS"],
            "as_of_date": pd.to_datetime(["2023-05-09", "2023-05-10"]),
            "quantity_held": [5.25, 0.0],
            "market_value_inr": [3701.25, 0.0],
        }
    )
    cf = implied_market_value_cash_flows_by_day(ap)
    assert cf[pd.Timestamp("2023-05-10")] == pytest.approx(-3701.25)

    daily = daily_returns_from_value_and_flows(
        pd.Series([3701.25, 0.0], index=pd.to_datetime(["2023-05-09", "2023-05-10"])), cf
    )
    assert daily.loc[pd.Timestamp("2023-05-10")] == pytest.approx(0.0)


def test_twr_cagr_correct_sign_across_three_staggered_full_exits():
    # Regression for the exact real bug class this fixes, using the
    # same shape as the synthetic ALPHA/BETA/GAMMA fixture (three
    # positions, each independently bought then fully exited on a
    # different date) but with a clean, self-manufactured asset-
    # performance series (no live yfinance dependency, so this test is
    # deterministic) -- a real net profit across the whole history must
    # never compute to a negative CAGR/TWR, regardless of how many
    # separate full-exit transitions the portfolio went through.
    dates = pd.to_datetime(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    )
    # Day1: buy A (cost 1000). Day2: buy B (cost 500) alongside A's
    # normal price growth. Day3: A fully exits at a real profit.
    # Day4: normal day. Day5: buy C (cost 300). Day6: B and C both
    # fully exit at a real profit -- three staggered full exits, a
    # real net profit throughout.
    ap = pd.DataFrame(
        {
            "ticker": (
                ["A"] * 6
                + ["B"] * 5
                + ["C"] * 2
            ),
            "as_of_date": (
                list(dates)
                + list(dates[1:])
                + list(dates[4:])
            ),
            "quantity_held": (
                [10, 10, 0, 0, 0, 0]
                + [5, 5, 5, 5, 0]
                + [3, 0]
            ),
            "market_value_inr": (
                [1000, 1050, 0, 0, 0, 0]
                + [500, 520, 520, 520, 0]
                + [300, 330]
            ),
        }
    )
    cf = implied_market_value_cash_flows_by_day(ap)
    holdings_value = ap.groupby("as_of_date")["market_value_inr"].sum().sort_index()

    twr = time_weighted_return(holdings_value, cf)
    days = (holdings_value.index[-1] - holdings_value.index[0]).days
    cagr_value = annualize_return(twr, days)

    # Real net profit: A exits at 1050 (cost 1000, +50), B exits at 520
    # (cost 500, +20), C exits at 330 (cost 300, +30) -- a real,
    # unambiguous net gain across the whole history.
    assert twr > 0, f"TWR must be positive for a real net profit, got {twr}"
    assert cagr_value > 0, f"CAGR must be positive for a real net profit, got {cagr_value}"


def test_xirr_hand_case_single_period():
    # Invest 1000 today, receive exactly 1100 in exactly 365 days.
    # XIRR must be 10%: -1000 + 1100/(1.10)**(365/365) = 0.
    cash_flows = [(date(2024, 1, 1), -1000.0), (date(2025, 1, 1), 1100.0)]
    result = xirr(cash_flows)
    assert result == pytest.approx(0.10, abs=1e-3)
