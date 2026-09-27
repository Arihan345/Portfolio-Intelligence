"""Regression test for a real bug found via a live upload of a real
30-ticker transaction CSV: a genuine oversell (PCJEWELLER.NS -- 5
shares ever bought across two BUYs, then a single SELL of 50) passed
every existing per-row structural check (positive quantity, valid
ticker/date/type) and reached the warehouse, where it read as a
permanently impossible -45 share holding and failed dbt's hard
quantity_held range test for the WHOLE build -- every downstream mart
was SKIPPED as a result.

Per this project's own Phase 2 design (a single bad row is rejected
individually; it never fails the whole batch), an oversell like this
belongs here, caught and excluded before it ever reaches the warehouse.
_reject_oversold_sells walks the file chronologically per underlying
company, netted across exchanges (the same base-symbol grouping
int_daily_holdings.sql/derive_fact_holdings use, since a real BUY on
.BO and SELL on .NS for the same company is one real position, not
two, and must NOT be flagged as an oversell just because it spans two
tickers).
"""
from __future__ import annotations

import tempfile
from datetime import date

import pandas as pd

from ingestion.portfolio_validation import validate_csv
from ingestion.providers.base import SplitEvent

_HEADER = "date,ticker,transaction_type,quantity,price,fees,tax,currency\n"


def _run(rows: list[str], splits: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write(_HEADER)
        tmp.writelines(row + "\n" for row in rows)
        path = tmp.name
    result = validate_csv(path, today=date(2026, 9, 27), splits=splits)
    return result.valid_rows, result.rejected_rows


def test_genuine_oversell_is_rejected_individually():
    rows = [
        "2024-12-11,PCJEWELLER.NS,BUY,4,177.85,0,0,INR",
        "2024-12-12,PCJEWELLER.NS,BUY,1,176.90,0,0,INR",
        "2025-12-29,PCJEWELLER.NS,SELL,50,9.18,0,0,INR",
    ]
    valid_rows, rejected_rows = _run(rows)

    assert len(valid_rows) == 2, "the two legitimate BUYs must still be accepted"
    assert len(rejected_rows) == 1
    reason = rejected_rows.iloc[0]["rejection_reason"]
    assert "oversold" in reason.lower() or "exceeds" in reason.lower()
    assert "PCJEWELLER" in reason


def test_legitimate_sell_is_not_rejected():
    rows = [
        "2024-01-01,TESTBUG.NS,BUY,50,100,0,0,INR",
        "2024-02-01,TESTBUG.NS,SELL,20,120,0,0,INR",
    ]
    valid_rows, rejected_rows = _run(rows)
    assert len(valid_rows) == 2
    assert len(rejected_rows) == 0


def test_cross_exchange_sell_is_not_falsely_flagged_as_oversold():
    """A BUY on .BO covering a SELL on .NS for the SAME company (a real,
    common Groww workflow -- NSE/BSE holdings of one equity are
    fungible in demat) must NOT be rejected just because the exact
    ticker string differs between the two legs."""
    rows = [
        "2024-01-01,TESTBUG.BO,BUY,5,100,0,0,INR",
        "2024-02-01,TESTBUG.NS,SELL,5,120,0,0,INR",
    ]
    valid_rows, rejected_rows = _run(rows)
    assert len(valid_rows) == 2, f"a fully-covered cross-exchange sell must not be rejected: {rejected_rows}"
    assert len(rejected_rows) == 0


def test_cross_exchange_oversell_is_still_caught():
    """The cross-exchange awareness must not become a loophole: selling
    MORE than the combined BUYs across both exchanges is still a real
    oversell and must still be rejected."""
    rows = [
        "2024-01-01,TESTBUG.BO,BUY,5,100,0,0,INR",
        "2024-02-01,TESTBUG.NS,SELL,20,120,0,0,INR",
    ]
    valid_rows, rejected_rows = _run(rows)
    assert len(valid_rows) == 1
    assert len(rejected_rows) == 1
    assert "TESTBUG" in rejected_rows.iloc[0]["rejection_reason"]


def test_sells_out_of_chronological_csv_order_are_still_validated_correctly():
    """The CSV row order need not be chronological -- validation must
    sort by transaction date, not file order, before netting."""
    rows = [
        "2024-02-01,TESTBUG.NS,SELL,5,120,0,0,INR",   # appears first in the file...
        "2024-01-01,TESTBUG.NS,BUY,5,100,0,0,INR",    # ...but happened after this BUY chronologically
    ]
    valid_rows, rejected_rows = _run(rows)
    assert len(valid_rows) == 2, f"a SELL fully covered by an earlier-dated BUY must not be rejected: {rejected_rows}"


# --------------------------- corporate actions (splits) ---------------------------
# Real bug: PCJEWELLER.NS underwent a real 1:10 split on 2024-12-16
# between a 5-share BUY and what was actually that same position's full
# exit, recorded post-split as a 50-share SELL -- rejected as an
# impossible oversell before apply_split_adjustments existed. These
# tests use a synthetic split (real split-history lookups belong to
# YFinanceProvider.fetch_corporate_actions, not this offline module) to
# prove the general mechanism, plus PCJEWELLER's own real numbers.

def test_presplit_buy_covers_postsplit_sell_with_synthetic_split():
    """5 shares bought before a synthetic 1:10 split, then a SELL of
    exactly 5*10=50 shares after it, must be accepted -- not flagged as
    an oversell of 45 shares."""
    rows = [
        "2024-12-11,TESTSPLIT.NS,BUY,5,177.85,0,0,INR",
        "2025-12-29,TESTSPLIT.NS,SELL,50,9.18,0,0,INR",
    ]
    splits = {"TESTSPLIT.NS": [SplitEvent(ticker="TESTSPLIT.NS", split_date=date(2024, 12, 16), ratio=10.0)]}
    valid_rows, rejected_rows = _run(rows, splits=splits)
    assert len(rejected_rows) == 0, f"a split-covered sell must not be rejected: {rejected_rows.to_dict('records')}"
    assert len(valid_rows) == 2
    # The BUY's quantity/price should be rewritten to post-split terms:
    # 5 shares @ 177.85 -> 50 shares @ 17.785 (same total value).
    buy_row = valid_rows[valid_rows["transaction_type"] == "BUY"].iloc[0]
    assert float(buy_row["quantity"]) == 50.0
    assert float(buy_row["price"]) == 17.785


def test_pcjeweller_real_split_makes_the_real_upload_row_valid():
    """The exact real-world row from the original incident report,
    given PCJEWELLER.NS's REAL split history (1:10 on 2024-12-16, per
    yfinance's Ticker('PCJEWELLER.NS').splits)."""
    rows = [
        "2024-12-11,PCJEWELLER.NS,BUY,4,177.85,0,0,INR",
        "2024-12-12,PCJEWELLER.NS,BUY,1,176.90,0,0,INR",
        "2025-12-29,PCJEWELLER.NS,SELL,50,9.18,0,0,INR",
    ]
    splits = {
        "PCJEWELLER.NS": [
            SplitEvent(ticker="PCJEWELLER.NS", split_date=date(2017, 7, 6), ratio=2.0),
            SplitEvent(ticker="PCJEWELLER.NS", split_date=date(2024, 12, 16), ratio=10.0),
        ]
    }
    valid_rows, rejected_rows = _run(rows, splits=splits)
    assert len(rejected_rows) == 0, f"PCJEWELLER's real split should make this SELL valid: {rejected_rows.to_dict('records')}"
    assert len(valid_rows) == 3
    total_buy_qty = valid_rows[valid_rows["transaction_type"] == "BUY"]["quantity"].astype(float).sum()
    assert total_buy_qty == 50.0, "4+1 shares x the real 1:10 split should equal exactly the 50-share SELL"


def test_split_adjustment_is_audited_not_silent():
    rows = [
        "2024-12-11,TESTSPLIT.NS,BUY,5,177.85,0,0,INR",
        "2025-12-29,TESTSPLIT.NS,SELL,50,9.18,0,0,INR",
    ]
    splits = {"TESTSPLIT.NS": [SplitEvent(ticker="TESTSPLIT.NS", split_date=date(2024, 12, 16), ratio=10.0)]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write(_HEADER)
        tmp.writelines(row + "\n" for row in rows)
        path = tmp.name
    result = validate_csv(path, today=date(2026, 9, 27), splits=splits)
    assert len(result.split_adjustment_notes) == 1
    assert "TESTSPLIT" in result.split_adjustment_notes[0]
    assert "split" in result.split_adjustment_notes[0].lower()


def test_split_awareness_does_not_make_a_genuine_oversell_permissive():
    """A ticker WITH real split data on file must still correctly reject
    a sell that exceeds even the split-adjusted available quantity --
    split-awareness explains a specific real case, it doesn't loosen the
    oversold check in general."""
    rows = [
        "2024-12-11,TESTSPLIT.NS,BUY,5,177.85,0,0,INR",
        "2025-12-29,TESTSPLIT.NS,SELL,999,9.18,0,0,INR",  # far more than 5*10=50
    ]
    splits = {"TESTSPLIT.NS": [SplitEvent(ticker="TESTSPLIT.NS", split_date=date(2024, 12, 16), ratio=10.0)]}
    valid_rows, rejected_rows = _run(rows, splits=splits)
    assert len(rejected_rows) == 1
    assert len(valid_rows) == 1
    assert "TESTSPLIT" in rejected_rows.iloc[0]["rejection_reason"]


def test_no_splits_argument_means_unadjusted_behavior():
    """Callers with no market-data provider available (e.g. this
    module's own other offline tests) get the original, unadjusted
    oversold check -- splits is opt-in, never a silent network call
    inside validate_csv itself."""
    rows = [
        "2024-12-11,PCJEWELLER.NS,BUY,4,177.85,0,0,INR",
        "2024-12-12,PCJEWELLER.NS,BUY,1,176.90,0,0,INR",
        "2025-12-29,PCJEWELLER.NS,SELL,50,9.18,0,0,INR",
    ]
    valid_rows, rejected_rows = _run(rows, splits=None)
    assert len(rejected_rows) == 1
    assert len(valid_rows) == 2
