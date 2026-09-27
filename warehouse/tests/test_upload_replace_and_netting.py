"""Regression test for a real, reproduced data-correctness bug found via
a live /portfolio/upload of a real transaction CSV: after uploading a
real portfolio, the dashboard still showed the old synthetic demo
tickers (TCS.NS/RELIANCE.NS) mixed in alongside the new real ones, and
total portfolio value was off by more than 10x.

Root cause: load_fact_transactions only ever INSERTed (ON CONFLICT DO
UPDATE on an exact-row match). A second upload for the same
portfolio_id never removed the first upload's rows, so every upload
accumulated on top of whatever was already there instead of replacing
it -- confirmed directly against the dev database (fact_transactions
still held TCS.NS/RELIANCE.NS rows from the original demo load after a
brand-new real CSV was uploaded for the same portfolio_id). This
project has exactly one portfolio row (dim_portfolio has a single
"Primary Portfolio"; the frontend has no portfolio switcher and always
requests portfolio_id=1), so "replace the portfolio's transactions on
every upload" is the correct, intended behavior -- not "append" and
not "create a new portfolio per upload".

A second hypothesis raised during the same investigation -- that
int_daily_holdings / fact_holdings fails to net SELL transactions
against prior BUY quantity, leaving fully-exited positions non-zero --
was investigated and ruled out: manually tracing a real BUY+SELL pair
already in the warehouse (TCS.NS: 10 bought, 3 sold) showed the
correct 10 -> 7 quantity, and the netting SQL is a straightforward
signed cumulative sum. It is still covered here as a permanent
regression test (test_full_exit_nets_to_exactly_zero) since it was the
leading hypothesis and a real, high-value invariant either way: a
fully bought-then-fully-sold position must disappear from current
holdings (quantity exactly 0), not linger.

A THIRD real bug, found in a later real upload (30 real tickers,
several traded on both NSE and BSE): the netting above was scoped
purely to raw asset_key, so a company bought on one exchange and sold
on the other (e.g. BUY 5 MOREPENLAB.BO, SELL 5 MOREPENLAB.NS -- a real,
common Groww workflow, since NSE/BSE holdings of one equity are
fungible in demat) had its SELL leg permanently read as an impossible
negative quantity with no same-ticker BUY to net against. This failed
dbt's hard dbt_utils_accepted_range test on quantity_held for 1,579
rows and blocked the entire downstream build (every mart SKIPPED).
Fixed in both int_daily_holdings.sql and derive_fact_holdings below by
folding every base-symbol's (ticker with the exchange suffix stripped)
transactions onto one canonical asset_key before netting.
test_cross_exchange_buy_sell_nets_to_zero and
test_cross_exchange_partial_sell_nets_correctly cover this against
derive_fact_holdings directly (fast, no dbt subprocess needed) since
both implementations share the identical bug and fix.

Runs against the real database (integration-style, like
test_concurrency.py), using a dedicated, disposable dim_portfolio row
so it never touches whatever is currently loaded under the app's real
portfolio_id=1.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import sqlalchemy as sa

from warehouse.load.load_warehouse import (
    derive_fact_holdings,
    engine,
    ensure_dim_date_coverage,
    load_fact_transactions,
    resolve_asset_key_for_date,
    start_pipeline_run,
    upsert_dim_asset,
)

ASSET_META = {
    "TESTBUG.NS": {
        "name": "Test Bug Co", "sector": "Test", "industry": "Test",
        "exchange": "NSE", "currency": "INR",
    },
    "TESTBUG.BO": {
        "name": "Test Bug Co", "sector": "Test", "industry": "Test",
        "exchange": "BSE", "currency": "INR",
    },
    "TESTHOLD.NS": {
        "name": "Test Hold Co", "sector": "Test", "industry": "Test",
        "exchange": "NSE", "currency": "INR",
    },
}

_ROW_DEFAULTS = {
    "fees": 0.0, "tax": 0.0, "currency": "INR",
    "near_duplicate_flag": False, "price_anomaly_flag": False,
}


def _txn(txn_date: str, ticker: str, txn_type: str, quantity: float, price: float) -> dict:
    return {
        "date": txn_date, "ticker": ticker, "transaction_type": txn_type,
        "quantity": quantity, "price": price, "price_inr": price, **_ROW_DEFAULTS,
    }


def _tickers_in_portfolio(portfolio_id: int) -> set[str]:
    with engine.begin() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT DISTINCT da.ticker FROM fact_transactions ft "
                "JOIN dim_asset da ON ft.asset_key = da.asset_key "
                "WHERE ft.portfolio_id = :p"
            ),
            {"p": portfolio_id},
        ).fetchall()
    return {r[0] for r in rows}


@pytest.fixture
def test_portfolio_id():
    with engine.begin() as conn:
        # dim_portfolio's serial sequence was never advanced past the
        # single manually-seeded portfolio_id=1 row, so a plain
        # INSERT ... DEFAULT would collide on it (nextval() still
        # returns 1). Advancing the sequence to the current max id
        # first is a safe, idempotent way to get a real unused id
        # without hardcoding one.
        conn.execute(
            sa.text(
                "SELECT setval('dim_portfolio_portfolio_id_seq', "
                "(SELECT COALESCE(MAX(portfolio_id), 1) FROM dim_portfolio))"
            )
        )
        pid = conn.execute(
            sa.text(
                "INSERT INTO dim_portfolio (user_id, portfolio_name, base_currency_code) "
                "VALUES (1, 'pytest-upload-replace', 'INR') RETURNING portfolio_id"
            )
        ).scalar_one()
    yield pid
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM fact_holdings WHERE portfolio_id = :p"), {"p": pid})
        conn.execute(sa.text("DELETE FROM fact_transactions WHERE portfolio_id = :p"), {"p": pid})
        conn.execute(sa.text("DELETE FROM dim_portfolio WHERE portfolio_id = :p"), {"p": pid})


def test_second_upload_replaces_not_appends(test_portfolio_id):
    run_id_1 = start_pipeline_run(dag_id="pytest")
    upsert_dim_asset(ASSET_META, date(2024, 1, 1), "pytest", run_id_1)

    first_upload = pd.DataFrame([_txn("2024-01-01", "TESTBUG.NS", "BUY", 10, 100.0)])
    load_fact_transactions(first_upload, test_portfolio_id, "pytest", run_id_1)
    assert _tickers_in_portfolio(test_portfolio_id) == {"TESTBUG.NS"}

    run_id_2 = start_pipeline_run(dag_id="pytest")
    second_upload = pd.DataFrame([_txn("2024-02-01", "TESTHOLD.NS", "BUY", 5, 200.0)])
    load_fact_transactions(second_upload, test_portfolio_id, "pytest", run_id_2)

    tickers_after_second = _tickers_in_portfolio(test_portfolio_id)
    assert tickers_after_second == {"TESTHOLD.NS"}, (
        f"a second upload must REPLACE the portfolio's transactions, not append to them "
        f"-- expected only {{'TESTHOLD.NS'}}, found {tickers_after_second}"
    )


def test_empty_upload_does_not_wipe_existing_portfolio(test_portfolio_id):
    """A garbage/all-rejected upload (nothing valid to load) must not be
    able to silently delete a working portfolio -- there is nothing to
    replace it WITH."""
    run_id_1 = start_pipeline_run(dag_id="pytest")
    upsert_dim_asset(ASSET_META, date(2024, 1, 1), "pytest", run_id_1)
    first_upload = pd.DataFrame([_txn("2024-01-01", "TESTBUG.NS", "BUY", 10, 100.0)])
    load_fact_transactions(first_upload, test_portfolio_id, "pytest", run_id_1)

    run_id_2 = start_pipeline_run(dag_id="pytest")
    load_fact_transactions(pd.DataFrame(columns=first_upload.columns), test_portfolio_id, "pytest", run_id_2)

    assert _tickers_in_portfolio(test_portfolio_id) == {"TESTBUG.NS"}


def test_full_exit_nets_to_exactly_zero(test_portfolio_id):
    run_id = start_pipeline_run(dag_id="pytest")
    upsert_dim_asset(ASSET_META, date(2024, 1, 1), "pytest", run_id)

    rows = pd.DataFrame([
        _txn("2024-01-01", "TESTBUG.NS", "BUY", 50, 100.0),
        _txn("2024-02-01", "TESTBUG.NS", "SELL", 50, 120.0),
    ])
    load_fact_transactions(rows, test_portfolio_id, "pytest", run_id)
    derive_fact_holdings(test_portfolio_id, "pytest", run_id)

    with engine.begin() as conn:
        qty = conn.execute(
            sa.text(
                "SELECT h.quantity_held FROM fact_holdings h "
                "JOIN dim_asset da ON h.asset_key = da.asset_key "
                "WHERE h.portfolio_id = :p AND da.ticker = 'TESTBUG.NS' "
                "ORDER BY h.date_key DESC LIMIT 1"
            ),
            {"p": test_portfolio_id},
        ).scalar_one()
    assert float(qty) == 0.0, f"a fully bought-then-fully-sold position must net to exactly 0, got {qty}"


def test_partial_sell_nets_correctly(test_portfolio_id):
    run_id = start_pipeline_run(dag_id="pytest")
    upsert_dim_asset(ASSET_META, date(2024, 1, 1), "pytest", run_id)

    rows = pd.DataFrame([
        _txn("2024-01-01", "TESTBUG.NS", "BUY", 50, 100.0),
        _txn("2024-02-01", "TESTBUG.NS", "SELL", 20, 120.0),
    ])
    load_fact_transactions(rows, test_portfolio_id, "pytest", run_id)
    derive_fact_holdings(test_portfolio_id, "pytest", run_id)

    with engine.begin() as conn:
        qty = conn.execute(
            sa.text(
                "SELECT h.quantity_held FROM fact_holdings h "
                "JOIN dim_asset da ON h.asset_key = da.asset_key "
                "WHERE h.portfolio_id = :p AND da.ticker = 'TESTBUG.NS' "
                "ORDER BY h.date_key DESC LIMIT 1"
            ),
            {"p": test_portfolio_id},
        ).scalar_one()
    assert float(qty) == 30.0, f"50 bought - 20 sold should leave exactly 30, got {qty}"


def _quantity_for_base_symbol(portfolio_id: int, base_symbol: str) -> float:
    """After cross-exchange netting, only ONE of a base symbol's
    exchange-suffixed tickers actually ends up with rows in
    fact_holdings (the canonical one) -- match by prefix so the test
    doesn't need to know or care which exchange won."""
    with engine.begin() as conn:
        qty = conn.execute(
            sa.text(
                "SELECT h.quantity_held FROM fact_holdings h "
                "JOIN dim_asset da ON h.asset_key = da.asset_key "
                "WHERE h.portfolio_id = :p AND da.ticker LIKE :pattern "
                "ORDER BY h.date_key DESC LIMIT 1"
            ),
            {"p": portfolio_id, "pattern": f"{base_symbol}.%"},
        ).scalar_one()
    return float(qty)


def test_cross_exchange_buy_sell_nets_to_zero(test_portfolio_id):
    """Real bug: BUY on .BO, SELL of the same quantity on .NS for the
    same company must net to exactly 0 -- not read as an oversold -N
    position on .NS with an untouched, separate +N position on .BO."""
    run_id = start_pipeline_run(dag_id="pytest")
    upsert_dim_asset(ASSET_META, date(2024, 1, 1), "pytest", run_id)

    rows = pd.DataFrame([
        _txn("2024-01-01", "TESTBUG.BO", "BUY", 5, 100.0),
        _txn("2024-02-01", "TESTBUG.NS", "SELL", 5, 120.0),
    ])
    load_fact_transactions(rows, test_portfolio_id, "pytest", run_id)
    derive_fact_holdings(test_portfolio_id, "pytest", run_id)

    qty = _quantity_for_base_symbol(test_portfolio_id, "TESTBUG")
    assert qty == 0.0, f"BUY 5 on .BO + SELL 5 on .NS (same company) should net to exactly 0, got {qty}"

    # And it must never appear as a separate, still-negative position
    # under the OTHER ticker either.
    with engine.begin() as conn:
        rows_found = conn.execute(
            sa.text(
                "SELECT da.ticker, h.quantity_held FROM fact_holdings h "
                "JOIN dim_asset da ON h.asset_key = da.asset_key "
                "WHERE h.portfolio_id = :p AND da.ticker LIKE 'TESTBUG.%' "
                "ORDER BY h.date_key DESC LIMIT 5"
            ),
            {"p": test_portfolio_id},
        ).fetchall()
    tickers_with_holdings = {r[0] for r in rows_found}
    assert len(tickers_with_holdings) == 1, (
        f"expected exactly one consolidated TESTBUG position, found rows under {tickers_with_holdings}"
    )


def test_cross_exchange_partial_sell_nets_correctly(test_portfolio_id):
    run_id = start_pipeline_run(dag_id="pytest")
    upsert_dim_asset(ASSET_META, date(2024, 1, 1), "pytest", run_id)

    rows = pd.DataFrame([
        _txn("2024-01-01", "TESTBUG.BO", "BUY", 10, 100.0),
        _txn("2024-02-01", "TESTBUG.NS", "SELL", 4, 120.0),
    ])
    load_fact_transactions(rows, test_portfolio_id, "pytest", run_id)
    derive_fact_holdings(test_portfolio_id, "pytest", run_id)

    qty = _quantity_for_base_symbol(test_portfolio_id, "TESTBUG")
    assert qty == 6.0, f"10 bought on .BO - 4 sold on .NS should leave exactly 6, got {qty}"


def test_new_ticker_backdated_transaction_gets_covering_dim_asset_row(test_portfolio_id):
    """Regression for a real, confirmed bug: uploading a synthetic
    edge-case fixture with a brand-new ticker (DUALLIST.BO) whose
    earliest real transaction was 2022-07-15 failed the whole pipeline
    with "no dim_asset version covers DUALLIST.BO as of 2022-07-15".

    Root cause: api/routers/portfolio.py's upload endpoint always called
    upsert_dim_asset with a single hardcoded date(2024, 1, 1) as
    effective_from for every ticker in the upload, regardless of that
    ticker's OWN earliest transaction date. Any brand-new ticker whose
    real first trade predated that constant got a dim_asset row whose
    coverage started too late, so resolve_asset_key_for_date (used by
    load_fact_transactions for every row) found no covering version and
    raised. The real portfolio never hit this because its earliest
    trade (2024-04-18) happened to be after the hardcoded constant --
    this was a real, general bug for any historical-dated new ticker,
    not specific to one CSV.

    Fixed by letting upsert_dim_asset take a {ticker: earliest_date}
    dict instead of one shared date, and by having the upload endpoint
    build that dict from the actual upload's own earliest date per
    ticker (see upsert_dim_asset's and the router's updated docstrings).
    """
    backdated_ticker = "TESTBACKDATE.BO"
    # Earlier than the hardcoded date(2024, 1, 1) the router used to
    # always pass, but still within dim_date's seeded 2023-2026 range
    # (warehouse/sql/04_seed_reference.sql) -- this test is about the
    # dim_asset gap specifically, not dim_date's separate seeded range.
    real_earliest_date = date(2023, 7, 15)
    asset_meta = {
        backdated_ticker: {
            "name": "Test Backdate Co", "sector": "Test", "industry": "Test",
            "exchange": "BSE", "currency": "INR",
        },
    }
    try:
        run_id = start_pipeline_run(dag_id="pytest")
        upsert_dim_asset(asset_meta, {backdated_ticker: real_earliest_date}, "pytest", run_id)

        rows = pd.DataFrame([_txn("2023-07-15", backdated_ticker, "BUY", 2, 15.0)])
        # This is exactly what raised ValueError("no dim_asset version
        # covers ... as of ...") before the fix -- load_fact_transactions
        # resolves an asset_key for this historical date via
        # resolve_asset_key_for_date on every row.
        load_fact_transactions(rows, test_portfolio_id, "pytest", run_id)

        with engine.begin() as conn:
            asset_key = resolve_asset_key_for_date(conn, backdated_ticker, real_earliest_date)
        assert asset_key is not None
    finally:
        with engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM fact_transactions WHERE portfolio_id = :p"), {"p": test_portfolio_id})
            conn.execute(sa.text("DELETE FROM dim_asset WHERE ticker = :t"), {"t": backdated_ticker})


def test_transaction_date_outside_dim_date_range_gets_extended(test_portfolio_id):
    """Regression for the SAME root-cause pattern as the dim_asset bug
    above, one dimension table further down the same insert path: a
    real, confirmed failure -- "insert or update on table
    fact_transactions violates foreign key constraint
    fact_transactions_date_key_fkey ... Key (date_key)=(20220715) is
    not present in table dim_date" -- surfaced uploading a fixture with
    a genuine 2022-07-15 transaction, AFTER the dim_asset gap for that
    same date was already fixed.

    dim_date was a one-time seed (warehouse/sql/04_seed_reference.sql:
    every calendar day 2023-01-01..2026-12-31, built once for the
    original demo setup) with no mechanism to extend for a later
    upload's real, earlier dates. ensure_dim_date_coverage backfills
    any missing dim_date rows for exactly the range an upload needs,
    called before load_fact_transactions in the router.
    """
    # Deliberately far outside dim_date's seeded/extended range (rather
    # than literally 2022-07-15, the bug report's own date) -- a real
    # upload since this fix landed can legitimately have already
    # extended dim_date backward, which would make that exact date a
    # false negative for this test. 1999 predates any real portfolio
    # data this project has ever ingested.
    historical_date = date(1999, 7, 15)
    date_key = int(historical_date.strftime("%Y%m%d"))
    with engine.begin() as conn:
        preexisting = conn.execute(
            sa.text("SELECT 1 FROM dim_date WHERE date_key = :dk"), {"dk": date_key}
        ).fetchone()
    assert preexisting is None, "test date must be genuinely outside dim_date's seeded range to be a real test"

    ticker = "TESTOLDDATE.NS"
    asset_meta = {
        ticker: {"name": "Test Old Date Co", "sector": "Test", "industry": "Test", "exchange": "NSE", "currency": "INR"},
    }
    try:
        ensure_dim_date_coverage(historical_date, historical_date)
        with engine.begin() as conn:
            row = conn.execute(sa.text("SELECT 1 FROM dim_date WHERE date_key = :dk"), {"dk": date_key}).fetchone()
        assert row is not None, "ensure_dim_date_coverage must insert the missing date row"

        run_id = start_pipeline_run(dag_id="pytest")
        upsert_dim_asset(asset_meta, {ticker: historical_date}, "pytest", run_id)

        rows = pd.DataFrame([_txn("1999-07-15", ticker, "BUY", 3, 100.0)])
        # This is exactly what raised a ForeignKeyViolation on
        # fact_transactions before the fix.
        n = load_fact_transactions(rows, test_portfolio_id, "pytest", run_id)
        assert n == 1
    finally:
        with engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM fact_transactions WHERE portfolio_id = :p"), {"p": test_portfolio_id})
            conn.execute(sa.text("DELETE FROM dim_asset WHERE ticker = :t"), {"t": ticker})
            conn.execute(sa.text("DELETE FROM dim_date WHERE date_key = :dk"), {"dk": date_key})
