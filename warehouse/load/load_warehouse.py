"""Loads Phase 2's cleaned output into the Phase 3 warehouse.

Order: dim_asset (SCD2 upsert) -> fact_transactions -> fact_daily_prices
-> fact_holdings (derived) -> fact_portfolio_value (derived) ->
fact_portfolio_returns (derived).

Idempotent: re-running with the same pipeline_run_id-worth of source
data is safe because every insert uses ON CONFLICT DO UPDATE / DO
NOTHING on the tables' natural uniqueness constraints.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import date, datetime

import pandas as pd
import sqlalchemy as sa

DB_URL = "postgresql+psycopg://arihan@localhost:5432/portfolio_analytics"

engine = sa.create_engine(DB_URL)

# The only real user this project has (see warehouse/sql/
# 04_seed_reference.sql) -- there is no auth system, so every portfolio
# a real upload creates belongs to this same account.
DEFAULT_USER_ID = 1


def create_portfolio(name: str, base_currency_code: str = "INR") -> int:
    """Inserts a new dim_portfolio row and returns its portfolio_id.

    dim_portfolio's original seed (04_seed_reference.sql) inserted
    portfolio_id=1 with an EXPLICIT literal, never through nextval() --
    so the sequence's own counter never advanced past its default
    starting position. A plain INSERT ... DEFAULT here relies on
    nextval() to pick the new row's id, which single-portfolio-app code
    elsewhere in this project already worked around per-call (see
    warehouse/tests/test_upload_replace_and_netting.py's test_portfolio_id
    fixture); advancing the sequence to the real current max once here,
    every time, is a cheap, idempotent guard against that same
    collision recurring for a genuinely new portfolio.
    """
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "SELECT setval('dim_portfolio_portfolio_id_seq', "
                "(SELECT COALESCE(MAX(portfolio_id), 1) FROM dim_portfolio))"
            )
        )
        return conn.execute(
            sa.text(
                "INSERT INTO dim_portfolio (user_id, portfolio_name, base_currency_code) "
                "VALUES (:user_id, :name, :currency) RETURNING portfolio_id"
            ),
            {"user_id": DEFAULT_USER_ID, "name": name, "currency": base_currency_code},
        ).scalar_one()


def delete_portfolio(portfolio_id: int) -> bool:
    """Deletes a saved portfolio and every fact row scoped to it.
    Returns False if the portfolio_id didn't exist (nothing to delete).

    Deletes child fact tables before dim_portfolio itself (all four
    have an FK constraint on portfolio_id -- see warehouse/sql/
    02_facts.sql). Doesn't touch dim_asset/fact_daily_prices, which are
    asset-level, not portfolio-scoped, and may be shared with other
    saved portfolios holding the same real ticker.

    This does NOT also clean the dbt-materialized mart_* tables (plain
    physical tables, not views) -- those still show this portfolio's
    now-orphaned rows until the next `dbt build`. Callers that need the
    marts consistent immediately after a delete (e.g. the API's DELETE
    endpoint, so a deleted portfolio doesn't linger in GET /portfolios'
    summary stats) must trigger a rebuild themselves, the same way
    upload already does.
    """
    with engine.begin() as conn:
        exists = conn.execute(
            sa.text("SELECT 1 FROM dim_portfolio WHERE portfolio_id = :pid"), {"pid": portfolio_id}
        ).fetchone()
        if not exists:
            return False
        _lock_portfolio_recompute(conn, portfolio_id)
        for table in ("fact_holdings", "fact_portfolio_value", "fact_portfolio_returns", "fact_transactions"):
            conn.execute(sa.text(f"DELETE FROM {table} WHERE portfolio_id = :pid"), {"pid": portfolio_id})
        conn.execute(sa.text("DELETE FROM dim_portfolio WHERE portfolio_id = :pid"), {"pid": portfolio_id})
        return True


def start_pipeline_run(dag_id: str) -> str:
    run_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO pipeline_runs (run_id, dag_id, status, started_at) "
                "VALUES (:run_id, :dag_id, 'RUNNING', now())"
            ),
            {"run_id": run_id, "dag_id": dag_id},
        )
    return run_id


ERROR_MESSAGE_MAX_CHARS = 8000


def finish_pipeline_run(
    run_id: str, status: str, rows_processed: int, error_message: str | None = None
) -> None:
    """error_message: pass the real exception text on a FAILED run so
    a failure is diagnosable from pipeline_runs itself later, not only
    from whatever terminal happened to be open when it failed -- a
    confirmed gap found during the Phase 10 monitoring investigation,
    where FAILED rows existed with no record of why.

    Truncation (a real, separate bug found in a follow-up
    investigation): an earlier version kept error_message[:2000] --
    the FIRST 2000 characters. For a verbose subprocess log like `dbt
    build`'s (which prints a START/OK line per model/test before its
    final error summary), the first 2000 characters is only the first
    ~15-20 lines and NEVER reaches the actual "ERROR: ..." block or the
    "Completed with N errors" summary line, which appear at the END of
    the output. The column is TEXT (unbounded in Postgres), so the only
    real constraint was this function's own truncation -- fixed by
    keeping the LAST ERROR_MESSAGE_MAX_CHARS characters (the tail, where
    the actionable error actually lives) instead of the first.
    """
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                UPDATE pipeline_runs
                SET status = :status,
                    rows_processed = :rows_processed,
                    completed_at = now(),
                    duration_seconds = EXTRACT(EPOCH FROM (now() - started_at)),
                    error_message = :error_message
                WHERE run_id = :run_id
                """
            ),
            {
                "status": status,
                "rows_processed": rows_processed,
                "run_id": run_id,
                "error_message": error_message[-ERROR_MESSAGE_MAX_CHARS:] if error_message else None,
            },
        )


def ensure_dim_date_coverage(start: date, end: date) -> int:
    """Inserts any missing dim_date rows spanning [start, end] (inclusive).

    dim_date was originally a one-time seed (warehouse/sql/
    04_seed_reference.sql: every calendar day 2023-01-01..2026-12-31,
    generated once for the original demo setup) with no mechanism to
    extend itself for a later upload's real dates -- the same
    fixed-range-never-extended shape as the dim_asset bug fixed
    earlier. Confirmed for real: uploading a fixture with a genuine
    2022-07-15 transaction failed with a fact_transactions foreign key
    violation ("Key (date_key)=(20220715) is not present in table
    dim_date") once the dim_asset gap for that same date was already
    fixed -- dim_date had exactly the same kind of gap, one dimension
    table further down the same insert path.

    is_trading_day is approximated as weekday (Mon-Fri), matching
    04_seed_reference.sql's own approximation -- a real NSE/BSE holiday
    calendar remains a documented follow-up, not something this
    extension mechanism needs to solve differently from the original
    seed.
    """
    with engine.begin() as conn:
        result = conn.execute(
            sa.text(
                """
                INSERT INTO dim_date (date_key, full_date, day_of_month, day_of_week, day_name,
                                       week_of_year, month_num, month_name, quarter, year, is_trading_day)
                SELECT
                    (to_char(d, 'YYYYMMDD'))::INT,
                    d,
                    extract(day FROM d)::SMALLINT,
                    extract(isodow FROM d)::SMALLINT,
                    to_char(d, 'Day'),
                    extract(week FROM d)::SMALLINT,
                    extract(month FROM d)::SMALLINT,
                    to_char(d, 'Month'),
                    extract(quarter FROM d)::SMALLINT,
                    extract(year FROM d)::SMALLINT,
                    extract(isodow FROM d) < 6
                FROM generate_series(:start, :end, '1 day'::INTERVAL) AS d
                ON CONFLICT DO NOTHING
                """
            ),
            {"start": start, "end": end},
        )
        return result.rowcount


def upsert_dim_asset(
    tickers_meta: dict[str, dict],
    as_of: date | dict[str, date],
    source_system: str,
    run_id: str,
) -> dict[str, int]:
    """SCD2 upsert. For each ticker: if no current row exists, insert one.
    If a current row exists but sector/industry changed, close it out
    (effective_to = as_of - 1 day) and insert a new current version.
    Returns {ticker: asset_key} for the row valid as of `as_of`.

    `as_of` is either a single date applied to every ticker (existing
    demo/test callers, where every ticker's data is known to start on
    the same date) or a {ticker: earliest_transaction_date} dict for a
    real upload, where different tickers genuinely have different first
    trade dates. Passing a single, upload-wide date for a NEW ticker's
    effective_from was a real, confirmed bug: a brand-new ticker whose
    earliest real transaction predated that fixed date (e.g. a synthetic
    fixture's DUALLIST.BO first traded 2022-07-15, while the caller
    always passed date(2024, 1, 1)) got a dim_asset row that only
    started covering 2024-01-01 onward -- resolve_asset_key_for_date
    then found no version covering the real 2022-07-15 transaction and
    the whole upload's pipeline failed with "no dim_asset version
    covers ... as of 2022-07-15". A dict lets each genuinely-new
    ticker's effective_from be backdated to ITS OWN earliest real
    transaction date instead of an unrelated upload-wide constant.

    Reclassification (existing ticker, changed sector/industry) is
    conceptually "discovered now," not backdated to a historical trade
    date -- that branch always closes/opens on the latest date in
    `as_of` (or the scalar itself), never an individual ticker's
    earliest date, since backdating a reclassification could otherwise
    open a new version's effective_from before its own predecessor's.
    """
    asset_keys: dict[str, int] = {}
    reclass_as_of = max(as_of.values()) if isinstance(as_of, dict) else as_of
    with engine.begin() as conn:
        for ticker, meta in tickers_meta.items():
            effective_from = as_of[ticker] if isinstance(as_of, dict) else as_of
            existing = conn.execute(
                sa.text(
                    "SELECT asset_key, sector, industry, effective_from FROM dim_asset "
                    "WHERE ticker = :ticker AND is_current"
                ),
                {"ticker": ticker},
            ).fetchone()

            if existing is None:
                result = conn.execute(
                    sa.text(
                        """
                        INSERT INTO dim_asset
                            (ticker, asset_name, sector, industry, exchange,
                             asset_type, currency_code, effective_from,
                             source_system, pipeline_run_id)
                        VALUES
                            (:ticker, :name, :sector, :industry, :exchange,
                             'EQUITY', :currency, :as_of, :source_system, :run_id)
                        RETURNING asset_key
                        """
                    ),
                    {
                        "ticker": ticker,
                        "name": meta.get("name"),
                        "sector": meta.get("sector"),
                        "industry": meta.get("industry"),
                        "exchange": meta.get("exchange"),
                        "currency": meta.get("currency", "INR"),
                        "as_of": effective_from,
                        "source_system": source_system,
                        "run_id": run_id,
                    },
                )
                asset_keys[ticker] = result.scalar_one()
                continue

            changed = existing.sector != meta.get("sector") or existing.industry != meta.get(
                "industry"
            )
            if not changed:
                # A previous upload/demo/test may have already created
                # this ticker's current row with a LATER effective_from
                # than this upload's own earliest transaction for it
                # (this was, in fact, exactly how the real dim_asset
                # bug kept resurfacing even after new-ticker inserts
                # were fixed above: HDFCBANK.NS already had a current
                # row from an earlier run with effective_from =
                # 2024-01-01, so re-uploading a fixture with an earlier
                # real 2022-06-01 transaction hit the "not changed"
                # branch here and reused that row as-is, still too
                # late). Backdating it is safe: unchanged metadata was
                # equally valid at the earlier date too, we simply
                # didn't have transaction data needing it yet.
                if effective_from < existing.effective_from:
                    conn.execute(
                        sa.text("UPDATE dim_asset SET effective_from = :ef WHERE asset_key = :key"),
                        {"ef": effective_from, "key": existing.asset_key},
                    )
                asset_keys[ticker] = existing.asset_key
                continue

            # Reclassification: close current version, insert new one.
            conn.execute(
                sa.text(
                    "UPDATE dim_asset SET effective_to = :prev_day, is_current = FALSE "
                    "WHERE asset_key = :key"
                ),
                {"prev_day": reclass_as_of, "key": existing.asset_key},
            )
            result = conn.execute(
                sa.text(
                    """
                    INSERT INTO dim_asset
                        (ticker, asset_name, sector, industry, exchange,
                         asset_type, currency_code, effective_from,
                         source_system, pipeline_run_id)
                    VALUES
                        (:ticker, :name, :sector, :industry, :exchange,
                         'EQUITY', :currency, :as_of, :source_system, :run_id)
                    RETURNING asset_key
                    """
                ),
                {
                    "ticker": ticker,
                    "name": meta.get("name"),
                    "sector": meta.get("sector"),
                    "industry": meta.get("industry"),
                    "exchange": meta.get("exchange"),
                    "currency": meta.get("currency", "INR"),
                    "as_of": reclass_as_of,
                    "source_system": source_system,
                    "run_id": run_id,
                },
            )
            asset_keys[ticker] = result.scalar_one()
    return asset_keys


def resolve_asset_key_for_date(conn, ticker: str, as_of: date) -> int:
    row = conn.execute(
        sa.text(
            "SELECT asset_key FROM dim_asset WHERE ticker = :ticker "
            "AND effective_from <= :as_of AND effective_to >= :as_of"
        ),
        {"ticker": ticker, "as_of": as_of},
    ).fetchone()
    if row is None:
        raise ValueError(f"no dim_asset version covers {ticker} as of {as_of}")
    return row.asset_key


def load_fact_transactions(
    cleaned: pd.DataFrame, portfolio_id: int, source_system: str, run_id: str
) -> int:
    """Loads `cleaned` as the COMPLETE transaction history for
    `portfolio_id`, replacing whatever was there before.

    This used to only INSERT (ON CONFLICT DO UPDATE on an exact-row
    match), so a second upload for the same portfolio_id never removed
    the first upload's rows -- confirmed as a real bug via /portfolio/
    upload: a synthetic demo portfolio's TCS.NS/RELIANCE.NS rows stayed
    in fact_transactions forever, and a real portfolio uploaded
    afterward was silently merged with it into one inflated, wrong
    portfolio rather than replacing it. The rest of the pipeline
    (derive_fact_holdings/_value/_returns below) already does a full
    DELETE-then-recompute for the portfolio_id on every call; this was
    the one place in the chain that didn't, so those "full recomputes"
    were faithfully recomputing from an ever-growing, never-cleared
    transaction table. Guarded by the same portfolio-scoped advisory
    lock the derive_fact_* functions use, for the same reason: a
    concurrent DELETE+INSERT here can otherwise race a concurrent
    derive_fact_holdings read of the same portfolio_id.

    A wholly empty `cleaned` (e.g. every row in this upload was
    rejected by validation) deliberately does NOT delete the existing
    portfolio -- there is nothing to replace it with, and a garbage/
    empty upload should not be able to silently wipe a working
    portfolio.
    """
    rows = 0
    with engine.begin() as conn:
        _lock_portfolio_recompute(conn, portfolio_id)
        if not cleaned.empty:
            conn.execute(
                sa.text("DELETE FROM fact_transactions WHERE portfolio_id = :p"), {"p": portfolio_id}
            )
        for _, r in cleaned.iterrows():
            txn_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
            date_key = int(txn_date.strftime("%Y%m%d"))
            asset_key = resolve_asset_key_for_date(conn, r["ticker"], txn_date)
            result = conn.execute(
                sa.text(
                    """
                    INSERT INTO fact_transactions
                        (portfolio_id, asset_key, date_key, transaction_type,
                         quantity, price, price_inr, fees, tax, currency_code,
                         near_duplicate_flag, price_anomaly_flag,
                         source_system, pipeline_run_id)
                    VALUES
                        (:portfolio_id, :asset_key, :date_key, :ttype,
                         :quantity, :price, :price_inr, :fees, :tax, :currency,
                         :near_dup, :anomaly, :source_system, :run_id)
                    ON CONFLICT (portfolio_id, asset_key, date_key, transaction_type, quantity, price)
                    DO UPDATE SET ingested_at = now(), pipeline_run_id = EXCLUDED.pipeline_run_id
                    """
                ),
                {
                    "portfolio_id": portfolio_id,
                    "asset_key": asset_key,
                    "date_key": date_key,
                    "ttype": r["transaction_type"],
                    "quantity": float(r["quantity"]),
                    "price": float(r["price"]),
                    "price_inr": float(r["price_inr"]) if pd.notna(r["price_inr"]) else None,
                    "fees": float(r["fees"]),
                    "tax": float(r["tax"]),
                    "currency": r["currency"],
                    "near_dup": bool(r["near_duplicate_flag"]),
                    "anomaly": bool(r["price_anomaly_flag"]),
                    "source_system": source_system,
                    "run_id": run_id,
                },
            )
            rows += result.rowcount
    return rows


def load_fact_daily_prices(
    ohlcv: pd.DataFrame, source_system: str, run_id: str
) -> int:
    rows = 0
    with engine.begin() as conn:
        for _, r in ohlcv.iterrows():
            price_date = r["date"]
            date_key = int(price_date.strftime("%Y%m%d"))
            asset_key = resolve_asset_key_for_date(conn, r["ticker"], price_date)
            result = conn.execute(
                sa.text(
                    """
                    INSERT INTO fact_daily_prices
                        (asset_key, date_key, open, high, low, close, volume,
                         source_system, pipeline_run_id)
                    VALUES
                        (:asset_key, :date_key, :open, :high, :low, :close, :volume,
                         :source_system, :run_id)
                    ON CONFLICT (asset_key, date_key)
                    DO UPDATE SET open = EXCLUDED.open, high = EXCLUDED.high,
                        low = EXCLUDED.low, close = EXCLUDED.close,
                        volume = EXCLUDED.volume, ingested_at = now(),
                        pipeline_run_id = EXCLUDED.pipeline_run_id
                    """
                ),
                {
                    "asset_key": asset_key,
                    "date_key": date_key,
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": int(r["volume"]),
                    "source_system": source_system,
                    "run_id": run_id,
                },
            )
            rows += result.rowcount
    return rows


# Reserved advisory-lock "class" id for the derive_fact_* full-recompute
# functions below. See _lock_portfolio_recompute's docstring for why
# this exists: DELETE-then-INSERT full recomputes are idempotent for
# sequential retries (Phase 1's frozen requirement) but are NOT safe
# under CONCURRENT execution on their own -- two overlapping calls for
# the same portfolio_id can each DELETE, then both INSERT the same
# primary key, causing a real UniqueViolation (reproduced directly
# against fact_holdings during the Phase 10 monitoring-page
# investigation: two threads calling derive_fact_holdings(1, ...)
# concurrently, one raised
# "duplicate key value violates unique constraint fact_holdings_pkey").
_DERIVE_LOCK_CLASS = 987654321


def _lock_portfolio_recompute(conn, portfolio_id: int) -> None:
    """Serializes full-recompute derive_fact_* calls for the same
    portfolio_id: a second concurrent call BLOCKS here until the first
    transaction commits (or rolls back), then proceeds against the
    now-consistent state, instead of racing its DELETE/INSERT against
    the first call's. Held for the lifetime of the transaction
    (pg_advisory_XACT_lock releases automatically on commit/rollback --
    no separate unlock call needed, and it can't be leaked by an
    exception).
    """
    conn.execute(
        sa.text("SELECT pg_advisory_xact_lock(:lock_class, :portfolio_id)"),
        {"lock_class": _DERIVE_LOCK_CLASS, "portfolio_id": portfolio_id},
    )


_DBT_BUILD_LOCK_KEY = 123456789


@contextmanager
def dbt_build_lock():
    """Session-level advisory lock serializing `dbt build` invocations
    warehouse-wide -- NOT portfolio-scoped like
    _lock_portfolio_recompute above, because dbt operates on the whole
    warehouse/schema (all marts, regardless of portfolio_id) and its
    own materialization strategy is not safe for concurrent runs: two
    `dbt build` processes racing to swap the same table model both try
    to create/drop the SAME "<model>__dbt_backup" table, which fails
    with "relation already exists" for whichever loses the race
    (reproduced directly: two concurrent /portfolio/upload requests,
    once the endpoint's own accidental serialization -- see
    api.routers.portfolio.upload_portfolio's docstring -- was removed).

    Uses a SESSION-level lock (pg_advisory_lock/unlock), not the
    transaction-scoped pg_advisory_xact_lock used above, because the
    protected work is an external subprocess call, not a single SQL
    transaction dbt's own connection could scope a xact-lock to.
    """
    conn = engine.connect()
    try:
        conn.execute(sa.text("SELECT pg_advisory_lock(:key)"), {"key": _DBT_BUILD_LOCK_KEY})
        conn.commit()
        yield
    finally:
        conn.execute(sa.text("SELECT pg_advisory_unlock(:key)"), {"key": _DBT_BUILD_LOCK_KEY})
        conn.commit()
        conn.close()


def derive_fact_holdings(portfolio_id: int, source_system: str, run_id: str) -> int:
    """Recompute holdings for every trading day between the portfolio's
    first transaction and today, by cumulative-summing BUY/SELL
    quantity per asset. Full recompute here (small dataset); production
    would advance incrementally from the last processed date.

    Cross-exchange netting: mirrors int_daily_holdings.sql's
    canonical_asset fix -- a real brokerage (Groww) lets the same
    underlying company be bought on one exchange and sold on the other
    (e.g. BUY 5 MOREPENLAB.BO, SELL 5 MOREPENLAB.NS; NSE/BSE holdings of
    one equity are fungible in demat), which is one real position, not
    two. Netting purely by raw asset_key treated that BUY and SELL as
    unrelated, leaving the SELL with no same-ticker BUY to net against
    -- a permanently, impossibly negative quantity_held (this is the
    same bug int_daily_holdings.sql had; this table is a separate,
    parallel computation of the same thing and needs the same fix, even
    though the live API/frontend read the dbt marts, not this table).
    Every base-symbol's transactions are folded onto ONE representative
    asset_key (the earliest-inserted one for that symbol) purely for
    this quantity/cost-basis computation; dim_asset and
    fact_transactions themselves are untouched.
    """
    with engine.begin() as conn:
        _lock_portfolio_recompute(conn, portfolio_id)
        conn.execute(sa.text("DELETE FROM fact_holdings WHERE portfolio_id = :p"), {"p": portfolio_id})
        conn.execute(
            sa.text(
                """
                WITH tx AS (
                    SELECT ft.*, da.is_current, split_part(da.ticker, '.', 1) AS base_symbol
                    FROM fact_transactions ft
                    JOIN dim_asset da ON da.asset_key = ft.asset_key
                    WHERE ft.portfolio_id = :portfolio_id
                      AND ft.transaction_type IN ('BUY','SELL')
                ),
                -- Must prefer the CURRENT SCD2 dim_asset row, not just
                -- MIN(asset_key) -- see int_daily_holdings.sql's matching
                -- comment for the real bug this fixes (a plain min() can
                -- pick an old, closed-out asset_key whose SCD2 validity
                -- window is tiny, silently truncating that ticker's
                -- entire holdings history to almost nothing).
                canonical_asset AS (
                    SELECT DISTINCT ON (base_symbol) base_symbol, asset_key
                    FROM tx
                    ORDER BY base_symbol, is_current DESC, asset_key ASC
                ),
                daily_moves AS (
                    -- Aggregate same-day activity for the SAME underlying
                    -- company (across exchanges) to one row per (asset,
                    -- date) BEFORE windowing, so a day with two
                    -- transactions never produces two holdings rows.
                    SELECT
                        ca.asset_key,
                        tx.date_key,
                        SUM(CASE tx.transaction_type WHEN 'BUY' THEN tx.quantity
                                                      WHEN 'SELL' THEN -tx.quantity
                                                      ELSE 0 END) AS qty_delta,
                        SUM(CASE tx.transaction_type WHEN 'BUY' THEN tx.price_inr * tx.quantity + tx.fees
                                                      WHEN 'SELL' THEN -(tx.price_inr * tx.quantity - tx.fees - tx.tax)
                                                      ELSE 0 END) AS cost_delta
                    FROM tx
                    JOIN canonical_asset ca ON ca.base_symbol = tx.base_symbol
                    GROUP BY ca.asset_key, tx.date_key
                ),
                bounds AS (
                    SELECT MIN(df.full_date) AS min_d, MAX(df.full_date) AS max_d
                    FROM fact_transactions ft
                    JOIN dim_date df ON df.date_key = ft.date_key
                    WHERE ft.portfolio_id = :portfolio_id
                ),
                calendar AS (
                    SELECT date_key FROM dim_date, bounds
                    WHERE full_date BETWEEN bounds.min_d AND bounds.max_d
                ),
                assets AS (
                    SELECT asset_key FROM canonical_asset
                )
                INSERT INTO fact_holdings
                    (portfolio_id, asset_key, date_key, quantity_held, cost_basis_inr,
                     source_system, pipeline_run_id)
                SELECT
                    :portfolio_id,
                    a.asset_key,
                    c.date_key,
                    COALESCE(SUM(m.qty_delta) OVER (
                        PARTITION BY a.asset_key ORDER BY c.date_key
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0) AS quantity_held,
                    COALESCE(SUM(m.cost_delta) OVER (
                        PARTITION BY a.asset_key ORDER BY c.date_key
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0) AS cost_basis_inr,
                    :source_system,
                    :run_id
                FROM calendar c
                CROSS JOIN assets a
                LEFT JOIN daily_moves m
                    ON m.asset_key = a.asset_key AND m.date_key = c.date_key
                """
            ),
            {"portfolio_id": portfolio_id, "source_system": source_system, "run_id": run_id},
        )
        count = conn.execute(
            sa.text("SELECT COUNT(*) FROM fact_holdings WHERE portfolio_id = :p"),
            {"p": portfolio_id},
        ).scalar_one()
    return count


def derive_fact_portfolio_value(portfolio_id: int, source_system: str, run_id: str) -> int:
    with engine.begin() as conn:
        _lock_portfolio_recompute(conn, portfolio_id)
        conn.execute(sa.text("DELETE FROM fact_portfolio_value WHERE portfolio_id = :p"), {"p": portfolio_id})
        conn.execute(
            sa.text(
                """
                -- Non-trading days (weekends/holidays) have no
                -- fact_daily_prices row for that exact date_key. Marking
                -- to market must still carry forward the last known
                -- close, or a weekend snapshot silently prices every
                -- holding at zero even though the position is real.
                WITH priced AS (
                    SELECT
                        h.portfolio_id,
                        h.asset_key,
                        h.date_key,
                        h.quantity_held,
                        h.cost_basis_inr,
                        (SELECT p.close FROM fact_daily_prices p
                         WHERE p.asset_key = h.asset_key AND p.date_key <= h.date_key
                         ORDER BY p.date_key DESC LIMIT 1) AS last_known_close
                    FROM fact_holdings h
                    WHERE h.portfolio_id = :portfolio_id
                )
                INSERT INTO fact_portfolio_value
                    (portfolio_id, date_key, market_value_inr, invested_capital_inr,
                     source_system, pipeline_run_id)
                SELECT
                    portfolio_id,
                    date_key,
                    SUM(quantity_held * COALESCE(last_known_close, 0)) AS market_value_inr,
                    SUM(cost_basis_inr) AS invested_capital_inr,
                    :source_system,
                    :run_id
                FROM priced
                GROUP BY portfolio_id, date_key
                """
            ),
            {"portfolio_id": portfolio_id, "source_system": source_system, "run_id": run_id},
        )
        count = conn.execute(
            sa.text("SELECT COUNT(*) FROM fact_portfolio_value WHERE portfolio_id = :p"),
            {"p": portfolio_id},
        ).scalar_one()
    return count


def derive_fact_portfolio_returns(portfolio_id: int, source_system: str, run_id: str) -> int:
    with engine.begin() as conn:
        _lock_portfolio_recompute(conn, portfolio_id)
        conn.execute(sa.text("DELETE FROM fact_portfolio_returns WHERE portfolio_id = :p"), {"p": portfolio_id})
        conn.execute(
            sa.text(
                """
                INSERT INTO fact_portfolio_returns
                    (portfolio_id, date_key, daily_return, cumulative_return,
                     source_system, pipeline_run_id)
                SELECT
                    portfolio_id,
                    date_key,
                    CASE WHEN LAG(market_value_inr) OVER w > 0
                         THEN (market_value_inr - LAG(market_value_inr) OVER w) / LAG(market_value_inr) OVER w
                         ELSE NULL END AS daily_return,
                    CASE WHEN FIRST_VALUE(market_value_inr) OVER w > 0
                         THEN (market_value_inr - FIRST_VALUE(market_value_inr) OVER w) / FIRST_VALUE(market_value_inr) OVER w
                         ELSE NULL END AS cumulative_return,
                    :source_system,
                    :run_id
                FROM fact_portfolio_value
                WHERE portfolio_id = :portfolio_id
                WINDOW w AS (PARTITION BY portfolio_id ORDER BY date_key)
                """
            ),
            {"portfolio_id": portfolio_id, "source_system": source_system, "run_id": run_id},
        )
        count = conn.execute(
            sa.text("SELECT COUNT(*) FROM fact_portfolio_returns WHERE portfolio_id = :p"),
            {"p": portfolio_id},
        ).scalar_one()
    return count
