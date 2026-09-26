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
from datetime import date, datetime

import pandas as pd
import sqlalchemy as sa

DB_URL = "postgresql+psycopg://arihan@localhost:5432/portfolio_analytics"

engine = sa.create_engine(DB_URL)


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


def finish_pipeline_run(run_id: str, status: str, rows_processed: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                UPDATE pipeline_runs
                SET status = :status,
                    rows_processed = :rows_processed,
                    completed_at = now(),
                    duration_seconds = EXTRACT(EPOCH FROM (now() - started_at))
                WHERE run_id = :run_id
                """
            ),
            {"status": status, "rows_processed": rows_processed, "run_id": run_id},
        )


def upsert_dim_asset(
    tickers_meta: dict[str, dict], as_of: date, source_system: str, run_id: str
) -> dict[str, int]:
    """SCD2 upsert. For each ticker: if no current row exists, insert one.
    If a current row exists but sector/industry changed, close it out
    (effective_to = as_of - 1 day) and insert a new current version.
    Returns {ticker: asset_key} for the row valid as of `as_of`.
    """
    asset_keys: dict[str, int] = {}
    with engine.begin() as conn:
        for ticker, meta in tickers_meta.items():
            existing = conn.execute(
                sa.text(
                    "SELECT asset_key, sector, industry FROM dim_asset "
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
                        "as_of": as_of,
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
                asset_keys[ticker] = existing.asset_key
                continue

            # Reclassification: close current version, insert new one.
            conn.execute(
                sa.text(
                    "UPDATE dim_asset SET effective_to = :prev_day, is_current = FALSE "
                    "WHERE asset_key = :key"
                ),
                {"prev_day": as_of, "key": existing.asset_key},
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
                    "as_of": as_of,
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
    rows = 0
    with engine.begin() as conn:
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


def derive_fact_holdings(portfolio_id: int, source_system: str, run_id: str) -> int:
    """Recompute holdings for every trading day between the portfolio's
    first transaction and today, by cumulative-summing BUY/SELL
    quantity per asset. Full recompute here (small dataset); production
    would advance incrementally from the last processed date.
    """
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM fact_holdings WHERE portfolio_id = :p"), {"p": portfolio_id})
        conn.execute(
            sa.text(
                """
                WITH daily_moves AS (
                    -- Aggregate same-day, same-asset BUY/SELL activity to
                    -- one row per (asset, date) BEFORE windowing, so a day
                    -- with two transactions doesn't produce two holdings
                    -- rows for that day.
                    SELECT
                        asset_key,
                        date_key,
                        SUM(CASE transaction_type WHEN 'BUY' THEN quantity
                                                   WHEN 'SELL' THEN -quantity
                                                   ELSE 0 END) AS qty_delta,
                        SUM(CASE transaction_type WHEN 'BUY' THEN price_inr * quantity + fees
                                                   WHEN 'SELL' THEN -(price_inr * quantity - fees - tax)
                                                   ELSE 0 END) AS cost_delta
                    FROM fact_transactions
                    WHERE portfolio_id = :portfolio_id
                      AND transaction_type IN ('BUY','SELL')
                    GROUP BY asset_key, date_key
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
                    SELECT DISTINCT asset_key FROM fact_transactions
                    WHERE portfolio_id = :portfolio_id AND transaction_type IN ('BUY','SELL')
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
