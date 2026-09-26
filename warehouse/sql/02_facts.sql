-- =====================================================================
-- FACT TABLES
-- =====================================================================

-- fact_transactions
-- Grain: one row per transaction event exactly as recorded in the
--     source CSV (post validation + cleaning). This is the finest
--     grain available and the only one that preserves audit
--     correctness: two BUYs of the same ticker on the same day must
--     stay distinct rows (different lots, different fees/tax), never
--     summed, or cost-basis and tax-lot analytics downstream would be
--     silently wrong.
-- PK: transaction_id (surrogate).
-- FKs: portfolio_id -> dim_portfolio, asset_key -> dim_asset
--     (point-in-time version as of transaction date),
--     date_key -> dim_date, currency_code -> dim_currency.
-- Measures: quantity, price, price_inr, fees, tax.
-- Update frequency: append-only, one batch per ingestion run;
--     idempotent upsert on (portfolio_id, ticker, txn_date,
--     transaction_type, quantity, price) to make retries safe.
-- Feeds from: cleaning/standardization stage output.
-- fact_dividends (separate deliverable) is a VIEW over this table
--     filtered to transaction_type = 'DIVIDEND', not a physical
--     table: dividends are structurally just transactions with a
--     price field that means "amount per share" instead of "trade
--     price", so duplicating them into their own loaded table would
--     create two sources of truth that can drift on independent
--     loads.
CREATE TABLE fact_transactions (
    transaction_id      SERIAL PRIMARY KEY,
    portfolio_id         INT NOT NULL REFERENCES dim_portfolio(portfolio_id),
    asset_key             INT NOT NULL REFERENCES dim_asset(asset_key),
    date_key              INT NOT NULL REFERENCES dim_date(date_key),
    transaction_type       TEXT NOT NULL CHECK (transaction_type IN
                            ('BUY','SELL','DIVIDEND','DEPOSIT','WITHDRAWAL')),
    quantity                NUMERIC(18,6) NOT NULL DEFAULT 0,
    price                   NUMERIC(18,6) NOT NULL DEFAULT 0,
    price_inr               NUMERIC(18,6),
    fees                    NUMERIC(18,6) NOT NULL DEFAULT 0,
    tax                     NUMERIC(18,6) NOT NULL DEFAULT 0,
    currency_code           CHAR(3) NOT NULL REFERENCES dim_currency(currency_code),
    near_duplicate_flag     BOOLEAN NOT NULL DEFAULT FALSE,
    price_anomaly_flag      BOOLEAN NOT NULL DEFAULT FALSE,
    source_system           TEXT NOT NULL,
    ingested_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id         TEXT,
    UNIQUE (portfolio_id, asset_key, date_key, transaction_type, quantity, price)
);
CREATE INDEX idx_fact_txn_portfolio ON fact_transactions(portfolio_id);
CREATE INDEX idx_fact_txn_asset ON fact_transactions(asset_key);
CREATE INDEX idx_fact_txn_date ON fact_transactions(date_key);

-- fact_daily_prices
-- Grain: one row per asset per trading day. This is the natural grain
--     of OHLCV data exactly as delivered by the market data provider
--     -- no aggregation choice to justify here, the source already
--     arrives at this grain.
-- PK: (asset_key, date_key) composite.
-- FKs: asset_key -> dim_asset, date_key -> dim_date.
-- Measures: open, high, low, close, volume.
-- Update frequency: daily append (one new row per asset per trading
--     day); idempotent upsert so re-fetching an overlapping range is
--     always safe.
-- Feeds from: MarketDataProvider.fetch_ohlcv output.
CREATE TABLE fact_daily_prices (
    asset_key       INT NOT NULL REFERENCES dim_asset(asset_key),
    date_key        INT NOT NULL REFERENCES dim_date(date_key),
    open            NUMERIC(18,6),
    high            NUMERIC(18,6),
    low             NUMERIC(18,6),
    close           NUMERIC(18,6),
    volume          BIGINT,
    source_system   TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id TEXT,
    PRIMARY KEY (asset_key, date_key)
);
CREATE INDEX idx_fact_prices_date ON fact_daily_prices(date_key);

-- fact_holdings
-- Grain: one row per portfolio per asset per day, representing the
--     point-in-time held quantity (and running cost basis) as of that
--     day's close.
-- Materialize vs. recompute tradeoff: recomputing holdings on demand
--     means replaying every transaction up to the requested date for
--     every query -- O(transaction count) work repeated by every
--     analytics module (performance, risk, Monte Carlo) on every call,
--     which gets worse as history grows. Materializing costs bounded
--     storage (portfolios x assets x trading days) and a daily
--     incremental job (advance from the last processed date, not a
--     full replay), but turns every downstream read into a single
--     point lookup. Given this platform re-reads holdings from many
--     independent consumers, materializing wins.
-- PK: (portfolio_id, asset_key, date_key).
-- FKs: portfolio_id -> dim_portfolio, asset_key -> dim_asset,
--     date_key -> dim_date.
-- Measures: quantity_held, cost_basis_inr (cumulative invested capital
--     still held).
-- Update frequency: daily incremental recompute from fact_transactions.
-- Feeds from: derived/computed from fact_transactions, not raw-loaded.
CREATE TABLE fact_holdings (
    portfolio_id    INT NOT NULL REFERENCES dim_portfolio(portfolio_id),
    asset_key       INT NOT NULL REFERENCES dim_asset(asset_key),
    date_key        INT NOT NULL REFERENCES dim_date(date_key),
    quantity_held   NUMERIC(18,6) NOT NULL,
    cost_basis_inr  NUMERIC(18,6) NOT NULL,
    source_system   TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id TEXT,
    PRIMARY KEY (portfolio_id, asset_key, date_key)
);
CREATE INDEX idx_fact_holdings_date ON fact_holdings(date_key);
CREATE INDEX idx_fact_holdings_portfolio ON fact_holdings(portfolio_id);

-- fact_portfolio_value
-- Grain: one row per portfolio per day -- the daily mark-to-market
--     NAV. Coarser than fact_holdings by design: this is the
--     portfolio-level rollup (sum of quantity_held * close across all
--     assets, plus cash) that Streamlit/Power BI chart directly,
--     rather than making every KPI consumer re-aggregate fact_holdings
--     x fact_daily_prices itself.
-- PK: (portfolio_id, date_key).
-- FKs: portfolio_id -> dim_portfolio, date_key -> dim_date.
-- Measures: market_value, invested_capital, total_value.
-- Update frequency: daily append, derived from fact_holdings joined to
--     fact_daily_prices.
-- Feeds from: derived/computed, not raw-loaded.
CREATE TABLE fact_portfolio_value (
    portfolio_id      INT NOT NULL REFERENCES dim_portfolio(portfolio_id),
    date_key          INT NOT NULL REFERENCES dim_date(date_key),
    market_value_inr  NUMERIC(18,6) NOT NULL,
    invested_capital_inr NUMERIC(18,6) NOT NULL,
    source_system     TEXT NOT NULL,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id   TEXT,
    PRIMARY KEY (portfolio_id, date_key)
);
CREATE INDEX idx_fact_pval_date ON fact_portfolio_value(date_key);

-- fact_portfolio_returns
-- Grain: one row per portfolio per day, precomputed daily return.
-- Precompute rationale: return calculation that correctly handles
--     external cash flows (deposits/withdrawals distorting a naive
--     value(t)/value(t-1) ratio) requires non-trivial logic (e.g.
--     Modified Dietz) that must be implemented once, tested, and
--     trusted -- not reimplemented ad hoc inside every Streamlit chart
--     or Monte Carlo bootstrap sampler, where subtly different
--     reimplementations would silently disagree.
-- PK: (portfolio_id, date_key).
-- FKs: portfolio_id -> dim_portfolio, date_key -> dim_date.
-- Measures: daily_return, cumulative_return.
-- Update frequency: daily append, derived from fact_portfolio_value.
-- Feeds from: derived/computed, not raw-loaded.
CREATE TABLE fact_portfolio_returns (
    portfolio_id       INT NOT NULL REFERENCES dim_portfolio(portfolio_id),
    date_key           INT NOT NULL REFERENCES dim_date(date_key),
    daily_return        NUMERIC(18,8),
    cumulative_return   NUMERIC(18,8),
    source_system       TEXT NOT NULL,
    ingested_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id      TEXT,
    PRIMARY KEY (portfolio_id, date_key)
);
CREATE INDEX idx_fact_returns_date ON fact_portfolio_returns(date_key);

-- fact_dividends: derived view, not a physical table (see rationale
-- in the fact_transactions comment above).
CREATE VIEW fact_dividends AS
SELECT
    transaction_id,
    portfolio_id,
    asset_key,
    date_key,
    price          AS dividend_amount,
    price_inr      AS dividend_amount_inr,
    currency_code,
    source_system,
    ingested_at,
    pipeline_run_id
FROM fact_transactions
WHERE transaction_type = 'DIVIDEND';
