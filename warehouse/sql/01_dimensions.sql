-- =====================================================================
-- DIMENSION TABLES
-- =====================================================================

-- dim_date
-- Grain: one row per calendar day.
-- PK: date_key (int, YYYYMMDD) chosen over a native DATE PK so fact
--     tables join/filter on a cheap integer and can be partitioned or
--     indexed identically across all fact tables.
-- FKs: none (root dimension).
-- Attributes: calendar breakdown + is_trading_day flag.
-- Update frequency: generated once, in bulk, for the full date range
--     the platform will ever need; never updated incrementally.
-- Feeds from: pre-generated, not sourced from any pipeline stage.
CREATE TABLE dim_date (
    date_key        INT PRIMARY KEY,               -- e.g. 20240601
    full_date       DATE NOT NULL UNIQUE,
    day_of_month    SMALLINT NOT NULL,
    day_of_week     SMALLINT NOT NULL,              -- 1=Mon .. 7=Sun
    day_name        TEXT NOT NULL,
    week_of_year    SMALLINT NOT NULL,
    month_num       SMALLINT NOT NULL,
    month_name      TEXT NOT NULL,
    quarter         SMALLINT NOT NULL,
    year            SMALLINT NOT NULL,
    -- Approximated as "not a weekend" for now; a real NSE/BSE holiday
    -- calendar is a documented follow-up, not in Phase 3 scope.
    is_trading_day  BOOLEAN NOT NULL
);

-- dim_currency
-- Grain: one row per ISO currency code.
-- PK: currency_code (natural key, e.g. 'INR').
-- FKs: none.
-- Update frequency: near-static reference table.
-- Feeds from: manually seeded reference data.
CREATE TABLE dim_currency (
    currency_code   CHAR(3) PRIMARY KEY,
    currency_name   TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    decimal_places  SMALLINT NOT NULL DEFAULT 2
);

-- dim_user
-- Grain: one row per platform user.
-- PK: user_id (surrogate).
-- FKs: none.
-- Update frequency: low; new row on signup.
-- Feeds from: application/auth layer (out of scope for this pipeline;
--     seeded directly here for Phase 3).
CREATE TABLE dim_user (
    user_id         SERIAL PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    display_name    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- dim_portfolio
-- Grain: one row per portfolio. A user may own multiple portfolios
--     (modeled from day one per Phase 3 requirement) even though the
--     working example only populates one.
-- PK: portfolio_id (surrogate).
-- FKs: user_id -> dim_user, base_currency_code -> dim_currency.
-- Update frequency: low; new row when a user creates a portfolio.
-- Feeds from: application layer / portfolio CSV upload metadata.
CREATE TABLE dim_portfolio (
    portfolio_id        SERIAL PRIMARY KEY,
    user_id             INT NOT NULL REFERENCES dim_user(user_id),
    portfolio_name      TEXT NOT NULL,
    base_currency_code  CHAR(3) NOT NULL REFERENCES dim_currency(currency_code),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_dim_portfolio_user ON dim_portfolio(user_id);

-- dim_asset (SCD Type 2)
-- Grain: one row per ticker per period during which its attributes
--     (notably sector/industry) were valid. A ticker with a
--     reclassified sector produces a NEW row, not an in-place update,
--     so historical facts keep joining to the classification that was
--     true at the time the fact occurred (point-in-time correctness
--     for sector attribution).
-- PK: asset_key (surrogate; this is what facts join on).
-- Natural key: ticker (stable across SCD versions).
-- FKs: currency_code -> dim_currency.
-- Sector/industry are embedded here rather than split into separate
--     dim_sector/dim_industry tables: in this schema they are pure
--     classification attributes of an asset with no independent facts
--     or hierarchy of their own, so normalizing them out would only
--     add a mandatory join to every analytics query for no benefit.
--     Revisit only if sector-level facts (e.g. sector benchmark
--     series) are ever needed independent of individual assets.
-- Update frequency: low; new version row only on reclassification.
-- Feeds from: cleaned market metadata (fetch_sector_industry) +
--     cleaned ticker list from portfolio ingestion.
CREATE TABLE dim_asset (
    asset_key       SERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL,
    asset_name      TEXT,
    sector          TEXT,
    industry        TEXT,
    exchange        TEXT,               -- NSE / BSE
    asset_type      TEXT NOT NULL DEFAULT 'EQUITY',
    currency_code   CHAR(3) NOT NULL REFERENCES dim_currency(currency_code),
    effective_from  DATE NOT NULL,
    effective_to    DATE NOT NULL DEFAULT '9999-12-31',
    is_current      BOOLEAN NOT NULL DEFAULT TRUE,
    source_system   TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id TEXT
);
CREATE INDEX idx_dim_asset_ticker ON dim_asset(ticker);
CREATE UNIQUE INDEX uq_dim_asset_current ON dim_asset(ticker) WHERE is_current;
