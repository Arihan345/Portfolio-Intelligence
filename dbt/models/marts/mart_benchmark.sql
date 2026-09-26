-- Grain: intended as one row per portfolio per day, comparing portfolio
-- daily/cumulative return against a benchmark index's return.
--
-- NOT populated yet -- this is a structural stub, not a faked result.
-- No benchmark index (e.g. NIFTY 50 / ^NSEI) has been ingested into
-- this warehouse: Phase 2's MarketDataProvider abstraction supports
-- fetching any ticker yfinance recognizes, so ^NSEI could be fetched
-- through the existing fetch_ohlcv() with no new code, but nothing has
-- loaded it into dim_asset (as asset_type = 'INDEX') or
-- fact_daily_prices yet. This model's join below will therefore return
-- zero rows until that load happens -- it is left in this state
-- deliberately rather than joining nothing and back-filling fake
-- benchmark numbers.
--
-- TODO (Phase 5 analytics engine, once benchmark data exists):
--   benchmark_daily_return = (close_t - close_t-1) / close_t-1
--   alpha = portfolio_return - benchmark_return
--   beta  = covariance(portfolio_return, benchmark_return) / variance(benchmark_return)
--     computed over a trailing window (e.g. 252 trading days)
select
    pp.portfolio_id,
    pp.value_date,
    pp.daily_return as portfolio_daily_return,
    bp.close as benchmark_close,
    (bp.close - lag(bp.close) over (order by pp.value_date))
        / nullif(lag(bp.close) over (order by pp.value_date), 0) as benchmark_daily_return
from {{ ref('mart_portfolio_performance') }} pp
join {{ ref('stg_assets') }} ba on ba.asset_type = 'INDEX' and ba.is_current
join {{ ref('stg_prices') }} bp on bp.asset_key = ba.asset_key and bp.price_date = pp.value_date
