-- Grain: one row per portfolio per asset per day.
-- Computable now: quantity held, cost basis, current market value,
-- unrealized gain/loss (market value vs. remaining cost basis).
-- TODO (Phase 5 analytics engine): realized gain/loss per SELL event
-- requires tax-lot accounting (FIFO/LIFO/specific-lot) to know which
-- purchase lot's cost basis a given sale consumed -- this model's
-- cost_basis_inr is a running weighted-average remaining basis, not a
-- lot ledger, so it cannot answer "what was the realized gain on the
-- 2024-06-01 sale" by itself. That needs a dedicated lot-tracking model.
-- Non-trading days (weekends/holidays) have no stg_prices row for that
-- exact date, so the last known close is carried forward -- otherwise
-- a Saturday snapshot would price every holding at zero (same issue
-- fixed in Phase 3's fact_portfolio_value load).
select
    h.portfolio_id,
    h.asset_key,
    a.ticker,
    a.sector,
    a.industry,
    h.as_of_date,
    h.quantity_held,
    h.cost_basis_inr,
    (
        select p.close from {{ ref('stg_prices') }} p
        where p.asset_key = h.asset_key and p.price_date <= h.as_of_date
        order by p.price_date desc limit 1
    ) as last_close_inr,
    h.quantity_held * coalesce((
        select p.close from {{ ref('stg_prices') }} p
        where p.asset_key = h.asset_key and p.price_date <= h.as_of_date
        order by p.price_date desc limit 1
    ), 0) as market_value_inr,
    (h.quantity_held * coalesce((
        select p.close from {{ ref('stg_prices') }} p
        where p.asset_key = h.asset_key and p.price_date <= h.as_of_date
        order by p.price_date desc limit 1
    ), 0)) - h.cost_basis_inr as unrealized_gain_inr
from {{ ref('int_daily_holdings') }} h
join {{ ref('stg_assets') }} a
    on a.asset_key = h.asset_key
   and h.as_of_date between a.effective_from and a.effective_to
