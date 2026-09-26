-- Grain: one row per portfolio per day. Corrected total NAV = holdings
-- market value (forward-filled last close, same non-trading-day
-- handling as Phase 3's fact_portfolio_value) + running cash balance
-- from int_cash_flow. This replaces reading fact_portfolio_value
-- directly for return calculations, because that table has no cash
-- leg and shows a fake -24.8% "return" on 2024-06-01 (the day of the
-- TCS.NS sale) purely from the mechanics of selling, not a market move.
with holdings_value as (
    select
        h.portfolio_id,
        h.as_of_date,
        sum(h.quantity_held * coalesce((
            select p.close from {{ ref('stg_prices') }} p
            where p.asset_key = h.asset_key and p.price_date <= h.as_of_date
            order by p.price_date desc limit 1
        ), 0)) as holdings_market_value_inr
    from {{ ref('int_daily_holdings') }} h
    group by h.portfolio_id, h.as_of_date
),

running_cash as (
    select
        portfolio_id,
        txn_date,
        sum(cash_delta) over (
            partition by portfolio_id order by txn_date
            rows between unbounded preceding and current row
        ) as cash_balance_inr
    from {{ ref('int_cash_flow') }}
)

select
    hv.portfolio_id,
    hv.as_of_date as value_date,
    hv.holdings_market_value_inr,
    -- carry the last cash balance forward to days with no transaction
    (
        select rc.cash_balance_inr from running_cash rc
        where rc.portfolio_id = hv.portfolio_id and rc.txn_date <= hv.as_of_date
        order by rc.txn_date desc limit 1
    ) as cash_balance_inr,
    hv.holdings_market_value_inr + coalesce((
        select rc.cash_balance_inr from running_cash rc
        where rc.portfolio_id = hv.portfolio_id and rc.txn_date <= hv.as_of_date
        order by rc.txn_date desc limit 1
    ), 0) as total_nav_inr
from holdings_value hv
