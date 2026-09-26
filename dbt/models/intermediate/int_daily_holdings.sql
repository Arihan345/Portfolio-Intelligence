-- Grain: one row per (portfolio, asset, day) between that asset's first
-- transaction and today. Computes the point-in-time holding quantity by
-- walking BUY/SELL transactions chronologically and cumulative-summing
-- signed quantity (BUY = +qty, SELL = -qty). This is the model the
-- brief's worked example checks directly: 10 TCS.NS bought 2024-01-01,
-- 3 sold 2024-06-01 must show 10 from Jan 1 and 7 from Jun 1 onward.
--
-- Same-day multiple transactions for one asset are summed BEFORE the
-- running total is computed (daily_moves CTE), so two trades landing on
-- the same day never produce two holdings rows for that day.
with tradeable as (
    select *
    from {{ ref('stg_transactions') }}
    where transaction_type in ('BUY', 'SELL')
),

daily_moves as (
    select
        portfolio_id,
        asset_key,
        txn_date,
        sum(case transaction_type
                when 'BUY' then quantity
                when 'SELL' then -quantity
            end) as qty_delta,
        sum(case transaction_type
                when 'BUY' then price_inr * quantity + fees
                when 'SELL' then -(price_inr * quantity - fees - tax)
            end) as cost_delta
    from tradeable
    group by portfolio_id, asset_key, txn_date
),

bounds as (
    select
        portfolio_id,
        asset_key,
        min(txn_date) as start_date
    from tradeable
    group by portfolio_id, asset_key
),

calendar as (
    select
        b.portfolio_id,
        b.asset_key,
        d.full_date as as_of_date
    from bounds b
    cross join {{ source('warehouse', 'dim_date') }} d
    where d.full_date between b.start_date and current_date
)

select
    c.portfolio_id,
    c.asset_key,
    c.as_of_date,
    coalesce(sum(m.qty_delta) over (
        partition by c.portfolio_id, c.asset_key order by c.as_of_date
        rows between unbounded preceding and current row
    ), 0) as quantity_held,
    coalesce(sum(m.cost_delta) over (
        partition by c.portfolio_id, c.asset_key order by c.as_of_date
        rows between unbounded preceding and current row
    ), 0) as cost_basis_inr
from calendar c
left join daily_moves m
    on m.portfolio_id = c.portfolio_id
   and m.asset_key = c.asset_key
   and m.txn_date = c.as_of_date
