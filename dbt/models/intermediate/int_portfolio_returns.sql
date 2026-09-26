-- Grain: one row per portfolio per day. Daily return computed from
-- day-over-day change in int_portfolio_value.total_nav_inr (holdings
-- market value + cash balance), NOT the warehouse's raw
-- fact_portfolio_value -- that table has no cash leg and produces a
-- fake return on every trade day (see int_portfolio_value's header
-- comment for the -24.8% bug this fixes).
-- This is still a simple price-return, not cash-flow-weighted
-- (TWR/XIRR); see mart_portfolio_performance's TODO for that.
select
    portfolio_id,
    value_date,
    holdings_market_value_inr,
    cash_balance_inr,
    total_nav_inr,
    case when lag(total_nav_inr) over (partition by portfolio_id order by value_date) is not null
              and lag(total_nav_inr) over (partition by portfolio_id order by value_date) != 0
         then (total_nav_inr - lag(total_nav_inr) over (partition by portfolio_id order by value_date))
              / abs(lag(total_nav_inr) over (partition by portfolio_id order by value_date))
         else null
    end as daily_return
from {{ ref('int_portfolio_value') }}
