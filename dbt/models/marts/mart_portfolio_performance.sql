-- Grain: one row per portfolio per day.
-- Computable now: holdings market value, cash balance, total NAV,
-- simple daily/cumulative return -- all sourced from
-- int_portfolio_returns, which already corrects for the cash-leg bug
-- in the warehouse's raw fact_portfolio_value (see that model's header
-- comment: a SELL used to register as a fake ~-25% one-day loss).
-- TODO (Phase 5 analytics engine): time-weighted return (TWR) and
-- money-weighted return (XIRR/Modified Dietz) that correctly handle
-- REAL external cash flows (actual DEPOSIT/WITHDRAWAL transactions,
-- once the example data includes any) rather than the naive
-- value(t)/value(t-1) ratio used here. Formula for Modified Dietz:
--   R = (V1 - V0 - CF) / (V0 + sum(CF_i * (1 - t_i/T)))
-- where CF is net external cash flow in the period and t_i is the day
-- fraction remaining after cash flow i.
select
    portfolio_id,
    value_date,
    holdings_market_value_inr,
    cash_balance_inr,
    total_nav_inr,
    daily_return,
    exp(sum(ln(1 + coalesce(daily_return, 0))) over (
        partition by portfolio_id order by value_date
    )) - 1 as cumulative_return
from {{ ref('int_portfolio_returns') }}
