-- Regression for a real, confirmed bug: int_daily_holdings.sql used to
-- compute cost_basis_inr as a running net-cash-flow sum (BUY cost
-- minus SELL PROCEEDS) instead of a real weighted-average remaining
-- cost basis. A fully-exited position (quantity_held = 0) should have
-- NOTHING left to have a cost basis on -- confirmed real: a fully
-- exited synthetic portfolio (bought for 2,165 total, sold for 2,690
-- total, a real 525 realized profit) showed a portfolio-level cost
-- basis of -525 (literally the net cash received) instead of the
-- correct 0. A dbt test passes when this query returns ZERO rows.
select *
from {{ ref('int_daily_holdings') }}
where quantity_held = 0
  and abs(cost_basis_inr) > 0.01
