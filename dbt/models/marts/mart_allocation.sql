-- Grain: one row per portfolio per asset per day -- allocation weight
-- of each asset (and its sector) within the portfolio on that day.
-- Fully computable now from mart_asset_performance; no Phase 5
-- dependency.
select
    ap.portfolio_id,
    ap.asset_key,
    ap.ticker,
    ap.sector,
    ap.industry,
    ap.as_of_date,
    ap.market_value_inr,
    sum(ap.market_value_inr) over (
        partition by ap.portfolio_id, ap.as_of_date
    ) as portfolio_total_value_inr,
    case when sum(ap.market_value_inr) over (partition by ap.portfolio_id, ap.as_of_date) > 0
         then ap.market_value_inr / sum(ap.market_value_inr) over (partition by ap.portfolio_id, ap.as_of_date)
         else null
    end as weight_in_portfolio
from {{ ref('mart_asset_performance') }} ap
