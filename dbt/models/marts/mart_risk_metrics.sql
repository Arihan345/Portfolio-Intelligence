-- Grain: one row per portfolio per day -- rolling risk metrics as of
-- that day.
-- Computable now from the real daily return series: trailing 30-day
-- volatility (annualized) and max drawdown to date.
-- TODO (Phase 5 analytics engine):
--   - Sharpe ratio = (annualized_return - risk_free_rate) / annualized_volatility
--     Not computed here: no risk-free rate series is ingested yet
--     (e.g. India 10Y G-Sec or 91-day T-Bill yield).
--   - Value at Risk (VaR) / Conditional VaR: requires either a fitted
--     return distribution or the Monte Carlo simulation engine's
--     resampled return paths (Phase 5), not a closed-form calc on this
--     short a real history.
--   - Beta vs. benchmark: requires mart_benchmark to have real index
--     return data loaded (see that model's TODO).
with returns as (
    select portfolio_id, value_date, daily_return, total_nav_inr
    from {{ ref('int_portfolio_returns') }}
),

rolling as (
    select
        portfolio_id,
        value_date,
        stddev_samp(daily_return) over (
            partition by portfolio_id order by value_date
            rows between 29 preceding and current row
        ) * sqrt(252) as volatility_30d_annualized,
        max(total_nav_inr) over (
            partition by portfolio_id order by value_date
            rows between unbounded preceding and current row
        ) as running_peak_value
    from returns
)

select
    r.portfolio_id,
    r.value_date,
    r.volatility_30d_annualized,
    ret.total_nav_inr,
    r.running_peak_value,
    case when r.running_peak_value > 0
         then (ret.total_nav_inr - r.running_peak_value) / r.running_peak_value
         else null
    end as drawdown_from_peak
from rolling r
join returns ret on ret.portfolio_id = r.portfolio_id and ret.value_date = r.value_date
