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
--
-- Cross-exchange netting: a real brokerage (Groww) lets the same
-- underlying company be bought on one exchange and sold on the other
-- (e.g. BUY 5 MOREPENLAB.BO, SELL 5 MOREPENLAB.NS) -- NSE/BSE holdings
-- of one equity are fungible in demat, so this is one real position,
-- not two. dim_asset correctly keeps .NS/.BO as separate asset_keys
-- (their prices genuinely differ slightly and each needs its own price
-- series), but netting the running quantity per RAW asset_key treated
-- that BUY and SELL as unrelated -- the SELL had no same-ticker BUY to
-- net against and read as a permanently, impossibly negative quantity
-- (caught for real: dbt_utils_accepted_range on quantity_held failed
-- 1,579 rows after a real 30-ticker upload with several NSE/BSE pairs).
-- canonical_asset folds every base-symbol's transactions (regardless of
-- which exchange-suffixed ticker recorded them) onto ONE representative
-- asset_key purely for this quantity/cost-basis computation; dim_asset
-- and fact_transactions themselves are untouched.
--
-- That representative asset_key must be a CURRENTLY-VALID dim_asset row
-- (is_current), not just whichever has the lowest asset_key. dim_asset
-- is SCD2: re-uploading the same ticker with even a slightly reworded
-- sector/industry (e.g. real yfinance metadata replacing an earlier
-- upload's hand-typed "IT Services" with "Information Technology
-- Services") closes out the old asset_key row (effective_to = that
-- day) and inserts a new one -- and the OLD, now-closed row usually has
-- the LOWER asset_key, since it was inserted first. A plain min()
-- picked that closed-out row as canonical, and mart_asset_performance's
-- join to stg_assets on effective_from/effective_to validity then
-- silently dropped every day outside that old row's now-tiny validity
-- window -- confirmed real: TCS.NS collapsed from a full multi-year
-- daily series to exactly ONE row, which broke Attribution's
-- reconciliation identity by silently excluding TCS.NS's entire
-- contribution (contribution math itself was never wrong; the input
-- data was incomplete). is_current desc, asset_key asc picks the live
-- row when one exists, and falls back to the lowest asset_key only in
-- the genuine multi-exchange tie case (e.g. real SUZLON.NS/SUZLON.BO,
-- both simultaneously current).
--
-- cost_basis_inr is the WEIGHTED-AVERAGE remaining cost basis of the
-- quantity still held -- NOT a running net-cash-flow sum. A previous
-- version of this model computed cost_delta on a SELL as -(sale
-- PROCEEDS), which happens to equal cumulative-buys-minus-cumulative-
-- sells: for a position that's still fully held (no sells yet) that's
-- indistinguishable from a real cost basis, which is exactly why this
-- went unnoticed -- but it silently stops being a cost basis the
-- moment ANY sell happens. Confirmed for real: a fully-exited synthetic
-- portfolio (bought for a total of 2,165, later sold for 2,690 -- a
-- real 525 realized profit) showed a portfolio-level "cost basis" of
-- -525 (literally the net cash received), when the correct remaining
-- cost basis of a fully-exited position is 0 (nothing is held, so
-- there is nothing left to have a cost basis). The correct SELL
-- treatment removes cost PROPORTIONAL to the quantity sold
-- (quantity_sold / quantity_held_before * cost_basis_before), leaving
-- the per-share average cost of whatever remains unchanged -- standard
-- weighted-average-cost inventory accounting, and exactly what this
-- model's sibling mart_asset_performance.sql already documents
-- cost_basis_inr as being ("a running weighted-average remaining
-- basis"), just not what the SQL actually computed. This is inherently
-- sequential (each move's post-state depends on the previous move's
-- post-state, not a flat cumulative sum), so it needs a recursive walk
-- over each asset's own moves in date order rather than the previous
-- version's single window-function SUM.
with recursive tradeable as (
    select
        t.*,
        a.is_current,
        split_part(a.ticker, '.', 1) as base_symbol
    from {{ ref('stg_transactions') }} t
    join {{ ref('stg_assets') }} a on a.asset_key = t.asset_key
    where t.transaction_type in ('BUY', 'SELL')
),

canonical_asset as (
    select distinct on (portfolio_id, base_symbol)
        portfolio_id, base_symbol, asset_key
    from tradeable
    order by portfolio_id, base_symbol, is_current desc, asset_key asc
),

daily_moves as (
    select
        t.portfolio_id,
        ca.asset_key,
        t.txn_date,
        sum(case when t.transaction_type = 'BUY' then t.quantity else 0 end) as buy_qty,
        sum(case when t.transaction_type = 'BUY' then t.price_inr * t.quantity + t.fees else 0 end) as buy_cost,
        sum(case when t.transaction_type = 'SELL' then t.quantity else 0 end) as sell_qty
    from tradeable t
    join canonical_asset ca
        on ca.portfolio_id = t.portfolio_id and ca.base_symbol = t.base_symbol
    group by t.portfolio_id, ca.asset_key, t.txn_date
),

move_seq as (
    select
        *,
        row_number() over (partition by portfolio_id, asset_key order by txn_date) as rn
    from daily_moves
),

-- Recursive, chronological walk per (portfolio, asset): each move's
-- post-quantity/post-cost-basis depends on the PREVIOUS move's
-- post-state, which a flat window-function SUM cannot express once a
-- SELL's cost impact must be proportional rather than additive.
running as (
    select
        portfolio_id,
        asset_key,
        txn_date,
        rn,
        greatest(buy_qty - sell_qty, 0) as quantity_held,
        greatest(
            buy_cost - buy_cost * (sell_qty / nullif(buy_qty, 0)),
            0
        ) as cost_basis_inr
    from move_seq
    where rn = 1

    union all

    select
        m.portfolio_id,
        m.asset_key,
        m.txn_date,
        m.rn,
        r.quantity_held - m.sell_qty + m.buy_qty as quantity_held,
        greatest(
            (r.cost_basis_inr - r.cost_basis_inr * (m.sell_qty / nullif(r.quantity_held, 0))) + m.buy_cost,
            0
        ) as cost_basis_inr
    from move_seq m
    join running r
        on r.portfolio_id = m.portfolio_id
       and r.asset_key = m.asset_key
       and r.rn = m.rn - 1
),

bounds as (
    select
        portfolio_id,
        asset_key,
        min(txn_date) as start_date
    from daily_moves
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
    coalesce((
        select r.quantity_held from running r
        where r.portfolio_id = c.portfolio_id and r.asset_key = c.asset_key and r.txn_date <= c.as_of_date
        order by r.txn_date desc limit 1
    ), 0) as quantity_held,
    coalesce((
        select r.cost_basis_inr from running r
        where r.portfolio_id = c.portfolio_id and r.asset_key = c.asset_key and r.txn_date <= c.as_of_date
        order by r.txn_date desc limit 1
    ), 0) as cost_basis_inr
from calendar c
