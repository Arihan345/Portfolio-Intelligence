-- Grain: one row per portfolio per day -- running cash balance implied
-- by transaction activity NOT already reflected in holdings market
-- value.
--
-- Why this model exists: the warehouse's fact_portfolio_value tracks
-- only stock market value (sum of quantity_held * close). It has no
-- cash leg, so a SELL looks like wealth evaporating (holdings drop,
-- nothing records the cash the sale generated) and int_portfolio_value
-- would show a fake single-day crash on every sale (see that model's
-- header comment for the -24.8% example this fixes).
--
-- Modeling assumption: BUY transactions are treated as externally
-- funded at the moment of purchase (cash out is exactly offset by an
-- implicit capital contribution), so a BUY has ZERO net effect on this
-- balance -- standard practice for return calculations, where new
-- contributions establish a new cost baseline rather than counting as
-- a loss. No DEPOSIT transactions exist in the current example data,
-- so this assumption is currently implicit for every BUY; it would be
-- replaced by real DEPOSIT records once the platform captures them.
-- SELL, DIVIDEND, explicit DEPOSIT and WITHDRAWAL all have a real cash
-- effect and are counted here.
select
    portfolio_id,
    txn_date,
    sum(case transaction_type
            when 'SELL' then (price_inr * quantity - fees - tax)
            when 'DIVIDEND' then price_inr
            when 'DEPOSIT' then price_inr
            when 'WITHDRAWAL' then -price_inr
            else 0
        end) as cash_delta
from {{ ref('stg_transactions') }}
group by portfolio_id, txn_date
