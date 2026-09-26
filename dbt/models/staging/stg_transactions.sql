-- 1:1 with fact_transactions. Light cleanup only: rename columns to
-- clearer names, resolve date_key to an actual date (still just a key
-- lookup, not business logic), cast types. No aggregation, no derived
-- measures, no filtering beyond the source's own grain.
select
    transaction_id,
    portfolio_id,
    asset_key,
    d.full_date                as txn_date,
    transaction_type,
    quantity::numeric           as quantity,
    price::numeric              as price,
    price_inr::numeric          as price_inr,
    fees::numeric                as fees,
    tax::numeric                  as tax,
    currency_code,
    near_duplicate_flag,
    price_anomaly_flag,
    source_system,
    ingested_at,
    pipeline_run_id
from {{ source('warehouse', 'fact_transactions') }} t
left join {{ source('warehouse', 'dim_date') }} d on d.date_key = t.date_key
