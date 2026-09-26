-- 1:1 with fact_daily_prices. Light cleanup only: resolve date_key to
-- an actual date, cast types, rename for clarity.
select
    asset_key,
    d.full_date     as price_date,
    open::numeric    as open,
    high::numeric    as high,
    low::numeric     as low,
    close::numeric   as close,
    volume::bigint   as volume,
    source_system,
    ingested_at,
    pipeline_run_id
from {{ source('warehouse', 'fact_daily_prices') }} p
left join {{ source('warehouse', 'dim_date') }} d on d.date_key = p.date_key
