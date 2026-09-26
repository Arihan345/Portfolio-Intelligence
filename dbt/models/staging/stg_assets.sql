-- 1:1 with dim_asset. Exposes every SCD2 version row as-is (rename +
-- cast only); point-in-time resolution (which version applies to a
-- given date) is a join condition for consumers, not something this
-- staging model decides.
select
    asset_key,
    ticker,
    asset_name,
    sector,
    industry,
    exchange,
    asset_type,
    currency_code,
    effective_from,
    effective_to,
    is_current,
    source_system,
    ingested_at,
    pipeline_run_id
from {{ source('warehouse', 'dim_asset') }}
