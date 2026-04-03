-- Silver layer: cleaned, deduplicated daily futures prices.
-- Applies: type casting, deduplication, null handling.

{{ config(materialized='table') }}

with deduplicated as (
    select
        date::date as trade_date,
        name as commodity,
        ticker,
        close as close_price,
        open as open_price,
        high as high_price,
        low as low_price,
        volume::bigint as volume,
        _loaded_at,
        row_number() over (
            partition by date::date, name
            order by _loaded_at desc
        ) as rn
    from {{ ref('bronze_futures_prices') }}
    where close is not null
      and date is not null
)

select
    trade_date,
    commodity,
    ticker,
    close_price,
    open_price,
    high_price,
    low_price,
    volume,
    _loaded_at
from deduplicated
where rn = 1
order by trade_date, commodity
