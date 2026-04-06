-- Gold layer: pivoted feature matrix ready for ML training.
-- Pivots daily prices from long format into wide format with one
-- column per commodity, then computes derived features.

{{ config(materialized='table') }}

with pivoted as (
    select
        trade_date,
        max(case when commodity = 'boc1' then close_price end) as boc1,
        max(case when commodity = 'sc1' then close_price end) as sc1,
        max(case when commodity = 'smc1' then close_price end) as smc1,
        max(case when commodity = 'zc1' then close_price end) as zc1,
        max(case when commodity = 'lcoc1' then close_price end) as lcoc1,
        max(case when commodity = 'hoc1' then close_price end) as hoc1,
        max(case when commodity = 'palm_oil' then close_price end) as palm_oil,
        max(case when commodity = 'rsc1' then close_price end) as rsc1
    from {{ ref('silver_futures_daily') }}
    group by trade_date
),

with_spreads as (
    select
        *,
        -- Crush spread: (11 * oil + meal) - soybeans
        (11 * boc1 + smc1 - sc1) as crush_spread,
        -- Oil / palm spread
        (boc1 - palm_oil) as oil_palm_spread,
        -- Soy / corn ratio
        case when zc1 > 0 then sc1 / zc1 else null end as soy_corn_ratio,
        -- Oil share
        case when sc1 > 0 then (boc1 * 11) / sc1 else null end as oil_share,
        -- Calendar features
        extract(month from trade_date) as month_num,
        sin(2 * pi() * extract(month from trade_date) / 12) as month_sin,
        cos(2 * pi() * extract(month from trade_date) / 12) as month_cos,
        extract(quarter from trade_date) as quarter
    from pivoted
    where boc1 is not null
)

select *
from with_spreads
order by trade_date
