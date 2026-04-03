-- Silver layer: cleaned WASDE supply & demand data.
-- Standardizes column names and filters to relevant commodities.

{{ config(materialized='table') }}

select
    *,
    current_timestamp as _transformed_at
from {{ ref('bronze_wasde_reports') }}
