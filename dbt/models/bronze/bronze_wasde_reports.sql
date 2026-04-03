-- Bronze layer: WASDE report data from USDA ingestion.
-- Loaded from Parquet files written by src/ingestion/usda_ingest.py.

{{ config(materialized='table') }}

select
    *,
    current_timestamp as _loaded_at
from read_parquet('../data/bronze/wasde_reports.parquet')
