-- Bronze layer: raw futures prices from yfinance ingestion.
-- Loaded from Parquet files written by src/ingestion/yfinance_ingest.py.
-- No transformations — preserves the raw schema.

{{ config(materialized='table') }}

select
    date,
    open,
    high,
    low,
    close,
    volume,
    ticker,
    name,
    current_timestamp as _loaded_at
from read_parquet('../data/bronze/futures_prices.parquet')
