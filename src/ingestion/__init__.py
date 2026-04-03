"""Data ingestion modules for Bronze layer."""

from src.ingestion.fred_ingest import ingest_fred_series
from src.ingestion.yfinance_ingest import ingest_futures_prices

__all__ = ["ingest_futures_prices", "ingest_fred_series"]
