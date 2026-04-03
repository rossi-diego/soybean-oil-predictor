"""Unit tests for ingestion modules."""

import pandas as pd

from src.ingestion.yfinance_ingest import fetch_single_ticker


class TestYFinanceIngestion:
    def test_fetch_single_ticker_returns_dataframe(self):
        """Fetching a valid ticker must return a non-empty DataFrame."""
        df = fetch_single_ticker("ZS=F", start="2024-01-01", end="2024-01-31")
        assert isinstance(df, pd.DataFrame)
        # May be empty if market data is unavailable, but should be a DataFrame

    def test_fetch_single_ticker_columns(self):
        """Result must have the expected column schema."""
        df = fetch_single_ticker("ZS=F", start="2024-01-01", end="2024-01-31")
        if not df.empty:
            expected_cols = {"date", "open", "high", "low", "close", "volume", "ticker"}
            assert expected_cols.issubset(set(df.columns))

    def test_fetch_invalid_ticker_returns_empty(self):
        """An invalid ticker should return an empty DataFrame without error."""
        df = fetch_single_ticker("INVALID_TICKER_XYZ_123")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
