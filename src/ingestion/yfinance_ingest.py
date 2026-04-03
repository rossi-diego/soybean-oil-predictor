"""Ingest commodity futures prices from Yahoo Finance into Bronze parquet files.

Uses yfinance to fetch OHLCV data for CBOT/NYMEX/ICE contracts.
Writes append-only Parquet files to data/bronze/.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import DATA_BRONZE, TICKERS
from src.log import get_logger

logger = get_logger(__name__)

DEFAULT_START_DATE = "2010-01-01"


def fetch_single_ticker(
    ticker_symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download historical daily data for a single ticker.

    Args:
        ticker_symbol: Yahoo Finance ticker (e.g. ``BO=F``).
        start: Start date (YYYY-MM-DD). Defaults to 2010-01-01.
        end: End date (YYYY-MM-DD). Defaults to today.

    Returns:
        DataFrame with columns [date, open, high, low, close, volume, ticker].
    """
    start = start or DEFAULT_START_DATE
    end = end or datetime.now().strftime("%Y-%m-%d")

    logger.info("fetching_ticker", ticker=ticker_symbol, start=start, end=end)

    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        logger.warning("empty_response", ticker=ticker_symbol)
        return pd.DataFrame()

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    df["ticker"] = ticker_symbol
    df = df[["date", "open", "high", "low", "close", "volume", "ticker"]]

    logger.info("fetched_ticker", ticker=ticker_symbol, rows=len(df))
    return df


def ingest_futures_prices(
    tickers: dict[str, str] | None = None,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch all commodity futures and write to Bronze parquet.

    Args:
        tickers: Mapping of internal name to Yahoo ticker symbol.
            Defaults to :data:`src.config.TICKERS`.
        start: Start date for historical data.
        end: End date for historical data.
        output_dir: Directory for parquet output. Defaults to data/bronze/.

    Returns:
        Combined DataFrame of all tickers.
    """
    tickers = tickers or TICKERS
    output_dir = output_dir or DATA_BRONZE
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for name, symbol in tickers.items():
        try:
            df = fetch_single_ticker(symbol, start=start, end=end)
            if not df.empty:
                df["name"] = name
                frames.append(df)
        except Exception:
            logger.exception("ingestion_failed", ticker=symbol, name=name)

    if not frames:
        logger.error("no_data_ingested")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    output_path = output_dir / "futures_prices.parquet"
    combined.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("bronze_written", path=str(output_path), rows=len(combined))

    return combined


def ingest_incremental(
    tickers: dict[str, str] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Append-only incremental ingestion of futures prices.

    Reads the existing Bronze parquet to find the latest date,
    then fetches only new data from the day after.

    Args:
        tickers: Mapping of internal name to Yahoo ticker symbol.
        output_dir: Directory for parquet files.

    Returns:
        DataFrame of newly ingested rows.
    """
    tickers = tickers or TICKERS
    output_dir = output_dir or DATA_BRONZE
    output_path = output_dir / "futures_prices.parquet"

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        latest_date = pd.to_datetime(existing["date"]).max()
        start = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("incremental_start", from_date=start)
    else:
        start = DEFAULT_START_DATE
        existing = pd.DataFrame()

    new_data = ingest_futures_prices(
        tickers=tickers, start=start, output_dir=output_dir
    )

    if not new_data.empty and not existing.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        combined.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("incremental_merged", total_rows=len(combined))

    return new_data
