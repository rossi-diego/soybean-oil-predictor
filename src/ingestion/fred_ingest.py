"""Ingest macroeconomic indicators from FRED (Federal Reserve Economic Data).

Provides context indicators that influence commodity prices:
- DXY (US Dollar Index proxy via DTWEXBGS)
- WTI Crude Oil (DCOILWTICO)
- 10Y Treasury Yield (DGS10)
- Brazilian Real exchange rate (DEXBZUS)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_datareader.data as web

from src.config import DATA_BRONZE
from src.log import get_logger

logger = get_logger(__name__)

FRED_SERIES = {
    "dxy_proxy": "DTWEXBGS",        # Trade-weighted USD index (broad)
    "wti_crude": "DCOILWTICO",       # WTI crude oil spot price
    "treasury_10y": "DGS10",         # 10-Year Treasury yield
    "brl_usd": "DEXBZUS",           # Brazilian Real per USD
    "cpi_yoy": "CPIAUCSL",          # Consumer Price Index (inflation proxy)
}

DEFAULT_START = "2010-01-01"


def fetch_fred_series(
    series_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch a single FRED time series.

    Args:
        series_id: FRED series identifier (e.g. ``DCOILWTICO``).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        DataFrame with columns [date, value, series_id].
    """
    start = start or DEFAULT_START
    logger.info("fetching_fred", series_id=series_id, start=start)

    df = web.DataReader(series_id, "fred", start=start, end=end)
    df = df.reset_index()
    df.columns = ["date", "value"]
    df["series_id"] = series_id
    df = df.dropna(subset=["value"])

    logger.info("fred_fetched", series_id=series_id, rows=len(df))
    return df


def ingest_fred_series(
    series_map: dict[str, str] | None = None,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch all configured FRED series and write to Bronze parquet.

    Args:
        series_map: Mapping of internal name to FRED series ID.
            Defaults to :data:`FRED_SERIES`.
        start: Start date for historical data.
        end: End date for historical data.
        output_dir: Directory for parquet output.

    Returns:
        Combined DataFrame of all FRED series.
    """
    series_map = series_map or FRED_SERIES
    output_dir = output_dir or DATA_BRONZE
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for name, series_id in series_map.items():
        try:
            df = fetch_fred_series(series_id, start=start, end=end)
            df["name"] = name
            frames.append(df)
        except Exception:
            logger.exception("fred_fetch_failed", series_id=series_id, name=name)

    if not frames:
        logger.error("no_fred_data_ingested")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    output_path = output_dir / "fred_macro.parquet"
    combined.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("fred_bronze_written", path=str(output_path), rows=len(combined))

    return combined
