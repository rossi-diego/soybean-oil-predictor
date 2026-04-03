"""Ingest WASDE (World Agricultural Supply and Demand Estimates) data from USDA.

WASDE reports are the single most important fundamental data source for
grains and oilseeds. Published monthly by the USDA, they contain production,
consumption, export, and ending stocks estimates that move markets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_BRONZE, Settings
from src.log import get_logger

logger = get_logger(__name__)

WASDE_CSV_URL = (
    "https://www.usda.gov/oce/commodity/wasde/wasde-report-data.csv"
)

OILSEED_COMMODITIES = [
    "Soybean Oil",
    "Soybeans",
    "Soybean Meal",
    "Corn",
    "Palm Oil",
]


def fetch_wasde_report(url: str | None = None) -> pd.DataFrame:
    """Download the full WASDE historical CSV from USDA.

    Args:
        url: URL to the WASDE CSV. Defaults to the official USDA endpoint.

    Returns:
        Raw DataFrame of WASDE data.
    """
    url = url or WASDE_CSV_URL
    logger.info("fetching_wasde", url=url)

    df = pd.read_csv(url, low_memory=False)
    logger.info("wasde_fetched", rows=len(df), columns=list(df.columns))
    return df


def filter_oilseed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter WASDE data to oilseed-relevant commodities.

    Args:
        df: Raw WASDE DataFrame.

    Returns:
        Filtered DataFrame with only oilseed commodities.
    """
    if "Commodity" not in df.columns:
        logger.warning("no_commodity_column", columns=list(df.columns))
        return df

    mask = df["Commodity"].isin(OILSEED_COMMODITIES)
    filtered = df[mask].copy()
    logger.info("wasde_filtered", rows=len(filtered))
    return filtered


def ingest_wasde(
    output_dir: Path | None = None,
    url: str | None = None,
) -> pd.DataFrame:
    """Fetch WASDE data and write to Bronze parquet.

    Args:
        output_dir: Output directory. Defaults to data/bronze/.
        url: Optional custom URL for the WASDE CSV.

    Returns:
        Filtered WASDE DataFrame.
    """
    output_dir = output_dir or DATA_BRONZE
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = fetch_wasde_report(url)
    filtered = filter_oilseed_data(raw)

    if filtered.empty:
        logger.warning("wasde_empty_after_filter")
        return filtered

    output_path = output_dir / "wasde_reports.parquet"
    filtered.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("wasde_bronze_written", path=str(output_path), rows=len(filtered))

    return filtered
