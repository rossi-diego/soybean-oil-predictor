"""Calendar and seasonality features.

Commodity prices exhibit strong seasonality tied to planting, growing,
and harvest cycles. Cyclical encoding preserves the circular nature
of time (December is close to January, not 11 months away).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)


def cyclical_month(dates: pd.Series) -> pd.DataFrame:
    """Encode month as sine/cosine pair for cyclical continuity.

    Args:
        dates: Series of datetime values.

    Returns:
        DataFrame with columns ``month_sin`` and ``month_cos``.
    """
    month = pd.to_datetime(dates).dt.month
    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=dates.index,
    )


def cyclical_day_of_week(dates: pd.Series) -> pd.DataFrame:
    """Encode day of week as sine/cosine pair.

    Args:
        dates: Series of datetime values.

    Returns:
        DataFrame with columns ``dow_sin`` and ``dow_cos``.
    """
    dow = pd.to_datetime(dates).dt.dayofweek
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 5),  # 5 trading days
            "dow_cos": np.cos(2 * np.pi * dow / 5),
        },
        index=dates.index,
    )


def quarter_indicator(dates: pd.Series) -> pd.Series:
    """Extract quarter as an integer feature (1-4).

    Args:
        dates: Series of datetime values.

    Returns:
        Series of quarter integers.
    """
    quarter = pd.to_datetime(dates).dt.quarter
    quarter.name = "quarter"
    return quarter


def crop_season_phase(dates: pd.Series) -> pd.Series:
    """Map months to South American soybean crop phases.

    - Oct-Dec: Planting (Brazil)
    - Jan-Mar: Growing / Development
    - Apr-Jun: Harvest (Brazil) / US Planting
    - Jul-Sep: US Growing / Safrinha harvest

    Args:
        dates: Series of datetime values.

    Returns:
        Series with crop phase labels.
    """
    month = pd.to_datetime(dates).dt.month
    phase_map = {
        10: "planting", 11: "planting", 12: "planting",
        1: "growing", 2: "growing", 3: "growing",
        4: "harvest", 5: "harvest", 6: "harvest",
        7: "us_growing", 8: "us_growing", 9: "us_growing",
    }
    phases = month.map(phase_map)
    phases.name = "crop_phase"
    return phases


def compute_all_calendar(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add all calendar features to a DataFrame.

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with calendar feature columns appended.
    """
    result = df.copy()

    if date_col not in df.columns:
        logger.warning("date_column_missing", column=date_col)
        return result

    dates = df[date_col]

    month_enc = cyclical_month(dates)
    result = pd.concat([result, month_enc], axis=1)

    dow_enc = cyclical_day_of_week(dates)
    result = pd.concat([result, dow_enc], axis=1)

    result["quarter"] = quarter_indicator(dates)
    result["crop_phase"] = crop_season_phase(dates)

    logger.info("calendar_features_added", count=6)
    return result
