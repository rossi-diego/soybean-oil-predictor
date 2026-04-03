"""Commodity spread features for soybean oil forecasting.

Spreads capture relative value and substitution dynamics that are
fundamental to commodity price discovery. These are the features
that differentiate a domain-expert model from a generic ML model.
"""

from __future__ import annotations

import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)


def crush_spread(
    soybeans: pd.Series,
    soybean_oil: pd.Series,
    soybean_meal: pd.Series,
) -> pd.Series:
    """Calculate the soybean crush spread (gross processing margin).

    The crush spread measures the profit from crushing soybeans into
    oil and meal. It is the primary profitability indicator for soybean
    processors (e.g., Oleoplan, Bunge, Cargill).

    Formula: (oil_value + meal_value) - soybean_cost
    Simplified: 11 * oil_price + meal_price - soybean_price

    (11 lbs of oil per bushel of soybeans is the standard yield assumption)

    Args:
        soybeans: Soybean futures price (cents/bushel).
        soybean_oil: Soybean oil futures price (cents/lb).
        soybean_meal: Soybean meal futures price ($/short ton).

    Returns:
        Crush spread series.
    """
    spread = (11 * soybean_oil) + soybean_meal - soybeans
    spread.name = "crush_spread"
    return spread


def oil_palm_spread(
    soybean_oil: pd.Series,
    palm_oil: pd.Series,
) -> pd.Series:
    """Calculate BOC1 vs CPO (crude palm oil) spread.

    This spread captures substitution dynamics between soybean oil and
    palm oil — the two most traded vegetable oils globally. When the
    spread widens, it signals soy oil is expensive relative to palm,
    incentivizing buyers to switch to palm oil.

    Args:
        soybean_oil: Soybean oil futures price.
        palm_oil: Crude palm oil futures price.

    Returns:
        Oil-palm spread series (same units as soybean_oil).
    """
    spread = soybean_oil - palm_oil
    spread.name = "oil_palm_spread"
    return spread


def soy_corn_ratio(
    soybeans: pd.Series,
    corn: pd.Series,
) -> pd.Series:
    """Calculate the soybean/corn price ratio.

    This ratio influences farmer planting decisions in the US and Brazil.
    A high ratio favors planting more soybeans (bearish for soy long-term),
    while a low ratio favors corn (bullish for soy).

    Typical range: 2.0 – 3.0

    Args:
        soybeans: Soybean futures price.
        corn: Corn futures price.

    Returns:
        Soy/corn ratio series.
    """
    ratio = soybeans / corn.replace(0, float("nan"))
    ratio.name = "soy_corn_ratio"
    return ratio


def oil_share(
    soybean_oil: pd.Series,
    soybeans: pd.Series,
) -> pd.Series:
    """Calculate soybean oil share of total soybean value.

    Oil share = (oil_price * 11) / soybean_price
    Measures how much of the soybean's value comes from oil vs meal.

    Args:
        soybean_oil: Soybean oil futures price (cents/lb).
        soybeans: Soybean futures price (cents/bushel).

    Returns:
        Oil share ratio series (typically 0.30 – 0.50).
    """
    share = (soybean_oil * 11) / soybeans.replace(0, float("nan"))
    share.name = "oil_share"
    return share


def compute_all_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Add all spread features to a DataFrame.

    Expects columns: boc1 (soybean oil), sc1 (soybeans), smc1 (meal),
    zc1 (corn), fcpoc1 (palm oil).

    Args:
        df: DataFrame with commodity price columns.

    Returns:
        DataFrame with spread columns appended.
    """
    result = df.copy()

    if all(c in df.columns for c in ["sc1", "boc1", "smc1"]):
        result["crush_spread"] = crush_spread(df["sc1"], df["boc1"], df["smc1"])
        logger.info("feature_added", feature="crush_spread")

    if all(c in df.columns for c in ["boc1", "fcpoc1"]):
        result["oil_palm_spread"] = oil_palm_spread(df["boc1"], df["fcpoc1"])
        logger.info("feature_added", feature="oil_palm_spread")

    if all(c in df.columns for c in ["sc1", "zc1"]):
        result["soy_corn_ratio"] = soy_corn_ratio(df["sc1"], df["zc1"])
        logger.info("feature_added", feature="soy_corn_ratio")

    if all(c in df.columns for c in ["boc1", "sc1"]):
        result["oil_share"] = oil_share(df["boc1"], df["sc1"])
        logger.info("feature_added", feature="oil_share")

    return result
