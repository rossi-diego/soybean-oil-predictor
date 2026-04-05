"""Stationary feature engineering for time-series modeling.

Raw commodity prices are non-stationary (unit root). This module transforms
them into stationary, mean-reverting features suitable for ML models:
- Percentage returns (differencing)
- Spread z-scores (mean-reverting by construction)
- Momentum signals (relative, not absolute)
- Volatility regime indicators
- Bollinger Band position (-1 to +1 scale)

These features are more robust to regime changes than raw price levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(
    df: pd.DataFrame,
    columns: list[str],
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """Compute percentage returns for given columns at multiple horizons.

    Args:
        df: DataFrame with price columns.
        columns: Column names to compute returns for.
        periods: Return horizons in trading days. Defaults to [1, 5, 20].

    Returns:
        DataFrame with return columns appended.
    """
    periods = periods or [1, 5, 20]
    result = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for p in periods:
            result[f"{col}_ret{p}d"] = df[col].pct_change(p)
    return result


def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trading-relevant spread and ratio z-scores.

    All features are mean-reverting (stationary) by construction:
    z-score = (current - 60d_mean) / 60d_std

    Args:
        df: DataFrame with commodity price columns.

    Returns:
        DataFrame with spread z-score columns appended.
    """
    result = df.copy()
    window = 60

    # Crush spread z-score: (11 * oil + meal) - beans
    if all(c in df.columns for c in ["boc1", "smc1", "sc1"]):
        crush = 11 * df["boc1"] + df["smc1"] - df["sc1"]
        ma = crush.rolling(window).mean()
        std = crush.rolling(window).std().replace(0, np.nan)
        result["crush_zscore"] = (crush - ma) / std

    # BOC1 / Crude ratio z-score (biodiesel economics)
    if all(c in df.columns for c in ["boc1", "lcoc1"]):
        ratio = df["boc1"] / df["lcoc1"].replace(0, np.nan)
        ma = ratio.rolling(window).mean()
        std = ratio.rolling(window).std().replace(0, np.nan)
        result["boc1_crude_zscore"] = (ratio - ma) / std

    # BOC1 / Heating oil ratio z-score
    if all(c in df.columns for c in ["boc1", "hoc1"]):
        ratio = df["boc1"] / df["hoc1"].replace(0, np.nan)
        ma = ratio.rolling(window).mean()
        std = ratio.rolling(window).std().replace(0, np.nan)
        result["boc1_ho_zscore"] = (ratio - ma) / std

    return result


def compute_momentum_features(df: pd.DataFrame, target: str = "boc1") -> pd.DataFrame:
    """Compute momentum and mean-reversion signals for the target.

    Args:
        df: DataFrame with target price column.
        target: Target column name.

    Returns:
        DataFrame with momentum features appended.
    """
    result = df.copy()
    price = df[target]

    # Momentum: 20-day return
    result[f"{target}_mom20d"] = price.pct_change(20)

    # Mean reversion: distance from 60-day MA (as fraction)
    ma60 = price.rolling(60).mean()
    result[f"{target}_mean_rev"] = (price - ma60) / ma60.replace(0, np.nan)

    # Rolling volatility: 20-day annualized
    result[f"{target}_vol20d"] = price.pct_change().rolling(20).std() * np.sqrt(252)

    # Bollinger Band position: -1 (lower band) to +1 (upper band)
    ma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std().replace(0, np.nan)
    result[f"{target}_bb_pos"] = (price - ma20) / (2 * std20)

    return result


def compute_lagged_target_returns(
    df: pd.DataFrame,
    target: str = "boc1",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged returns of the target variable (autoregressive features).

    Uses LAGGED returns (shifted by 1+) to avoid look-ahead bias.
    lag1 = yesterday's return, lag2 = 2 days ago, etc.

    Args:
        df: DataFrame with target column.
        target: Target column name.
        lags: Lag periods. Defaults to [1, 2, 3, 5].

    Returns:
        DataFrame with lagged return columns appended.
    """
    lags = lags or [1, 2, 3, 5]
    result = df.copy()
    daily_return = df[target].pct_change()

    for lag in lags:
        result[f"{target}_retlag{lag}"] = daily_return.shift(lag)

    return result


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical month encoding.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with month_sin and month_cos columns.
    """
    result = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        month = df.index.month
    else:
        return result

    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)
    return result


def build_stationary_features(
    df: pd.DataFrame,
    target: str = "boc1",
    price_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full stationary feature matrix and target.

    Combines all feature engineering steps and drops NaN rows
    from rolling windows.

    Args:
        df: Raw price DataFrame with DatetimeIndex.
        target: Target column name.
        price_cols: Columns to compute returns for.

    Returns:
        Tuple of (feature_matrix X, target_returns y).
    """
    price_cols = price_cols or ["smc1", "sc1", "lcoc1", "hoc1", "rsc1"]

    # Target: next-day return (what we're predicting)
    y = df[target].pct_change().shift(-1)  # shift -1 = predict tomorrow's return
    y.name = "target_return"

    # Features (all use t and earlier data — no look-ahead)
    featured = df.copy()
    featured = compute_returns(featured, price_cols, periods=[1, 5, 20])
    featured = compute_spread_features(featured)
    featured = compute_momentum_features(featured, target)
    featured = compute_lagged_target_returns(featured, target)
    featured = compute_calendar_features(featured)

    # Select only engineered features (not raw prices)
    feature_cols = [c for c in featured.columns if c not in df.columns]
    X = featured[feature_cols]

    # Drop NaN rows (first ~60 from rolling windows, last 1 from target shift)
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return X, y
