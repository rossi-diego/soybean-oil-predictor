"""Technical / statistical features for time-series modeling.

These features capture momentum, volatility regime, and mean-reversion
signals that complement the fundamental spread features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)


def rolling_volatility(
    series: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute annualized rolling volatility at multiple windows.

    Args:
        series: Price series.
        windows: Rolling window sizes in trading days.
            Defaults to [5, 21, 63] (1 week, 1 month, 1 quarter).

    Returns:
        DataFrame with one column per window (e.g. ``vol_5d``).
    """
    windows = windows or [5, 21, 63]
    returns = series.pct_change()
    result = pd.DataFrame(index=series.index)

    for w in windows:
        col_name = f"vol_{w}d"
        result[col_name] = returns.rolling(w).std() * np.sqrt(252)
        logger.debug("rolling_vol_computed", window=w)

    return result


def lagged_returns(
    series: pd.Series,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Compute lagged percentage returns.

    Args:
        series: Price series.
        lags: Lag periods in trading days.
            Defaults to [1, 5, 21] (1 day, 1 week, 1 month).

    Returns:
        DataFrame with one column per lag (e.g. ``ret_1d``).
    """
    lags = lags or [1, 5, 21]
    result = pd.DataFrame(index=series.index)

    for lag in lags:
        col_name = f"ret_{lag}d"
        result[col_name] = series.pct_change(lag)

    return result


def rolling_zscore(
    series: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Compute rolling z-score for mean-reversion signals.

    Z-score = (current - rolling_mean) / rolling_std
    Values > 2 or < -2 indicate extreme deviations from the local mean.

    Args:
        series: Price series.
        window: Rolling window in trading days (default 63 = 1 quarter).

    Returns:
        Rolling z-score series.
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    zscore = (series - rolling_mean) / rolling_std.replace(0, float("nan"))
    zscore.name = f"zscore_{window}d"
    return zscore


def momentum(
    series: pd.Series,
    fast: int = 21,
    slow: int = 63,
) -> pd.Series:
    """Compute simple moving average crossover momentum signal.

    Positive values indicate the fast MA is above the slow MA (bullish).

    Args:
        series: Price series.
        fast: Fast moving average period.
        slow: Slow moving average period.

    Returns:
        Momentum signal (fast_ma - slow_ma) / slow_ma.
    """
    fast_ma = series.rolling(fast).mean()
    slow_ma = series.rolling(slow).mean()
    mom = (fast_ma - slow_ma) / slow_ma.replace(0, float("nan"))
    mom.name = f"momentum_{fast}_{slow}"
    return mom


def compute_all_technical(
    df: pd.DataFrame,
    target_col: str = "boc1",
) -> pd.DataFrame:
    """Add all technical features for the target column.

    Args:
        df: DataFrame containing the target price column.
        target_col: Name of the target price column.

    Returns:
        DataFrame with technical feature columns appended.
    """
    result = df.copy()

    if target_col not in df.columns:
        logger.warning("target_column_missing", column=target_col)
        return result

    price = df[target_col]

    vol_df = rolling_volatility(price)
    result = pd.concat([result, vol_df], axis=1)

    ret_df = lagged_returns(price)
    result = pd.concat([result, ret_df], axis=1)

    result[f"{target_col}_zscore"] = rolling_zscore(price)
    result[f"{target_col}_momentum"] = momentum(price)

    logger.info("technical_features_added", count=vol_df.shape[1] + ret_df.shape[1] + 2)
    return result
