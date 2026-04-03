"""Unit tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from src.features.spreads import (
    compute_all_spreads,
    crush_spread,
    oil_palm_spread,
    oil_share,
    soy_corn_ratio,
)
from src.features.technical import (
    lagged_returns,
    momentum,
    rolling_volatility,
    rolling_zscore,
)
from src.features.calendar import (
    compute_all_calendar,
    crop_season_phase,
    cyclical_month,
)


@pytest.fixture
def sample_prices():
    """Sample commodity price DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-01", periods=n),
        "boc1": np.random.uniform(30, 60, n),
        "sc1": np.random.uniform(900, 1500, n),
        "smc1": np.random.uniform(280, 450, n),
        "zc1": np.random.uniform(400, 700, n),
        "fcpoc1": np.random.uniform(2000, 5000, n),
        "lcoc1": np.random.uniform(50, 100, n),
        "hoc1": np.random.uniform(1.5, 3.5, n),
        "rsc1": np.random.uniform(400, 800, n),
    })


class TestSpreads:
    def test_crush_spread_shape(self, sample_prices):
        result = crush_spread(sample_prices["sc1"], sample_prices["boc1"], sample_prices["smc1"])
        assert len(result) == len(sample_prices)
        assert result.name == "crush_spread"

    def test_oil_palm_spread(self, sample_prices):
        result = oil_palm_spread(sample_prices["boc1"], sample_prices["fcpoc1"])
        assert len(result) == len(sample_prices)
        assert result.name == "oil_palm_spread"

    def test_soy_corn_ratio_positive(self, sample_prices):
        result = soy_corn_ratio(sample_prices["sc1"], sample_prices["zc1"])
        assert (result.dropna() > 0).all()

    def test_soy_corn_ratio_handles_zero_corn(self):
        soybeans = pd.Series([1000, 1100, 1200])
        corn = pd.Series([500, 0, 600])
        result = soy_corn_ratio(soybeans, corn)
        assert pd.isna(result.iloc[1])

    def test_oil_share_range(self, sample_prices):
        result = oil_share(sample_prices["boc1"], sample_prices["sc1"])
        valid = result.dropna()
        assert (valid > 0).all()

    def test_compute_all_spreads_adds_columns(self, sample_prices):
        result = compute_all_spreads(sample_prices)
        assert "crush_spread" in result.columns
        assert "oil_palm_spread" in result.columns
        assert "soy_corn_ratio" in result.columns
        assert "oil_share" in result.columns


class TestTechnical:
    def test_rolling_volatility_shape(self, sample_prices):
        result = rolling_volatility(sample_prices["boc1"])
        assert "vol_5d" in result.columns
        assert "vol_21d" in result.columns
        assert "vol_63d" in result.columns
        assert len(result) == len(sample_prices)

    def test_lagged_returns_shape(self, sample_prices):
        result = lagged_returns(sample_prices["boc1"])
        assert "ret_1d" in result.columns
        assert "ret_5d" in result.columns
        assert "ret_21d" in result.columns

    def test_rolling_zscore_centered(self, sample_prices):
        result = rolling_zscore(sample_prices["boc1"], window=20)
        valid = result.dropna()
        assert abs(valid.mean()) < 1.0  # z-scores should be roughly centered

    def test_momentum_sign(self):
        prices = pd.Series(range(1, 101), dtype=float)
        result = momentum(prices, fast=5, slow=20)
        valid = result.dropna()
        assert (valid > 0).all()  # upward trending = positive momentum


class TestCalendar:
    def test_cyclical_month_range(self, sample_prices):
        result = cyclical_month(sample_prices["date"])
        assert result["month_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()

    def test_cyclical_month_december_close_to_january(self):
        dates = pd.Series([pd.Timestamp("2023-12-15"), pd.Timestamp("2024-01-15")])
        result = cyclical_month(dates)
        sin_diff = abs(result["month_sin"].iloc[0] - result["month_sin"].iloc[1])
        cos_diff = abs(result["month_cos"].iloc[0] - result["month_cos"].iloc[1])
        assert sin_diff < 1.0  # December and January should be close in cyclical space

    def test_crop_season_phase_mapping(self):
        dates = pd.Series([
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-04-15"),
            pd.Timestamp("2023-07-15"),
            pd.Timestamp("2023-10-15"),
        ])
        result = crop_season_phase(dates)
        assert result.iloc[0] == "growing"
        assert result.iloc[1] == "harvest"
        assert result.iloc[2] == "us_growing"
        assert result.iloc[3] == "planting"

    def test_compute_all_calendar_adds_columns(self, sample_prices):
        result = compute_all_calendar(sample_prices)
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns
        assert "quarter" in result.columns
        assert "crop_phase" in result.columns
