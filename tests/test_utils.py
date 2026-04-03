"""Unit tests for src.utils.coefficients_dataframe."""

import numpy as np
import pandas as pd
import pytest

from src.utils import coefficients_dataframe


def test_coefficients_dataframe_returns_dataframe():
    """Result must be a DataFrame with a single column named 'coefficient'."""
    coefs = np.array([0.5, -1.2, 0.3])
    columns = ["a", "b", "c"]
    result = coefficients_dataframe(coefs, columns)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["coefficient"]


def test_coefficients_dataframe_index_matches_columns():
    """The DataFrame index must contain the supplied feature names."""
    coefs = np.array([1.0, 2.0])
    columns = ["feat_x", "feat_y"]
    result = coefficients_dataframe(coefs, columns)
    assert set(result.index) == set(columns)


def test_coefficients_dataframe_sorted_ascending():
    """Values must be sorted in ascending order by coefficient."""
    coefs = np.array([3.0, -1.0, 0.0])
    columns = ["c", "a", "b"]
    result = coefficients_dataframe(coefs, columns)
    values = result["coefficient"].tolist()
    assert values == sorted(values)


def test_coefficients_dataframe_empty_inputs():
    """Empty inputs must produce an empty DataFrame with 'coefficient' column."""
    result = coefficients_dataframe([], [])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["coefficient"]
    assert len(result) == 0


def test_coefficients_dataframe_length_mismatch_raises():
    """Mismatched lengths must raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        coefficients_dataframe(np.array([1.0, 2.0]), ["only_one"])
