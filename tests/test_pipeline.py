"""Integration tests for the trained regression pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.config import CLEAN_DATA
from src.utils import load_model


@pytest.fixture(scope="module")
def model():
    """Load the trained pipeline once for the entire test module."""
    return load_model()


@pytest.fixture(scope="module")
def feature_names(model) -> list[str]:
    """Extract feature names from the pipeline preprocessor."""
    pipe = getattr(model, "regressor", model)
    preproc = pipe.named_steps.get("preprocessor")
    if preproc is not None and hasattr(preproc, "get_feature_names_out"):
        import re
        return [
            re.sub(r".*__", "", n)
            for n in preproc.get_feature_names_out()
        ]
    df = pd.read_parquet(CLEAN_DATA)
    return [c for c in df.columns if c != "boc1"]


@pytest.fixture(scope="module")
def sample_input(feature_names) -> pd.DataFrame:
    """Build a single-row DataFrame with mean values from the training set."""
    df = pd.read_parquet(CLEAN_DATA)
    means = df[[c for c in feature_names if c in df.columns]].mean()
    return pd.DataFrame([means.to_dict()])


def test_model_prediction_returns_float(model, sample_input):
    """model.predict() on a valid input must return a scalar float."""
    pred = model.predict(sample_input)
    value = float(np.ravel(pred)[0])
    assert isinstance(value, float)
    assert not np.isnan(value)


def test_model_prediction_shape(model, sample_input):
    """model.predict() on one sample must return an array of shape (1,)."""
    pred = model.predict(sample_input)
    assert np.ravel(pred).shape == (1,)


def test_model_prediction_in_reasonable_range(model, sample_input):
    """Predicted BOC1 price must fall within the plausible historical range."""
    pred = float(np.ravel(model.predict(sample_input))[0])
    assert 10 < pred < 5000, f"Prediction {pred} is outside the expected range."
