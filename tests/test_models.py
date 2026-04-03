"""Unit tests for model evaluation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import (
    compare_models,
    directional_accuracy,
    evaluate_regression,
)
from src.models.pipeline import (
    build_regression_model_pipeline,
    organize_cv_results,
)


class TestEvaluation:
    def test_evaluate_regression_keys(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        metrics = evaluate_regression(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert "directional_accuracy" in metrics

    def test_evaluate_regression_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = evaluate_regression(y_true, y_pred)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_directional_accuracy_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert directional_accuracy(y_true, y_pred) == 1.0

    def test_directional_accuracy_opposite(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        assert directional_accuracy(y_true, y_pred) == 0.0

    def test_directional_accuracy_single_sample(self):
        assert directional_accuracy(np.array([1.0]), np.array([1.0])) == 0.0

    def test_compare_models_sorted_by_rmse(self):
        results = {
            "model_a": {"metrics": {"mae": 2.0, "rmse": 3.0, "r2": 0.8, "mape": 0.1, "directional_accuracy": 0.6}},
            "model_b": {"metrics": {"mae": 1.0, "rmse": 1.5, "r2": 0.9, "mape": 0.05, "directional_accuracy": 0.7}},
        }
        df = compare_models(results)
        assert df.index[0] == "model_b"  # better RMSE first


class TestPipeline:
    def test_build_pipeline_without_preprocessor(self):
        from sklearn.linear_model import LinearRegression

        pipeline = build_regression_model_pipeline(LinearRegression())
        assert "reg" in pipeline.named_steps

    def test_build_pipeline_with_preprocessor(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        pipeline = build_regression_model_pipeline(
            LinearRegression(), preprocessor=StandardScaler()
        )
        assert "preprocessor" in pipeline.named_steps
        assert "reg" in pipeline.named_steps

    def test_organize_cv_results_shape(self):
        results = {
            "model_a": {
                "fit_time": np.array([0.1, 0.2]),
                "score_time": np.array([0.01, 0.02]),
                "test_r2": np.array([0.9, 0.85]),
                "test_neg_mean_absolute_error": np.array([-1.0, -1.2]),
                "test_neg_root_mean_squared_error": np.array([-1.5, -1.8]),
            },
        }
        df = organize_cv_results(results)
        assert len(df) == 2  # 2 folds
        assert "model" in df.columns
