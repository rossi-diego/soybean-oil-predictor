"""Train models and generate walk-forward backtest results.

Usage:
    python scripts/train_model.py

Trains XGBoost and Ridge models from commodities_clean_data.parquet,
runs walk-forward validation, and saves models + backtest results.
Designed to run during Docker build or locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "commodities_clean_data.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
BACKTEST_FILE = PROJECT_ROOT / "data" / "walk_forward_backtest.parquet"

TARGET = "boc1"
RANDOM_STATE = 42


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of correct up/down predictions."""
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float((true_dir == pred_dir).sum() / len(true_dir))


def train_and_save():
    """Train models, run walk-forward backtest, save everything."""
    if not DATA_FILE.exists():
        print(f"ERROR: Training data not found at {DATA_FILE}")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE
    )

    print(f"Training data: {len(X_train)} rows, {len(feature_cols)} features")
    print(f"Test data: {len(X_test)} rows")

    # --- XGBoost ---
    xgb_params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = xgb.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    r2 = float(r2_score(y_test, y_pred))
    print(f"XGBoost — RMSE: {rmse:.4f}, R2: {r2:.4f}")

    xgb_path = MODEL_DIR / "xgboost_baseline.joblib"
    joblib.dump(xgb, xgb_path)
    print(f"Saved: {xgb_path}")

    # --- Ridge ---
    ridge_pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    ridge_pipe.fit(X_train, y_train)

    y_pred_r = ridge_pipe.predict(X_test)
    rmse_r = float(np.sqrt(np.mean((y_test - y_pred_r) ** 2)))
    r2_r = float(r2_score(y_test, y_pred_r))
    print(f"Ridge   — RMSE: {rmse_r:.4f}, R2: {r2_r:.4f}")

    ridge_path = MODEL_DIR / "ridge_regression.joblib"
    joblib.dump(ridge_pipe, ridge_path)
    print(f"Saved: {ridge_path}")

    # --- Walk-forward backtest ---
    print("\nRunning walk-forward backtest (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    backtest_rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        preds = model.predict(X_te)
        residuals = y_te.values - preds
        residual_std = float(np.std(residuals))

        for i, idx in enumerate(test_idx):
            backtest_rows.append({
                "row_index": int(idx),
                "fold": fold,
                "actual": float(y_te.iloc[i]),
                "predicted": float(preds[i]),
                "residual": float(residuals[i]),
                "lower": float(preds[i] - 1.96 * residual_std),
                "upper": float(preds[i] + 1.96 * residual_std),
            })

        fold_mae = float(mean_absolute_error(y_te, preds))
        fold_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        fold_r2 = float(r2_score(y_te, preds))
        fold_da = directional_accuracy(y_te.values, preds)
        print(
            f"  Fold {fold}: MAE={fold_mae:.2f}, RMSE={fold_rmse:.2f}, "
            f"R2={fold_r2:.4f}, DirAcc={fold_da:.1%} ({len(test_idx)} samples)"
        )

    backtest_df = pd.DataFrame(backtest_rows)
    backtest_df.to_parquet(BACKTEST_FILE, index=False)
    print(f"\nBacktest saved: {BACKTEST_FILE} ({len(backtest_df)} predictions)")

    # Overall backtest metrics
    all_actual = backtest_df["actual"].values
    all_pred = backtest_df["predicted"].values
    print(f"Overall — MAE: {mean_absolute_error(all_actual, all_pred):.2f}, "
          f"RMSE: {np.sqrt(np.mean((all_actual - all_pred) ** 2)):.2f}, "
          f"R2: {r2_score(all_actual, all_pred):.4f}, "
          f"DirAcc: {directional_accuracy(all_actual, all_pred):.1%}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train_and_save()
