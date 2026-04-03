"""Train models from existing parquet data.

Usage:
    python scripts/train_model.py

Trains XGBoost and Ridge models from commodities_clean_data.parquet,
saves them to models/. Designed to run during Docker build or at startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "commodities_clean_data.parquet"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET = "boc1"
RANDOM_STATE = 42


def train_and_save():
    """Train XGBoost and Ridge models, save to models/ directory."""
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
    xgb = XGBRegressor(
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
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = xgb.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    r2 = float(1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
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
    r2_r = float(1 - np.sum((y_test - y_pred_r) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    print(f"Ridge   — RMSE: {rmse_r:.4f}, R2: {r2_r:.4f}")

    ridge_path = MODEL_DIR / "ridge_regression.joblib"
    joblib.dump(ridge_pipe, ridge_path)
    print(f"Saved: {ridge_path}")

    print("Training complete.")


if __name__ == "__main__":
    train_and_save()
