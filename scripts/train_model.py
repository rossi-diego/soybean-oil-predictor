"""Train models, run walk-forward backtest, and generate all diagnostics.

Usage:
    python scripts/train_model.py

Trains 5 models, runs walk-forward validation, computes learning curves,
feature importance, and residual diagnostics. All results saved as JSON
files served by the API at zero latency.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "commodities_clean_data.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

TARGET = "boc1"
RANDOM_STATE = 42

XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)

MODELS = {
    "xgboost": lambda: XGBRegressor(**XGB_PARAMS),
    "ridge": lambda: Pipeline([("scaler", RobustScaler()), ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
    "lasso": lambda: Pipeline([("scaler", RobustScaler()), ("reg", Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000))]),
    "elasticnet": lambda: Pipeline([("scaler", RobustScaler()), ("reg", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000))]),
    "linear": lambda: Pipeline([("scaler", RobustScaler()), ("reg", LinearRegression())]),
}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    return float((np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).mean())


def train_and_save():
    if not DATA_FILE.exists():
        print(f"ERROR: Data not found at {DATA_FILE}")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)

    # Use level prices as primary features (captures cross-sectional relationships)
    # plus stationary technical indicators for regime awareness.
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.features.stationary import (
        compute_momentum_features,
        compute_spread_features,
    )

    LIVE_FEATURES = ["smc1", "sc1", "lcoc1", "hoc1", "rsc1"]

    # Add stationary features alongside levels
    enhanced = df.copy()
    enhanced = compute_spread_features(enhanced)
    enhanced = compute_momentum_features(enhanced, TARGET)

    # Combine: raw prices + spread z-scores + momentum
    stationary_cols = [c for c in enhanced.columns if c not in df.columns]
    feature_cols = [c for c in LIVE_FEATURES if c in df.columns] + stationary_cols

    # Drop NaN rows from rolling windows
    X = enhanced[feature_cols].dropna()
    y = df[TARGET].loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE
    )

    print(f"Data: {len(df)} rows, {len(feature_cols)} features")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # =========================================================================
    # 1. Train all models + compute test metrics
    # =========================================================================
    comparison = []
    trained_models = {}

    for name, factory in MODELS.items():
        model = factory()
        t0 = time.time()

        if name == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)

        train_time = round(time.time() - t0, 3)
        y_pred = model.predict(X_test)

        metrics = {
            "model": name,
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
            "rmse": round(float(np.sqrt(np.mean((y_test - y_pred) ** 2))), 2),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "directional_accuracy": round(directional_accuracy(y_test.values, y_pred), 4),
            "train_time_s": train_time,
        }
        comparison.append(metrics)
        trained_models[name] = model

        print(f"{name:12} MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
              f"R2={metrics['r2']:.4f}  DirAcc={metrics['directional_accuracy']:.1%}  "
              f"Time={train_time:.3f}s")

        # Save model
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    # Also save the XGBoost as xgboost_baseline.joblib for backward compat
    joblib.dump(trained_models["xgboost"], MODEL_DIR / "xgboost_baseline.joblib")

    comparison.sort(key=lambda x: x["mae"])
    champion = comparison[0]["model"]
    print(f"\nChampion: {champion} (lowest MAE)")

    with open(DATA_DIR / "model_comparison.json", "w") as f:
        json.dump({"models": comparison, "champion": champion}, f, indent=2)

    # =========================================================================
    # 2. Walk-forward backtest (champion only for residuals, all for fold metrics)
    # =========================================================================
    print("\nWalk-forward backtest (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    backtest_rows = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        fold_results = {"fold": fold, "n_train": len(train_idx), "n_test": len(test_idx)}

        for name, factory in MODELS.items():
            m = factory()
            if name == "xgboost":
                m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            else:
                m.fit(X_tr, y_tr)

            preds = m.predict(X_te)
            residuals = y_te.values - preds
            fold_results[f"{name}_mae"] = round(float(mean_absolute_error(y_te, preds)), 2)
            fold_results[f"{name}_rmse"] = round(float(np.sqrt(np.mean(residuals ** 2))), 2)
            fold_results[f"{name}_r2"] = round(float(r2_score(y_te, preds)), 4)
            fold_results[f"{name}_da"] = round(directional_accuracy(y_te.values, preds), 4)

            # Store residuals for champion
            if name == champion:
                residual_std = float(np.std(residuals))
                for i, idx in enumerate(test_idx):
                    date_val = df.index[idx]
                    date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)[:10]
                    backtest_rows.append({
                        "row_index": int(idx),
                        "date": date_str,
                        "fold": fold,
                        "actual": float(y_te.iloc[i]),
                        "predicted": float(preds[i]),
                        "residual": float(residuals[i]),
                        "lower": float(preds[i] - 1.96 * residual_std),
                        "upper": float(preds[i] + 1.96 * residual_std),
                    })

        fold_metrics.append(fold_results)
        print(f"  Fold {fold}: {fold_results[f'{champion}_mae']:.2f} MAE ({len(test_idx)} samples)")

    # Save backtest parquet
    backtest_df = pd.DataFrame(backtest_rows)
    backtest_df.to_parquet(DATA_DIR / "walk_forward_backtest.parquet", index=False)

    # Save fold metrics
    with open(DATA_DIR / "walk_forward_folds.json", "w") as f:
        json.dump({"folds": fold_metrics, "champion": champion, "models": list(MODELS.keys())}, f, indent=2)

    # =========================================================================
    # 3. Champion diagnostics (residuals for scatter/histogram)
    # =========================================================================
    champ_model = trained_models[champion]
    y_pred_full = champ_model.predict(X_test)
    residuals_full = y_test.values - y_pred_full

    diagnostics = {
        "model": champion,
        "points": [
            {
                "actual": round(float(y_test.iloc[i]), 2),
                "predicted": round(float(y_pred_full[i]), 2),
                "residual": round(float(residuals_full[i]), 2),
                "index": int(i),
            }
            for i in range(len(y_test))
        ],
        "residual_stats": {
            "mean": round(float(np.mean(residuals_full)), 4),
            "std": round(float(np.std(residuals_full)), 4),
            "skew": round(float(pd.Series(residuals_full).skew()), 4),
            "kurtosis": round(float(pd.Series(residuals_full).kurtosis()), 4),
        },
        "residual_histogram": [],
    }

    counts, edges = np.histogram(residuals_full, bins=30)
    for i in range(len(counts)):
        diagnostics["residual_histogram"].append({
            "x": round(float(edges[i]), 2),
            "count": int(counts[i]),
        })

    with open(DATA_DIR / "champion_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    # =========================================================================
    # 4. Feature importance
    # =========================================================================
    importance_data = {"models": {}}

    # XGBoost native importance
    xgb = trained_models["xgboost"]
    imp = sorted(
        [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(feature_cols, xgb.feature_importances_)],
        key=lambda x: x["importance"], reverse=True
    )
    importance_data["models"]["xgboost"] = imp

    # Linear model coefficients
    for name in ["ridge", "lasso", "elasticnet", "linear"]:
        m = trained_models[name]
        coefs = m.named_steps["reg"].coef_
        imp = sorted(
            [{"feature": f, "importance": round(float(abs(c)), 4), "coefficient": round(float(c), 4)}
             for f, c in zip(feature_cols, coefs)],
            key=lambda x: x["importance"], reverse=True
        )
        importance_data["models"][name] = imp

    with open(DATA_DIR / "feature_importance.json", "w") as f:
        json.dump(importance_data, f, indent=2)

    # =========================================================================
    # 5. Learning curves
    # =========================================================================
    print("\nComputing learning curves...")
    lc_data = {"models": {}}

    for name in ["xgboost", "ridge", "lasso"]:
        m = MODELS[name]()
        try:
            sizes, train_scores, val_scores = learning_curve(
                m, X, y, cv=3,
                train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            lc_data["models"][name] = {
                "train_sizes": [int(s) for s in sizes],
                "train_mae": [round(float(-s.mean()), 2) for s in train_scores.T] if train_scores.ndim > 1 else [],
                "val_mae": [round(float(-s.mean()), 2) for s in val_scores.T] if val_scores.ndim > 1 else [],
            }
            # Fix: learning_curve returns (n_sizes, n_folds) — need to mean across folds
            lc_data["models"][name]["train_mae"] = [round(float(-train_scores[i].mean()), 2) for i in range(len(sizes))]
            lc_data["models"][name]["val_mae"] = [round(float(-val_scores[i].mean()), 2) for i in range(len(sizes))]
            print(f"  {name}: {len(sizes)} points")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    with open(DATA_DIR / "learning_curves.json", "w") as f:
        json.dump(lc_data, f, indent=2)

    # =========================================================================
    # 6. Model metadata (updated)
    # =========================================================================
    overall_residual_std = float(np.std(backtest_df["residual"].values))
    overall_metrics = {
        "mae": round(float(mean_absolute_error(backtest_df["actual"], backtest_df["predicted"])), 2),
        "rmse": round(float(np.sqrt(np.mean(backtest_df["residual"].values ** 2))), 2),
        "r2": round(float(r2_score(backtest_df["actual"], backtest_df["predicted"])), 4),
        "directional_accuracy": round(directional_accuracy(backtest_df["actual"].values, backtest_df["predicted"].values), 4),
    }

    metadata = {
        "algorithm": "XGBRegressor",
        "champion": champion,
        "hyperparameters": XGB_PARAMS,
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_total_samples": len(df),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "target": TARGET,
        "residual_std": round(overall_residual_std, 4),
        "backtest_metrics": {**overall_metrics, "n_predictions": len(backtest_df), "n_folds": 5},
        "trained_at": datetime.now().isoformat(),
        "train_start": df.index[0].strftime("%Y-%m-%d"),
        "train_end": df.index[len(X_train) - 1].strftime("%Y-%m-%d"),
        "test_start": df.index[len(X_train)].strftime("%Y-%m-%d"),
        "test_end": df.index[-1].strftime("%Y-%m-%d"),
        "feature_ranges": {
            col: {
                "min": round(float(X[col].min()), 2),
                "max": round(float(X[col].max()), 2),
                "mean": round(float(X[col].mean()), 2),
                "p5": round(float(X[col].quantile(0.05)), 2),
                "p95": round(float(X[col].quantile(0.95)), 2),
            }
            for col in feature_cols
        },
    }

    with open(DATA_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nAll files saved to {DATA_DIR}/")
    print("Training complete.")


if __name__ == "__main__":
    train_and_save()
