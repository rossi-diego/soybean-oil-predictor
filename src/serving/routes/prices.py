"""Live commodity prices from Yahoo Finance."""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import DATA_FOLDER, FEATURE_COLUMNS, TICKERS
from src.log import get_logger

logger = get_logger(__name__)

router = APIRouter()

# In-memory cache: {ticker: {price, timestamp}}
_price_cache: dict = {}
_CACHE_TTL = 900  # 15 minutes


class PriceData(BaseModel):
    """Single commodity price."""

    name: str
    ticker: str
    price: float
    currency: str
    timestamp: str


class LivePricesResponse(BaseModel):
    """Response with all live commodity prices."""

    prices: list[PriceData]
    cached: bool
    fetched_at: str


def _fetch_live_prices() -> dict[str, float]:
    """Fetch current prices from Yahoo Finance with caching."""
    now = time.time()

    if _price_cache and (now - _price_cache.get("_ts", 0)) < _CACHE_TTL:
        logger.info("prices_from_cache")
        return _price_cache

    logger.info("fetching_live_prices")
    prices = {"_ts": now, "_cached": False}

    for name, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                prices[name] = float(hist["Close"].iloc[-1])
            else:
                logger.warning("empty_price_data", ticker=symbol)
        except Exception:
            logger.exception("price_fetch_failed", ticker=symbol)

    _price_cache.clear()
    _price_cache.update(prices)
    _price_cache["_cached"] = True

    return prices


@router.get("/prices/latest", response_model=LivePricesResponse)
async def get_latest_prices() -> LivePricesResponse:
    """Fetch latest commodity futures prices.

    Returns current closing prices for all tracked commodities.
    Results are cached for 15 minutes to avoid rate limiting.
    """
    try:
        prices = _fetch_live_prices()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch prices: {e}") from e

    cached = prices.get("_cached", False)
    ts = prices.get("_ts", time.time())

    price_list = []
    for name, symbol in TICKERS.items():
        if name in prices:
            price_list.append(PriceData(
                name=name,
                ticker=symbol,
                price=round(prices[name], 4),
                currency="USD",
                timestamp=datetime.fromtimestamp(ts).isoformat(),
            ))

    return LivePricesResponse(
        prices=price_list,
        cached=cached,
        fetched_at=datetime.fromtimestamp(ts).isoformat(),
    )


@router.get("/predict/live")
async def predict_live() -> dict:
    """Predict BOC1 using the latest live commodity prices.

    Fetches current market prices, feeds them to the model,
    and returns the prediction alongside the input prices.
    """
    from src.serving.model_cache import get_model

    model, model_name = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    prices = _fetch_live_prices()

    feature_cols = FEATURE_COLUMNS
    missing = [c for c in feature_cols if c not in prices]
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch prices for: {missing}",
        )

    import numpy as np

    input_data = pd.DataFrame([{col: prices[col] for col in feature_cols}])

    try:
        prediction = model.predict(input_data)
        predicted_price = float(np.ravel(prediction)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    return {
        "predicted_price": round(predicted_price, 2),
        "model_name": model_name,
        "input_prices": {col: round(prices[col], 4) for col in feature_cols},
        "cached": prices.get("_cached", False),
        "fetched_at": datetime.fromtimestamp(prices.get("_ts", 0)).isoformat(),
    }


@router.get("/spreads")
async def get_spread_signals() -> dict:
    """Compute spread signals with 30-day MA, trend, and interpretation.

    Returns crush spread and oil/palm spread with actionable
    trading context for commodity risk managers.
    """
    from src.serving.spread_signals import (
        compute_trend,
        interpret_crush_spread,
        interpret_oil_palm_spread,
    )

    tickers_needed = {
        "boc1": TICKERS.get("boc1", "ZL=F"),
        "sc1": TICKERS.get("sc1", "ZS=F"),
        "smc1": TICKERS.get("smc1", "ZM=F"),
        "fcpoc1": TICKERS.get("fcpoc1", "PALM.L"),
    }

    history_data: dict[str, list[float]] = {}
    for name, symbol in tickers_needed.items():
        try:
            hist = yf.Ticker(symbol).history(period="60d")
            if not hist.empty:
                history_data[name] = hist["Close"].dropna().tolist()
        except Exception:
            logger.warning("spread_ticker_failed", ticker=symbol)

    signals = []

    # Crush spread: 11 * oil + meal - beans
    if all(k in history_data for k in ("boc1", "smc1", "sc1")):
        min_len = min(len(history_data["boc1"]), len(history_data["smc1"]), len(history_data["sc1"]))
        crush_series = [
            11 * history_data["boc1"][i] + history_data["smc1"][i] - history_data["sc1"][i]
            for i in range(min_len)
        ]
        current = crush_series[-1] if crush_series else 0
        ma30 = sum(crush_series[-30:]) / min(30, len(crush_series)) if crush_series else 0
        deviation_pct = ((current - ma30) / abs(ma30) * 100) if ma30 != 0 else 0
        trend = compute_trend(crush_series)
        interp = interpret_crush_spread(current, ma30, trend)

        signals.append({
            "name": "crush_spread",
            "label": "Crush Spread",
            "value": round(current, 2),
            "unit": "$/ton",
            "ma30": round(ma30, 2),
            "deviation_pct": round(deviation_pct, 1),
            "trend": trend,
            **interp,
        })

    # Oil/palm spread: boc1 - fcpoc1/100
    if all(k in history_data for k in ("boc1", "fcpoc1")):
        min_len = min(len(history_data["boc1"]), len(history_data["fcpoc1"]))
        op_series = [
            history_data["boc1"][i] - history_data["fcpoc1"][i] / 100
            for i in range(min_len)
        ]
        current = op_series[-1] if op_series else 0
        ma30 = sum(op_series[-30:]) / min(30, len(op_series)) if op_series else 0
        deviation_pct = ((current - ma30) / abs(ma30) * 100) if ma30 != 0 else 0
        trend = compute_trend(op_series)
        interp = interpret_oil_palm_spread(current, ma30, trend)

        signals.append({
            "name": "oil_palm_spread",
            "label": "Oil / Palm Spread",
            "value": round(current, 2),
            "unit": "c/lb",
            "ma30": round(ma30, 2),
            "deviation_pct": round(deviation_pct, 1),
            "trend": trend,
            **interp,
        })

    return {"spreads": signals}


@router.get("/backtest")
async def get_backtest() -> dict:
    """Return walk-forward out-of-sample backtest results.

    These predictions were generated at build time using TimeSeriesSplit
    with 5 folds. Each prediction was made using only past data —
    no look-ahead bias.
    """
    import numpy as np

    backtest_path = DATA_FOLDER / "walk_forward_backtest.parquet"
    if not backtest_path.exists():
        raise HTTPException(status_code=404, detail="Backtest data not available")

    df = pd.read_parquet(backtest_path)

    actual = df["actual"].values
    predicted = df["predicted"].values
    residuals = actual - predicted

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1 - np.sum(residuals ** 2) / np.sum((actual - actual.mean()) ** 2))

    true_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    dir_acc = float((true_dir == pred_dir).sum() / len(true_dir)) if len(true_dir) > 0 else 0.0

    points = []
    for _, row in df.iterrows():
        points.append({
            "date": int(row["row_index"]),
            "actual": round(float(row["actual"]), 2),
            "predicted": round(float(row["predicted"]), 2),
            "lower": round(float(row["lower"]), 2),
            "upper": round(float(row["upper"]), 2),
        })

    return {
        "points": points,
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "directional_accuracy": round(dir_acc, 4),
        },
        "n_points": len(points),
        "n_folds": int(df["fold"].nunique()),
        "model": "xgboost_baseline",
    }


@router.get("/prices/history")
async def get_price_history(days: int = 90) -> dict:
    """Return historical BOC1 prices with model predictions.

    Fetches daily closing prices from Yahoo Finance and runs the model
    on each day's feature set to produce a predicted vs actual chart.
    """
    import numpy as np

    from src.serving.model_cache import get_model

    days = min(days, 365)

    boc1_ticker = yf.Ticker(TICKERS.get("boc1", "ZL=F"))
    boc1_hist = boc1_ticker.history(period=f"{days}d")

    if boc1_hist.empty:
        raise HTTPException(status_code=502, detail="Could not fetch BOC1 history")

    feature_tickers = {
        name: yf.Ticker(symbol)
        for name, symbol in TICKERS.items()
        if name != "boc1" and name != "zc1"
    }

    feature_data = {}
    for name, ticker in feature_tickers.items():
        hist = ticker.history(period=f"{days}d")
        if not hist.empty:
            feature_data[name] = hist["Close"]

    model, model_name = get_model()

    result = []
    feature_cols = FEATURE_COLUMNS

    for date, row in boc1_hist.iterrows():
        actual = float(row["Close"])
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
        entry = {"date": date_str, "actual": round(actual, 2), "predicted": None}

        if model is not None:
            features = {}
            all_available = True
            for col in feature_cols:
                if col in feature_data:
                    nearest = feature_data[col].asof(date)
                    if pd.notna(nearest):
                        features[col] = float(nearest)
                    else:
                        all_available = False
                        break
                else:
                    all_available = False
                    break

            if all_available:
                try:
                    input_df = pd.DataFrame([features])
                    pred = model.predict(input_df)
                    entry["predicted"] = round(float(np.ravel(pred)[0]), 2)
                except Exception:
                    pass

        result.append(entry)

    return {
        "history": result,
        "model_name": model_name or "none",
        "days": len(result),
    }
