"""Live commodity prices from Yahoo Finance."""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import TICKERS
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

    feature_cols = ["smc1", "sc1", "lcoc1", "hoc1", "fcpoc1", "rsc1"]
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
