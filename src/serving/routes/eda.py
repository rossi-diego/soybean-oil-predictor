"""EDA endpoints — historical prices, correlations, distributions, spreads, seasonality, stationarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.config import CLEAN_DATA
from src.log import get_logger

logger = get_logger(__name__)

router = APIRouter()

LABELS = {
    "boc1": "Soybean Oil",
    "smc1": "Soybean Meal",
    "sc1": "Soybeans",
    "lcoc1": "Brent Crude",
    "hoc1": "Heating Oil",
    "fcpoc1": "Palm Oil",
    "rsc1": "Wheat",
}

_df_cache: pd.DataFrame | None = None


def _load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    if not CLEAN_DATA.exists():
        raise HTTPException(status_code=404, detail="Training data not available")
    df = pd.read_parquet(CLEAN_DATA)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    _df_cache = df
    return df


@router.get("/eda/prices")
async def eda_prices(period: str = "all") -> dict:
    """Historical commodity price series with optional normalization."""
    df = _load_data()
    cols = [c for c in LABELS if c in df.columns]

    if period != "all":
        years = {"1Y": 1, "3Y": 3, "5Y": 5}.get(period, None)
        if years:
            cutoff = df.index.max() - pd.DateOffset(years=years)
            df = df[df.index >= cutoff]

    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    series = {}
    normalized = {}
    for col in cols:
        vals = df[col].tolist()
        series[col] = [round(v, 2) if pd.notna(v) else None for v in vals]
        first = next((v for v in vals if pd.notna(v)), None)
        if first and first != 0:
            normalized[col] = [round(v / first * 100, 2) if pd.notna(v) else None for v in vals]

    return {
        "dates": dates,
        "series": series,
        "normalized": normalized,
        "n_points": len(dates),
        "labels": {c: LABELS[c] for c in cols},
    }


@router.get("/eda/correlations")
async def eda_correlations(method: str = "pearson") -> dict:
    """Correlation matrix between all features."""
    df = _load_data()
    cols = [c for c in LABELS if c in df.columns]
    sub = df[cols].dropna()

    if method == "spearman":
        corr = sub.corr(method="spearman")
    else:
        corr = sub.corr(method="pearson")

    return {
        "features": cols,
        "matrix": [[round(corr.iloc[i, j], 4) for j in range(len(cols))] for i in range(len(cols))],
        "method": method,
        "labels": {c: LABELS[c] for c in cols},
    }


@router.get("/eda/rolling-correlations")
async def eda_rolling_correlations(window: int = 60) -> dict:
    """Rolling correlation between BOC1 and each other commodity."""
    df = _load_data()
    target = "boc1"
    cols = [c for c in LABELS if c in df.columns and c != target]
    window = min(max(window, 20), 252)

    dates = []
    series: dict[str, list] = {c: [] for c in cols}

    sub = df[[target] + cols].dropna()
    for i in range(window, len(sub)):
        chunk = sub.iloc[i - window : i]
        dates.append(chunk.index[-1].strftime("%Y-%m-%d"))
        for c in cols:
            r = float(chunk[target].corr(chunk[c]))
            series[c].append(round(r, 4) if pd.notna(r) else None)

    return {
        "dates": dates,
        "series": series,
        "window": window,
        "target": target,
        "labels": {c: LABELS[c] for c in cols},
    }


@router.get("/eda/distributions")
async def eda_distributions() -> dict:
    """Histogram and quartile data for each feature, plus recent 30-day window."""
    df = _load_data()
    cols = [c for c in LABELS if c in df.columns]
    recent = df.tail(30)
    result = {}

    for col in cols:
        vals = df[col].dropna()
        recent_vals = recent[col].dropna()

        hist_counts, hist_edges = np.histogram(vals, bins=25)
        recent_counts, _ = np.histogram(recent_vals, bins=hist_edges)

        bins = []
        for i in range(len(hist_counts)):
            bins.append({
                "x": round(float(hist_edges[i]), 2),
                "x_end": round(float(hist_edges[i + 1]), 2),
                "count": int(hist_counts[i]),
                "recent_count": int(recent_counts[i]),
            })

        q = vals.quantile([0, 0.25, 0.5, 0.75, 1.0])
        result[col] = {
            "bins": bins,
            "quartiles": {
                "min": round(float(q[0.0]), 2),
                "q1": round(float(q[0.25]), 2),
                "median": round(float(q[0.5]), 2),
                "q3": round(float(q[0.75]), 2),
                "max": round(float(q[1.0]), 2),
            },
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std()), 2),
            "n": len(vals),
            "recent_n": len(recent_vals),
        }

    return {"features": result, "labels": {c: LABELS[c] for c in cols}}


@router.get("/eda/spreads")
async def eda_spreads() -> dict:
    """Historical spread time series with percentile bands."""
    df = _load_data()
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    result: dict = {"dates": dates, "spreads": {}}

    # CME Board Crush: (meal * 0.022) + (oil * 0.11) - (beans / 100), result $/bu
    if all(c in df.columns for c in ["boc1", "smc1", "sc1"]):
        crush = (df["smc1"] * 0.022) + (df["boc1"] * 0.11) - (df["sc1"] / 100)
        p10, p90 = float(crush.quantile(0.1)), float(crush.quantile(0.9))
        result["spreads"]["crush"] = {
            "values": [round(v, 2) if pd.notna(v) else None for v in crush],
            "label": "Board Crush",
            "unit": "$/bu",
            "p10": round(p10, 2),
            "p90": round(p90, 2),
            "mean": round(float(crush.mean()), 2),
        }

    # BOPO: soy oil USD/mt - palm oil (fcpoc1 is MYR/mt in training data,
    # approximate conversion to USD/mt using historical avg ~4.3 MYR/USD)
    if all(c in df.columns for c in ["boc1", "fcpoc1"]):
        boc1_usd_mt = df["boc1"] * 22.0462
        # fcpoc1 in training data is MYR/mt, convert to USD/mt (~4.3 MYR/USD avg)
        cpo_usd_mt = df["fcpoc1"] / 4.3
        bopo = boc1_usd_mt - cpo_usd_mt
        p10, p90 = float(bopo.quantile(0.1)), float(bopo.quantile(0.9))
        result["spreads"]["bopo"] = {
            "values": [round(v, 0) if pd.notna(v) else None for v in bopo],
            "label": "BOPO Spread",
            "unit": "$/mt",
            "p10": round(p10, 0),
            "p90": round(p90, 0),
            "mean": round(float(bopo.mean()), 0),
        }

    # Soy/Corn ratio (use rsc1 as proxy since training data has rsc1 not zc1 separately)
    # rsc1 in training data was originally canola/wheat, but for EDA we show what we have
    if all(c in df.columns for c in ["sc1", "rsc1"]):
        ratio = df["sc1"] / df["rsc1"].replace(0, np.nan)
        p10, p90 = float(ratio.quantile(0.1)), float(ratio.quantile(0.9))
        result["spreads"]["soy_grain"] = {
            "values": [round(v, 2) if pd.notna(v) else None for v in ratio],
            "label": "Soy / Grain Ratio",
            "unit": "ratio",
            "p10": round(p10, 2),
            "p90": round(p90, 2),
            "mean": round(float(ratio.mean()), 2),
        }

    return result


@router.get("/eda/seasonality")
async def eda_seasonality() -> dict:
    """Monthly BOC1 statistics for seasonal analysis."""
    df = _load_data()
    if "boc1" not in df.columns:
        raise HTTPException(status_code=404, detail="BOC1 data not available")

    df_m = df[["boc1"]].dropna().copy()
    df_m["month"] = df_m.index.month

    months = []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for m in range(1, 13):
        vals = df_m[df_m["month"] == m]["boc1"]
        if len(vals) == 0:
            continue
        q = vals.quantile([0, 0.25, 0.5, 0.75, 1.0])
        months.append({
            "month": m,
            "name": month_names[m - 1],
            "min": round(float(q[0.0]), 2),
            "q1": round(float(q[0.25]), 2),
            "median": round(float(q[0.5]), 2),
            "q3": round(float(q[0.75]), 2),
            "max": round(float(q[1.0]), 2),
            "mean": round(float(vals.mean()), 2),
            "n": len(vals),
        })

    return {"months": months, "target": "boc1"}


@router.get("/eda/stationarity")
async def eda_stationarity() -> dict:
    """Returns distribution, ADF tests, and ACF/PACF for time series analysis."""

    df = _load_data()
    cols = [c for c in LABELS if c in df.columns]

    # ADF tests
    from statsmodels.tsa.stattools import acf, adfuller, pacf

    adf_results = {}
    for col in cols:
        series = df[col].dropna()
        try:
            stat, pvalue, lags, nobs, crit, _ = adfuller(series, maxlag=20)
            adf_results[col] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(pvalue), 6),
                "lags": int(lags),
                "n_obs": int(nobs),
                "stationary": bool(pvalue < 0.05),
                "critical_values": {k: round(float(v), 4) for k, v in crit.items()},
            }
        except Exception:
            adf_results[col] = {"stationary": None, "p_value": None}

    # Returns distribution (BOC1)
    boc1 = df["boc1"].dropna()
    returns = boc1.pct_change().dropna()
    ret_counts, ret_edges = np.histogram(returns, bins=40)
    returns_hist = []
    for i in range(len(ret_counts)):
        returns_hist.append({
            "x": round(float(ret_edges[i]) * 100, 2),
            "x_end": round(float(ret_edges[i + 1]) * 100, 2),
            "count": int(ret_counts[i]),
        })

    # ACF / PACF for BOC1 returns (not levels — levels are autocorrelated by definition)
    try:
        acf_vals = acf(returns, nlags=30, fft=True)
        pacf_vals = pacf(returns, nlags=30)
    except Exception:
        acf_vals = np.zeros(31)
        pacf_vals = np.zeros(31)
    n = len(returns)
    ci = 1.96 / np.sqrt(n) if n > 0 else 0

    acf_data = [{"lag": i, "value": round(float(v), 4)} for i, v in enumerate(acf_vals)]
    pacf_data = [{"lag": i, "value": round(float(v), 4)} for i, v in enumerate(pacf_vals)]

    return {
        "adf_tests": adf_results,
        "returns_hist": returns_hist,
        "returns_stats": {
            "mean": round(float(returns.mean()) * 100, 4),
            "std": round(float(returns.std()) * 100, 4),
            "skew": round(float(returns.skew()), 4),
            "kurtosis": round(float(returns.kurtosis()), 4),
        },
        "acf": acf_data,
        "pacf": pacf_data,
        "confidence_interval": round(float(ci), 4),
        "labels": {c: LABELS[c] for c in cols},
    }
