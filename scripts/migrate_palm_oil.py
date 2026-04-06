"""Migrate palm oil data: rename fcpoc1 -> palm_oil, convert MYR/mt -> USD/mt.

Historical data (fcpoc1) is in MYR/mt from Bursa Malaysia FCPO.
New data uses CPO=F from CME which is already in USD/mt.

This script:
1. Loads the existing clean parquet
2. Fetches MYR/USD exchange rate history from yfinance
3. Converts historical fcpoc1 (MYR/mt) to USD/mt using daily FX rates
4. Fetches CPO=F daily data to fill the gap from where fcpoc1 ends
5. Merges and saves as palm_oil column
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_FOLDER = Path(__file__).resolve().parents[1] / "data"
CLEAN_DATA = DATA_FOLDER / "commodities_clean_data.parquet"


def migrate():
    df = pd.read_parquet(CLEAN_DATA)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if "fcpoc1" not in df.columns:
        if "palm_oil" in df.columns:
            print("Already migrated — palm_oil column exists.")
            return
        raise ValueError("No fcpoc1 or palm_oil column found")

    # 1. Fetch MYR/USD exchange rate history
    print("Fetching MYR=X (MYR per USD) exchange rate...")
    fx = yf.Ticker("MYR=X").history(period="max")["Close"]
    fx.index = fx.index.tz_localize(None)
    fx = fx.sort_index()
    print(f"  FX data: {len(fx)} rows, {fx.index.min().date()} to {fx.index.max().date()}")

    # 2. Convert historical fcpoc1 from MYR/mt to USD/mt
    fcpoc1 = df["fcpoc1"].dropna()
    print(f"  fcpoc1: {len(fcpoc1)} rows, {fcpoc1.index.min().date()} to {fcpoc1.index.max().date()}")

    # Align FX to fcpoc1 dates using forward-fill
    fx_aligned = fx.reindex(fcpoc1.index, method="ffill")
    # Fill any remaining NaN with backfill
    fx_aligned = fx_aligned.fillna(method="bfill")
    # For dates before FX data starts, use the median rate
    fx_aligned = fx_aligned.fillna(fx_aligned.median())

    palm_oil_usd = fcpoc1 / fx_aligned
    palm_oil_usd.name = "palm_oil"
    print(f"  Converted: last MYR value {fcpoc1.iloc[-1]:.0f} MYR/mt / {fx_aligned.iloc[-1]:.4f} MYR/USD = {palm_oil_usd.iloc[-1]:.2f} USD/mt")

    # 3. Fetch CPO=F (CME, already USD/mt) to fill the gap
    last_fcpoc1_date = fcpoc1.index.max()
    print(f"\nFetching CPO=F (CME USD/mt) from {last_fcpoc1_date.date()}...")
    cpo = yf.Ticker("CPO=F").history(start=last_fcpoc1_date.strftime("%Y-%m-%d"))["Close"]
    cpo.index = cpo.index.tz_localize(None)
    cpo = cpo.sort_index()
    # Only keep dates after the last fcpoc1 date to avoid overlap
    cpo_new = cpo[cpo.index > last_fcpoc1_date]
    print(f"  CPO=F new data: {len(cpo_new)} rows")
    if len(cpo_new) > 0:
        print(f"  Range: {cpo_new.index.min().date()} to {cpo_new.index.max().date()}")
        print(f"  Last price: {cpo_new.iloc[-1]:.2f} USD/mt")

    # 4. Merge: historical converted + new CPO=F
    palm_combined = pd.concat([palm_oil_usd, cpo_new.rename("palm_oil")])
    palm_combined = palm_combined[~palm_combined.index.duplicated(keep="last")]
    palm_combined = palm_combined.sort_index()

    # 5. Update the dataframe
    df["palm_oil"] = palm_combined.reindex(df.index)
    df = df.drop(columns=["fcpoc1"])

    # 6. Save
    df.to_parquet(CLEAN_DATA, index=True)
    print(f"\nSaved {CLEAN_DATA}")
    print(f"  Columns: {list(df.columns)}")

    palm = df["palm_oil"].dropna()
    print(f"  palm_oil: {len(palm)} rows, {palm.index.min().date()} to {palm.index.max().date()}")
    print(f"  Last value: {palm.iloc[-1]:.2f} USD/mt")
    last_6m = df.loc[df.index >= (df.index.max() - pd.Timedelta(days=180)), "palm_oil"]
    print(f"  Last 6 months: {last_6m.notna().sum()} non-null out of {len(last_6m)} rows")


if __name__ == "__main__":
    migrate()
