"""Fetch gold price and indicator data from FRED and yfinance."""

import os
import pathlib

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

FRED_SERIES = {
    "gold_price": "GOLDAMGBD228NLBM",
    "real_rate": "DFII10",
    "wti_oil": "DCOILWTICO",
    "usd_index": "DTWEXBGS",
}

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"


def fetch_fred(api_key: str | None = None) -> dict[str, pd.Series]:
    """Fetch all FRED series. Returns dict of name -> Series."""
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY not set")

    fred = Fred(api_key=key)
    data = {}
    for name, series_id in FRED_SERIES.items():
        print(f"  Fetching FRED {series_id} ({name})...")
        s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
        s.name = name
        data[name] = s
    return data


def fetch_yfinance_dxy() -> pd.Series:
    """Fetch DXY from Yahoo Finance as fallback for USD index."""
    import yfinance as yf

    print("  Fetching DXY from Yahoo Finance...")
    ticker = yf.Ticker("DX-Y.NYB")
    df = ticker.history(start=START_DATE, end=END_DATE)
    s = df["Close"]
    s.name = "usd_index"
    s.index = s.index.tz_localize(None)
    return s


def cache_raw(data: dict[str, pd.Series]) -> None:
    """Save raw series to CSV for caching."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, s in data.items():
        path = RAW_DIR / f"{name}.csv"
        s.to_frame(name).to_csv(path)
        print(f"  Cached {path}")


def load_cached() -> dict[str, pd.Series] | None:
    """Load cached raw data if available."""
    if not RAW_DIR.exists():
        return None
    data = {}
    for name in FRED_SERIES:
        path = RAW_DIR / f"{name}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        data[name] = df.iloc[:, 0]
    return data


def load_local_csvs() -> dict[str, pd.DataFrame]:
    """Load manually curated CSVs (central bank purchases, AISC)."""
    cb = pd.read_csv(DATA_DIR / "central_bank_purchases.csv", parse_dates=["date"], index_col="date")
    aisc = pd.read_csv(DATA_DIR / "aisc_gold_mining.csv", parse_dates=["date"], index_col="date")
    return {"cb_purchases": cb, "aisc": aisc}


def generate_sample_data() -> dict[str, pd.Series]:
    """Generate realistic sample data when no API key is available."""
    import numpy as np

    print("  No FRED API key found. Generating sample data...")
    dates = pd.date_range(START_DATE, END_DATE, freq="B")
    np.random.seed(42)
    n = len(dates)

    # Gold price: trending up from ~1100 to ~2600 with noise
    t = np.linspace(0, 1, n)
    gold = 1100 + 1500 * (t ** 1.3) + np.cumsum(np.random.normal(0, 3, n))
    gold = np.clip(gold, 1000, 3200)

    # Real rate: oscillates roughly -1% to 2%
    real_rate = 0.5 * np.sin(2 * np.pi * t * 3) - 0.2 + np.cumsum(np.random.normal(0, 0.02, n))
    real_rate = np.clip(real_rate, -2.5, 3.0)

    # WTI oil: 40-120 range
    oil = 65 + 25 * np.sin(2 * np.pi * t * 2.5) + np.cumsum(np.random.normal(0, 0.5, n))
    oil = np.clip(oil, 15, 130)

    # USD index: 95-130 range
    usd = 110 + 10 * np.sin(2 * np.pi * t * 2) + np.cumsum(np.random.normal(0, 0.1, n))
    usd = np.clip(usd, 85, 135)

    return {
        "gold_price": pd.Series(gold, index=dates, name="gold_price"),
        "real_rate": pd.Series(real_rate, index=dates, name="real_rate"),
        "wti_oil": pd.Series(oil, index=dates, name="wti_oil"),
        "usd_index": pd.Series(usd, index=dates, name="usd_index"),
    }


def fetch_all(api_key: str | None = None, skip_fetch: bool = False) -> dict[str, pd.Series]:
    """Main entry point: fetch or load all market data."""
    if skip_fetch:
        cached = load_cached()
        if cached:
            print("  Using cached data.")
            return cached

    # Try FRED first
    key = api_key or os.environ.get("FRED_API_KEY")
    if key:
        try:
            data = fetch_fred(key)
            cache_raw(data)
            return data
        except Exception as e:
            print(f"  FRED fetch failed: {e}")

    # Try cached data
    cached = load_cached()
    if cached:
        print("  Using cached data.")
        return cached

    # Fall back to sample data
    data = generate_sample_data()
    cache_raw(data)
    return data
