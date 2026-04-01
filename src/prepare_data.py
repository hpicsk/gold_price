"""Prepare and merge all data into a single monthly DataFrame."""

import pandas as pd
import numpy as np

from .fetch_data import load_local_csvs


def resample_to_monthly(series: pd.Series) -> pd.Series:
    """Resample a daily series to month-end frequency using last value."""
    return series.resample("ME").last().dropna()


def prepare_features(market_data: dict[str, pd.Series]) -> pd.DataFrame:
    """Merge all indicators into a single monthly DataFrame."""
    # Resample market data to monthly
    monthly = {}
    for name, s in market_data.items():
        monthly[name] = resample_to_monthly(s)

    # Combine into DataFrame
    df = pd.DataFrame(monthly)

    # Load and merge local CSVs
    local = load_local_csvs()

    # Central bank purchases: quarterly -> monthly via forward-fill
    cb = local["cb_purchases"].resample("ME").ffill()
    cb.columns = ["cb_purchases"]

    # AISC: annual -> monthly via forward-fill
    aisc = local["aisc"].resample("ME").ffill()
    aisc.columns = ["aisc"]

    # Merge all
    df = df.join(cb, how="left")
    df = df.join(aisc, how="left")

    # Forward-fill remaining gaps in cb_purchases and aisc
    df["cb_purchases"] = df["cb_purchases"].ffill()
    df["aisc"] = df["aisc"].ffill()

    # Derived feature: gold-oil ratio
    if "gold_price" in df.columns and "wti_oil" in df.columns:
        df["gold_oil_ratio"] = df["gold_price"] / df["wti_oil"].replace(0, np.nan)

    # Drop rows with any NaN
    df = df.dropna()

    return df


def train_test_split(df: pd.DataFrame, test_months: int = 18):
    """Time-based train/test split."""
    split_idx = len(df) - test_months
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def get_feature_target(df: pd.DataFrame):
    """Split into features X and target y."""
    feature_cols = [c for c in df.columns if c != "gold_price"]
    X = df[feature_cols]
    y = df["gold_price"]
    return X, y
