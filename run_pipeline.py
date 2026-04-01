#!/usr/bin/env python3
"""Gold Price Regression Model — Full Pipeline.

Usage:
    python run_pipeline.py                     # uses FRED_API_KEY env var or sample data
    python run_pipeline.py --fred-api-key KEY  # provide key directly
    python run_pipeline.py --skip-fetch        # use cached/sample data
"""

import argparse
import sys

from src.fetch_data import fetch_all
from src.prepare_data import prepare_features, train_test_split, get_feature_target
from src.model import GoldPriceModels
from src.charts import generate_all


def main():
    parser = argparse.ArgumentParser(description="Gold Price Regression Pipeline")
    parser.add_argument("--fred-api-key", type=str, default=None, help="FRED API key")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching, use cached data")
    args = parser.parse_args()

    # Step 1: Fetch data
    print("=" * 60)
    print("Step 1: Fetching data...")
    print("=" * 60)
    market_data = fetch_all(api_key=args.fred_api_key, skip_fetch=args.skip_fetch)

    # Step 2: Prepare features
    print("\n" + "=" * 60)
    print("Step 2: Preparing features...")
    print("=" * 60)
    df = prepare_features(market_data)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Features: {list(df.columns)}")

    # Step 3: Train/test split
    print("\n" + "=" * 60)
    print("Step 3: Splitting data...")
    print("=" * 60)
    train, test = train_test_split(df, test_months=18)
    X_train, y_train = get_feature_target(train)
    X_test, y_test = get_feature_target(test)
    print(f"  Train: {len(train)} months, Test: {len(test)} months")

    # Step 4: Train models
    print("\n" + "=" * 60)
    print("Step 4: Training models...")
    print("=" * 60)
    models = GoldPriceModels()
    models.train(X_train, y_train)

    # Step 5: Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluating models...")
    print("=" * 60)
    metrics_df = models.evaluate(X_test, y_test)
    print("\n" + metrics_df.to_string(index=False))

    # Step 6: Generate charts
    print("\n" + "=" * 60)
    print("Step 6: Generating charts...")
    print("=" * 60)
    predictions = models.predict(X_test)
    importance = models.get_feature_importance()
    generate_all(df, y_test, predictions, importance)

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Charts saved to: docs/images/")
    print(f"  Open docs/index.html in a browser to view the site.")

    return metrics_df, models


if __name__ == "__main__":
    main()
