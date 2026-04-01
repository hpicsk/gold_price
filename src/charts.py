"""Generate charts for the GitHub Pages site."""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DOCS_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs"
IMG_DIR = DOCS_DIR / "images"

COLORS = {
    "gold": "#D4A843",
    "dark": "#1a1a2e",
    "accent": "#c0392b",
    "blue": "#2980b9",
    "green": "#27ae60",
    "gray": "#7f8c8d",
}


def setup():
    """Create output directory and set style."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})


def plot_actual_vs_predicted(y_test: pd.Series, predictions: dict, save: bool = True):
    """Plot gold price actual vs predicted for all models."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test.values, color=COLORS["gold"], linewidth=2, label="Actual", marker="o", markersize=3)

    model_colors = [COLORS["blue"], COLORS["green"], COLORS["accent"]]
    for (name, y_pred), color in zip(predictions.items(), model_colors):
        ax.plot(y_test.index, y_pred, color=color, linewidth=1.5, linestyle="--", label=name)

    ax.set_title("Gold Price: Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price (USD/oz)")
    ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=45)

    if save:
        fig.savefig(IMG_DIR / "actual_vs_predicted.png")
    plt.close(fig)
    print("  Saved actual_vs_predicted.png")


def plot_feature_importance(importance: dict, save: bool = True):
    """Plot feature importance from Gradient Boosting."""
    gb_imp = importance["Gradient Boosting"]
    features = list(gb_imp.keys())
    values = list(gb_imp.values())

    # Sort by importance
    idx = np.argsort(values)
    features = [features[i] for i in idx]
    values = [values[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, values, color=COLORS["gold"], edgecolor=COLORS["dark"], linewidth=0.5)
    ax.set_title("Feature Importance (Gradient Boosting)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")

    if save:
        fig.savefig(IMG_DIR / "feature_importance.png")
    plt.close(fig)
    print("  Saved feature_importance.png")


def plot_coefficients(importance: dict, save: bool = True):
    """Plot linear regression coefficients (standardized)."""
    ols_coef = importance["OLS (Linear Regression)"]
    features = list(ols_coef.keys())
    values = list(ols_coef.values())

    colors = [COLORS["green"] if v > 0 else COLORS["accent"] for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, values, color=colors, edgecolor=COLORS["dark"], linewidth=0.5)
    ax.axvline(x=0, color=COLORS["gray"], linewidth=0.8)
    ax.set_title("Linear Regression Coefficients (Standardized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Coefficient Value")

    if save:
        fig.savefig(IMG_DIR / "coefficients.png")
    plt.close(fig)
    print("  Saved coefficients.png")


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    """Plot correlation heatmap of all features."""
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")

    if save:
        fig.savefig(IMG_DIR / "correlation_heatmap.png")
    plt.close(fig)
    print("  Saved correlation_heatmap.png")


def plot_scatter_indicators(df: pd.DataFrame, save: bool = True):
    """Plot scatter plots of gold price vs each indicator."""
    features = [c for c in df.columns if c != "gold_price"]
    n_features = len(features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.scatter(df[feat], df["gold_price"], alpha=0.5, s=15, color=COLORS["gold"], edgecolor=COLORS["dark"], linewidth=0.3)
        ax.set_xlabel(feat)
        ax.set_ylabel("Gold Price (USD/oz)")
        ax.set_title(f"Gold vs {feat}", fontsize=10, fontweight="bold")

        # Add trendline
        z = np.polyfit(df[feat], df["gold_price"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS["accent"], linewidth=1.5, linestyle="--")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Gold Price vs Individual Indicators", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        fig.savefig(IMG_DIR / "scatter_indicators.png")
    plt.close(fig)
    print("  Saved scatter_indicators.png")


def plot_gold_timeseries(df: pd.DataFrame, save: bool = True):
    """Plot gold price time series."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["gold_price"], color=COLORS["gold"], linewidth=1.5)
    ax.fill_between(df.index, df["gold_price"], alpha=0.2, color=COLORS["gold"])
    ax.set_title("Gold Price (USD/oz) — Historical", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/oz)")
    ax.tick_params(axis="x", rotation=45)

    if save:
        fig.savefig(IMG_DIR / "gold_timeseries.png")
    plt.close(fig)
    print("  Saved gold_timeseries.png")


def generate_all(df: pd.DataFrame, y_test: pd.Series, predictions: dict, importance: dict):
    """Generate all charts."""
    setup()
    print("\nGenerating charts...")
    plot_gold_timeseries(df)
    plot_actual_vs_predicted(y_test, predictions)
    plot_feature_importance(importance)
    plot_coefficients(importance)
    plot_correlation_heatmap(df)
    plot_scatter_indicators(df)
    print("All charts saved to docs/images/")
