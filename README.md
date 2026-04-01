# Gold Price Regression Model

A multi-factor regression model for gold price valuation using 5 key economic indicators.

## Indicators

1. **Real Interest Rate** (10Y TIPS yield) — inverse relationship with gold
2. **US Dollar Index** (trade-weighted) — inverse relationship
3. **Central Bank Gold Purchases** — positive relationship
4. **Gold Mining Cost (AISC)** — acts as price floor
5. **Oil Price (WTI)** — positive correlation (USD-denominated commodity)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# With FRED API key (fetches live data)
export FRED_API_KEY=your_key_here
python run_pipeline.py

# Without API key (uses bundled sample data)
python run_pipeline.py
```

The pipeline fetches data, trains 3 regression models (OLS, Ridge, Gradient Boosting), generates charts, and outputs results to `docs/images/`.

## GitHub Pages

The `docs/` folder contains a static site presenting the model results. Enable GitHub Pages from the `docs/` folder in repository settings.

## Data Sources

- [FRED](https://fred.stlouisfed.org/) — gold price, real rates, WTI oil, USD index
- [World Gold Council](https://www.gold.org/goldhub/data) — central bank purchases, AISC
- [Yahoo Finance](https://finance.yahoo.com/) — DXY fallback
