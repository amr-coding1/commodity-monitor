# Cross-Commodity Inventory & Term Structure Monitor

Tracks exchange-reported inventory levels across LME metals and ICE soft commodities, analysing their relationship with calendar spreads to quantify the storage economics that drive physical commodity trading.

## Motivation

In physical commodity trading, the relationship between inventory levels and the futures term structure is fundamental. When warehouse stocks draw down, spot premiums rise and the curve moves into backwardation — the market pays a premium for immediate delivery. When stocks build, contango widens as the market charges for storage. This project monitors that dynamic across five commodities spanning two asset classes.

## Commodities Covered

| Commodity | Exchange | Category |
|-----------|----------|----------|
| Copper | LME / COMEX | Metals |
| Aluminium | LME / COMEX | Metals |
| Coffee Arabica | ICE | Softs |
| Cocoa | ICE | Softs |
| Sugar #11 | ICE | Softs |

## Methodology

### Inventory Normalisation
- **Rolling Z-scores** (1-year and 3-year windows) normalise stock levels across commodities with different units and scales
- **Stocks-to-use ratios** convert absolute inventory to days of consumption coverage using USDA/USGS annual data

### Term Structure Analysis
- **Calendar spreads** computed from actual M1−M2 continuous contracts via Nasdaq Data Link CHRIS database
- **Regime classification**: backwardation (M1 > M2), contango (M1 < M2), or flat (< 0.2% of M1)

### Storage Economics Quantification
- **Lagged Pearson correlations** (0, 1, 2, 4 weeks) with Fisher z-transform confidence intervals
- **OLS sensitivity** (β): spread percentage change per unit z-score change
- **Walk-forward validation** (60/40 split) to test correlation stability out-of-sample
- **Signal alignment**: detects when inventory and term structure send conflicting signals (divergence often precedes regime shifts)

## Data Sources

| Data | Source | Auth | Fallback |
|------|--------|------|----------|
| LME warehouse stocks | lme.com reports | Free | Synthetic (OU process) |
| ICE certified stocks | ice.com public reports | None | Synthetic (OU process) |
| Futures M1 + M2 | Nasdaq Data Link CHRIS | Free API key | yfinance M1 + synthetic M2 |
| Softs consumption | USDA FAS PSD API | Free API key | — |
| Metals consumption | USGS Mineral Commodity Summaries | None | Hardcoded estimates |

**Note on synthetic data:** LME and ICE block automated downloads (Cloudflare, rate limiting). When live data is unavailable, the ingestion modules generate deterministic synthetic inventory using Ornstein-Uhlenbeck mean-reverting processes calibrated to real-world stock ranges. Similarly, when only M1 futures are available (yfinance), a synthetic M2 is generated using a mean-reverting carry model. This allows the full analytics pipeline to run and demonstrate the methodology. With API keys configured (Nasdaq Data Link for CHRIS M1+M2, USDA for consumption), the pipeline uses real data automatically.

## Dashboard

Four views via Streamlit:

1. **Overview** — Heatmap/table of all commodities with z-scores, regimes, and signal alignment. Red = tight, green = surplus, amber = divergent
2. **Individual Commodity** — Dual-axis time series (Plotly), scatter plots, rolling correlation, summary statistics
3. **Cross-Commodity** — Normalised z-score overlay, sensitivity comparison bar chart, correlation matrix heatmap
4. **Regime Analysis** — Current regime table, z-score with regime shading, threshold crossing backtest

## Setup

```bash
# Clone and install
cd commodity-monitor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Nasdaq Data Link and USDA API keys

# Run ingestion
python scripts/run_ingestion.py

# Run analysis
python scripts/run_analysis.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
commodity-monitor/
├── config/commodities.json       # Commodity metadata and tickers
├── src/
│   ├── settings.py               # Paths, API config, constants
│   ├── database.py               # SQLite schema and CRUD
│   ├── ingestion/                # Data fetchers (LME, ICE, futures, consumption)
│   ├── processing/               # Z-scores, spreads, regime classification
│   ├── analysis/                 # Correlations, sensitivity, cross-commodity
│   └── reporting/                # Charts (matplotlib) and Jinja2 commentary
├── dashboard/app.py              # Streamlit dashboard
├── scripts/                      # CLI entry points
├── tests/                        # 63 pytest tests
└── templates/                    # Jinja2 report template
```

## Tests

```bash
python -m pytest tests/ -v
```

63 tests covering z-score computation, regime classification, signal alignment, Pearson with Fisher CI, walk-forward validation, OLS sensitivity, URL builders, and database CRUD.

## Limitations

- **Exchange data access**: LME (Cloudflare) and ICE (rate limiting) block automated downloads — inventory data falls back to calibrated synthetic series. Live data works when fetching manually or with appropriate access
- **CHRIS database** on Nasdaq Data Link has intermittent gaps; yfinance fallback provides M1 only, with synthetic M2 for spread analytics
- **Consumption data** for metals uses hardcoded USGS estimates rather than a live API
- **Z-score normalisation** assumes roughly stationary inventory distributions — structural supply shifts (e.g., new mine openings, crop failures) may invalidate historical z-scores
- **Correlation ≠ causation** — the inventory-spread relationship is well-established in theory but the observed statistical relationships may weaken during regime transitions
- **Synthetic data caveat** — the synthetic inventory and M2 data demonstrate the methodology but do not reflect actual market conditions. With proper API keys, the pipeline switches to real data automatically
