"""Central configuration — paths, API endpoints, analysis constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "commodities.db"
TEMPLATES_DIR = ROOT_DIR / "templates"
CHARTS_DIR = DATA_DIR / "charts"

# Ensure runtime directories exist
for _d in (DATA_DIR, CACHE_DIR, CHARTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API keys ─────────────────────────────────────────────────────────────
NASDAQ_API_KEY: str | None = os.getenv("NASDAQ_DATA_LINK_API_KEY")
USDA_API_KEY: str | None = os.getenv("USDA_API_KEY")

# ── API endpoints ────────────────────────────────────────────────────────
LME_WAREHOUSE_URL = (
    "https://www.lme.com/api/Lists/DownloadLinks"
    "?pageId=5F3EFCBC-72E0-4644-8043-7B6F26C68ED3"
)
ICE_CERTIFIED_STOCK_URL = (
    "https://www.ice.com/publicdocs/futures_us_reports"
    "/{commodity}/{commodity}_cert_stock_{date}.xls"
)
USDA_PSD_API_URL = "https://apps.fas.usda.gov/PSDOnlineService/api/CommodityData"
USGS_MINERALS_URL = (
    "https://minerals.usgs.gov/minerals/pubs/commodity"
    "/{commodity}/mcs-{year}-{commodity}.pdf"
)

# ── Analysis constants ───────────────────────────────────────────────────
ZSCORE_WINDOW_1Y: int = 252        # trading days
ZSCORE_WINDOW_3Y: int = 756
WALK_FORWARD_TRAIN_FRACTION: float = 0.6
CORRELATION_LAGS_WEEKS: list[int] = [0, 1, 2, 4]
ROLLING_CORR_WINDOW: int = 63      # ~3 months of trading days
