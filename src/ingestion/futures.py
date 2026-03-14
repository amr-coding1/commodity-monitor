"""Fetch futures prices from Nasdaq Data Link (CHRIS) with yfinance fallback.

When only M1 is available (yfinance), generates a synthetic M2 using a
mean-reverting contango/backwardation model so the spread analytics pipeline
can still demonstrate the full methodology.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.database import get_commodity_meta, upsert_futures_batch
from src.settings import NASDAQ_API_KEY

logger = logging.getLogger(__name__)

# Typical carry as % of M1, calibrated to real-world observations
# Positive = typical contango, negative = typical backwardation
CARRY_PARAMS: dict[str, dict[str, float]] = {
    "copper":    {"mean_carry_pct": 0.003, "vol": 0.004, "mr": 0.02},
    "aluminium": {"mean_carry_pct": 0.005, "vol": 0.005, "mr": 0.02},
    "coffee":    {"mean_carry_pct": -0.002, "vol": 0.006, "mr": 0.015},
    "cocoa":     {"mean_carry_pct": -0.003, "vol": 0.008, "mr": 0.01},
    "sugar":     {"mean_carry_pct": 0.004, "vol": 0.005, "mr": 0.02},
}


def fetch_futures_chris(commodity: str) -> pd.DataFrame | None:
    """Fetch M1 and M2 continuous futures from Nasdaq Data Link CHRIS database.

    Returns DataFrame with columns [date, m1_close, m2_close, spread] or None.
    """
    if not NASDAQ_API_KEY:
        logger.warning("NASDAQ_DATA_LINK_API_KEY not set — skipping CHRIS fetch")
        return None

    import nasdaqdatalink

    nasdaqdatalink.ApiConfig.api_key = NASDAQ_API_KEY
    meta = get_commodity_meta(commodity)
    m1_code = meta["futures"]["chris_m1"]
    m2_code = meta["futures"]["chris_m2"]

    try:
        m1 = nasdaqdatalink.get(m1_code, returns="pandas")
        m2 = nasdaqdatalink.get(m2_code, returns="pandas")
    except Exception as e:
        logger.warning("CHRIS fetch failed for %s: %s", commodity, e)
        return None

    # CHRIS datasets use 'Settle' or 'Last' for close price
    m1_col = "Settle" if "Settle" in m1.columns else "Last"
    m2_col = "Settle" if "Settle" in m2.columns else "Last"

    m1_series = m1[m1_col].rename("m1_close")
    m2_series = m2[m2_col].rename("m2_close")

    merged = pd.concat([m1_series, m2_series], axis=1, join="inner")
    merged = merged.dropna()
    merged["spread"] = merged["m1_close"] - merged["m2_close"]
    merged.index.name = "date"
    merged = merged.reset_index()
    merged["date"] = pd.to_datetime(merged["date"]).dt.date.astype(str)

    logger.info(
        "CHRIS: fetched %d rows for %s (M1+M2)",
        len(merged), commodity,
    )
    return merged


def fetch_futures_yfinance(commodity: str) -> pd.DataFrame | None:
    """Fallback: fetch M1 continuous futures from yfinance.

    Returns DataFrame with [date, m1_close, m2_close=None, spread=None].
    """
    import yfinance as yf

    meta = get_commodity_meta(commodity)
    ticker = meta["futures"]["yfinance"]

    try:
        data = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
    except Exception as e:
        logger.warning("yfinance fetch failed for %s (%s): %s", commodity, ticker, e)
        return None

    if data.empty:
        logger.warning("yfinance returned empty data for %s", commodity)
        return None

    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    m1_values = data["Close"].values.astype(float)
    m2_values, spreads = _generate_synthetic_m2(commodity, m1_values)

    df = pd.DataFrame({
        "date": data.index.date.astype(str),
        "m1_close": m1_values,
        "m2_close": m2_values,
        "spread": spreads,
    })
    df = df.dropna(subset=["m1_close"])

    logger.info(
        "yfinance: fetched %d rows for %s (M1 real, M2 synthetic)",
        len(df), commodity,
    )
    return df


def _generate_synthetic_m2(
    commodity: str,
    m1_prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic M2 prices from real M1 using mean-reverting carry model.

    The carry (M2-M1 as % of M1) follows an Ornstein-Uhlenbeck process,
    producing realistic contango/backwardation dynamics.
    Deterministically seeded for reproducibility.
    """
    params = CARRY_PARAMS.get(commodity, {"mean_carry_pct": 0.003, "vol": 0.005, "mr": 0.02})
    seed = sum(ord(c) for c in commodity) + 200
    rng = np.random.RandomState(seed)

    mu = params["mean_carry_pct"]
    sigma = params["vol"]
    kappa = params["mr"]

    n = len(m1_prices)
    carry = np.zeros(n)
    carry[0] = mu + rng.randn() * sigma * 0.5

    for i in range(1, n):
        dW = rng.randn()
        carry[i] = carry[i - 1] + kappa * (mu - carry[i - 1]) + sigma * dW * 0.1

    m2_prices = m1_prices * (1 + carry)
    spreads = m1_prices - m2_prices  # positive = backwardation

    return m2_prices, spreads


def backfill_futures(conn: sqlite3.Connection, commodity: str) -> int:
    """Fetch futures data, trying CHRIS first then yfinance fallback.

    Returns number of rows upserted.
    """
    df = fetch_futures_chris(commodity)
    source = "chris"

    if df is None or df.empty:
        logger.info("Falling back to yfinance for %s", commodity)
        df = fetch_futures_yfinance(commodity)
        source = "yfinance"

    if df is None or df.empty:
        logger.warning("No futures data available for %s", commodity)
        return 0

    rows = [
        (
            row["date"],
            commodity,
            float(row["m1_close"]),
            float(row["m2_close"]) if pd.notna(row.get("m2_close")) else None,
            float(row["spread"]) if pd.notna(row.get("spread")) else None,
            source,
        )
        for _, row in df.iterrows()
    ]

    upsert_futures_batch(conn, rows)
    logger.info("Upserted %d futures rows for %s from %s", len(rows), commodity, source)
    return len(rows)
