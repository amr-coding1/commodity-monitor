"""Inventory normalisation — z-scores and stocks-to-use ratios."""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from src.database import (
    get_connection,
    get_commodity_meta,
    get_consumption_for_commodity,
    get_inventory_series,
    upsert_daily_analytics,
)
from src.settings import ZSCORE_WINDOW_1Y, ZSCORE_WINDOW_3Y

logger = logging.getLogger(__name__)


def compute_inventory_zscore(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling z-score of an inventory series.

    z = (x - rolling_mean) / rolling_std

    Args:
        series: Inventory levels indexed by date.
        window: Rolling window in trading days.

    Returns:
        Z-score series, NaN where insufficient history.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def compute_stocks_to_use_days(
    stock_level: float,
    annual_consumption: float,
) -> float | None:
    """Convert stock level to days-of-consumption coverage.

    Args:
        stock_level: Current inventory in commodity units.
        annual_consumption: Annual consumption in same units.

    Returns:
        Days of supply coverage, or None if consumption is zero/missing.
    """
    if annual_consumption <= 0:
        return None
    daily_consumption = annual_consumption / 365.0
    return stock_level / daily_consumption


def process_inventory_analytics(conn: sqlite3.Connection, commodity: str) -> int:
    """Compute z-scores and stocks-to-use for a commodity, write to daily_analytics.

    Returns number of rows written.
    """
    inv_df = get_inventory_series(conn, commodity)
    if inv_df.empty:
        logger.warning("No inventory data for %s", commodity)
        return 0

    stock_col = "stock_level"
    series = inv_df[stock_col].astype(float)

    # Z-scores
    z1y = compute_inventory_zscore(series, ZSCORE_WINDOW_1Y)
    z3y = compute_inventory_zscore(series, ZSCORE_WINDOW_3Y)

    # Stocks-to-use: use most recent consumption year
    cons_df = get_consumption_for_commodity(conn, commodity)
    annual_consumption = None
    if not cons_df.empty:
        annual_consumption = cons_df.sort_values("year", ascending=False).iloc[0][
            "annual_consumption"
        ]

    count = 0
    for dt in inv_df.index:
        dt_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
        stock = series.loc[dt]

        stu = None
        if annual_consumption and annual_consumption > 0:
            stu = compute_stocks_to_use_days(stock, annual_consumption)

        z1 = z1y.get(dt)
        z3 = z3y.get(dt)

        upsert_daily_analytics(
            conn,
            dt_str,
            commodity,
            zscore_1y=float(z1) if pd.notna(z1) else None,
            zscore_3y=float(z3) if pd.notna(z3) else None,
            stocks_to_use=stu,
        )
        count += 1

    conn.commit()
    logger.info("Wrote %d inventory analytics rows for %s", count, commodity)
    return count
