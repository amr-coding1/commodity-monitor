"""SQLite database — schema, connection management, CRUD helpers."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.settings import CONFIG_DIR, DB_PATH

logger = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inventory (
    date        TEXT    NOT NULL,
    commodity   TEXT    NOT NULL,
    exchange    TEXT    NOT NULL,
    stock_level REAL    NOT NULL,
    unit        TEXT    NOT NULL,
    PRIMARY KEY (date, commodity)
);

CREATE TABLE IF NOT EXISTS futures_prices (
    date        TEXT    NOT NULL,
    commodity   TEXT    NOT NULL,
    m1_close    REAL    NOT NULL,
    m2_close    REAL,
    spread      REAL,
    source      TEXT    NOT NULL,
    PRIMARY KEY (date, commodity)
);

CREATE TABLE IF NOT EXISTS consumption (
    year        INTEGER NOT NULL,
    commodity   TEXT    NOT NULL,
    annual_consumption REAL NOT NULL,
    unit        TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    PRIMARY KEY (year, commodity)
);

CREATE TABLE IF NOT EXISTS daily_analytics (
    date            TEXT    NOT NULL,
    commodity       TEXT    NOT NULL,
    zscore_1y       REAL,
    zscore_3y       REAL,
    stocks_to_use   REAL,
    spread          REAL,
    spread_pct      REAL,
    regime          TEXT,
    signal_alignment TEXT,
    PRIMARY KEY (date, commodity)
);
"""

# ── Config loader (lazy, module-level cache) ─────────────────────────────
_config_cache: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    """Load and cache config/commodities.json."""
    global _config_cache
    if _config_cache is None:
        path = CONFIG_DIR / "commodities.json"
        with open(path) as f:
            _config_cache = json.load(f)
        logger.info("Loaded config from %s", path)
    return _config_cache


def get_commodity_meta(commodity: str) -> dict[str, Any]:
    """Return metadata dict for a single commodity key."""
    cfg = load_config()
    return cfg["commodities"][commodity]


def get_regime_thresholds() -> dict[str, float]:
    """Return regime classification thresholds."""
    cfg = load_config()
    return cfg["regime_thresholds"]


def list_commodities() -> list[str]:
    """Return sorted list of commodity keys."""
    cfg = load_config()
    return sorted(cfg["commodities"].keys())


# ── Connection ───────────────────────────────────────────────────────────
def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a WAL-mode connection with Row factory."""
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    logger.info("Database schema initialised")


# ── CRUD: Inventory ─────────────────────────────────────────────────────
def upsert_inventory(
    conn: sqlite3.Connection,
    dt: date | str,
    commodity: str,
    exchange: str,
    stock_level: float,
    unit: str,
) -> None:
    """Insert or update a single inventory row."""
    conn.execute(
        """
        INSERT INTO inventory (date, commodity, exchange, stock_level, unit)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date, commodity) DO UPDATE SET
            exchange    = excluded.exchange,
            stock_level = excluded.stock_level,
            unit        = excluded.unit
        """,
        (str(dt), commodity, exchange, stock_level, unit),
    )


def upsert_inventory_batch(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, str, float, str]],
) -> None:
    """Batch upsert inventory rows: (date, commodity, exchange, stock_level, unit)."""
    conn.executemany(
        """
        INSERT INTO inventory (date, commodity, exchange, stock_level, unit)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date, commodity) DO UPDATE SET
            exchange    = excluded.exchange,
            stock_level = excluded.stock_level,
            unit        = excluded.unit
        """,
        rows,
    )
    conn.commit()


# ── CRUD: Futures ────────────────────────────────────────────────────────
def upsert_futures_price(
    conn: sqlite3.Connection,
    dt: date | str,
    commodity: str,
    m1_close: float,
    m2_close: float | None,
    spread: float | None,
    source: str,
) -> None:
    """Insert or update a single futures price row."""
    conn.execute(
        """
        INSERT INTO futures_prices (date, commodity, m1_close, m2_close, spread, source)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, commodity) DO UPDATE SET
            m1_close = excluded.m1_close,
            m2_close = excluded.m2_close,
            spread   = excluded.spread,
            source   = excluded.source
        """,
        (str(dt), commodity, m1_close, m2_close, spread, source),
    )


def upsert_futures_batch(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, float, float | None, float | None, str]],
) -> None:
    """Batch upsert futures rows."""
    conn.executemany(
        """
        INSERT INTO futures_prices (date, commodity, m1_close, m2_close, spread, source)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, commodity) DO UPDATE SET
            m1_close = excluded.m1_close,
            m2_close = excluded.m2_close,
            spread   = excluded.spread,
            source   = excluded.source
        """,
        rows,
    )
    conn.commit()


# ── CRUD: Consumption ───────────────────────────────────────────────────
def upsert_consumption(
    conn: sqlite3.Connection,
    year: int,
    commodity: str,
    annual_consumption: float,
    unit: str,
    source: str,
) -> None:
    """Insert or update a single consumption row."""
    conn.execute(
        """
        INSERT INTO consumption (year, commodity, annual_consumption, unit, source)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(year, commodity) DO UPDATE SET
            annual_consumption = excluded.annual_consumption,
            unit               = excluded.unit,
            source             = excluded.source
        """,
        (year, commodity, annual_consumption, unit, source),
    )
    conn.commit()


# ── CRUD: Daily Analytics ───────────────────────────────────────────────
def upsert_daily_analytics(
    conn: sqlite3.Connection,
    dt: date | str,
    commodity: str,
    *,
    zscore_1y: float | None = None,
    zscore_3y: float | None = None,
    stocks_to_use: float | None = None,
    spread: float | None = None,
    spread_pct: float | None = None,
    regime: str | None = None,
    signal_alignment: str | None = None,
) -> None:
    """Upsert a daily analytics row, preserving existing non-null fields."""
    conn.execute(
        """
        INSERT INTO daily_analytics
            (date, commodity, zscore_1y, zscore_3y, stocks_to_use,
             spread, spread_pct, regime, signal_alignment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, commodity) DO UPDATE SET
            zscore_1y       = COALESCE(excluded.zscore_1y,       daily_analytics.zscore_1y),
            zscore_3y       = COALESCE(excluded.zscore_3y,       daily_analytics.zscore_3y),
            stocks_to_use   = COALESCE(excluded.stocks_to_use,   daily_analytics.stocks_to_use),
            spread          = COALESCE(excluded.spread,          daily_analytics.spread),
            spread_pct      = COALESCE(excluded.spread_pct,      daily_analytics.spread_pct),
            regime          = COALESCE(excluded.regime,          daily_analytics.regime),
            signal_alignment = COALESCE(excluded.signal_alignment, daily_analytics.signal_alignment)
        """,
        (str(dt), commodity, zscore_1y, zscore_3y, stocks_to_use,
         spread, spread_pct, regime, signal_alignment),
    )


def upsert_daily_analytics_batch(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
) -> None:
    """Batch upsert daily analytics from list of dicts."""
    for row in rows:
        upsert_daily_analytics(conn, **row)
    conn.commit()


# ── Query helpers ────────────────────────────────────────────────────────
def get_inventory_series(
    conn: sqlite3.Connection,
    commodity: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Return inventory time series as a DataFrame with date index."""
    query = "SELECT date, stock_level FROM inventory WHERE commodity = ?"
    params: list[Any] = [commodity]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    if not df.empty:
        df.set_index("date", inplace=True)
    return df


def get_futures_series(
    conn: sqlite3.Connection,
    commodity: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Return futures prices as a DataFrame with date index."""
    query = (
        "SELECT date, m1_close, m2_close, spread "
        "FROM futures_prices WHERE commodity = ?"
    )
    params: list[Any] = [commodity]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    if not df.empty:
        df.set_index("date", inplace=True)
    return df


def get_latest_analytics(
    conn: sqlite3.Connection,
    commodity: str | None = None,
) -> pd.DataFrame:
    """Return the most recent analytics row per commodity."""
    if commodity:
        query = """
            SELECT * FROM daily_analytics
            WHERE commodity = ?
            ORDER BY date DESC LIMIT 1
        """
        return pd.read_sql_query(query, conn, params=[commodity])
    query = """
        SELECT da.* FROM daily_analytics da
        INNER JOIN (
            SELECT commodity, MAX(date) as max_date
            FROM daily_analytics GROUP BY commodity
        ) latest ON da.commodity = latest.commodity AND da.date = latest.max_date
        ORDER BY da.commodity
    """
    return pd.read_sql_query(query, conn)


def get_analytics_series(
    conn: sqlite3.Connection,
    commodity: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Return daily analytics time series as a DataFrame."""
    query = "SELECT * FROM daily_analytics WHERE commodity = ?"
    params: list[Any] = [commodity]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    if not df.empty:
        df.set_index("date", inplace=True)
    return df


def get_consumption_for_commodity(
    conn: sqlite3.Connection,
    commodity: str,
    year: int | None = None,
) -> pd.DataFrame:
    """Return consumption data. If year given, return that year only."""
    if year is not None:
        query = "SELECT * FROM consumption WHERE commodity = ? AND year = ?"
        return pd.read_sql_query(query, conn, params=[commodity, year])
    query = "SELECT * FROM consumption WHERE commodity = ? ORDER BY year"
    return pd.read_sql_query(query, conn, params=[commodity])
