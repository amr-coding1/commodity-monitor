"""Shared test fixtures — in-memory database with schema and seed data."""
from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta

import pytest

from src.database import SCHEMA_SQL, init_db


@pytest.fixture
def db():
    """In-memory SQLite database with schema initialised."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


@pytest.fixture
def seeded_db(db):
    """Database with realistic seed data for all 5 commodities."""
    commodities = [
        ("copper", "LME", "tonnes"),
        ("aluminium", "LME", "tonnes"),
        ("coffee", "ICE", "bags"),
        ("cocoa", "ICE", "tonnes"),
        ("sugar", "ICE", "short_tons"),
    ]
    base_date = date(2024, 1, 2)

    for commodity, exchange, unit in commodities:
        # Inventory: 200 days of data with a trend
        base_stock = {
            "copper": 200000,
            "aluminium": 500000,
            "coffee": 3000000,
            "cocoa": 250000,
            "sugar": 400000,
        }[commodity]

        for i in range(200):
            dt = base_date + timedelta(days=i)
            if dt.weekday() >= 5:
                continue
            # Add sinusoidal variation to simulate real inventory cycles
            variation = math.sin(i / 40 * math.pi) * base_stock * 0.15
            stock = base_stock + variation + (i * base_stock * 0.0005)

            db.execute(
                "INSERT OR REPLACE INTO inventory VALUES (?, ?, ?, ?, ?)",
                (str(dt), commodity, exchange, stock, unit),
            )

        # Futures: 200 days of M1 and M2
        base_price = {
            "copper": 8500,
            "aluminium": 2300,
            "coffee": 180,
            "cocoa": 5000,
            "sugar": 22,
        }[commodity]

        for i in range(200):
            dt = base_date + timedelta(days=i)
            if dt.weekday() >= 5:
                continue
            import math
            m1 = base_price + math.sin(i / 30 * math.pi) * base_price * 0.05
            # M2 slightly higher (contango) with periodic flips to backwardation
            contango_flip = -1 if math.sin(i / 60 * math.pi) > 0.5 else 1
            m2 = m1 + contango_flip * base_price * 0.01
            spread = m1 - m2

            db.execute(
                "INSERT OR REPLACE INTO futures_prices VALUES (?, ?, ?, ?, ?, ?)",
                (str(dt), commodity, m1, m2, spread, "test"),
            )

        # Consumption: annual data
        annual = {
            "copper": 26000000,
            "aluminium": 69000000,
            "coffee": 170000000,
            "cocoa": 5000000,
            "sugar": 180000000,
        }[commodity]

        for year in [2022, 2023, 2024]:
            db.execute(
                "INSERT OR REPLACE INTO consumption VALUES (?, ?, ?, ?, ?)",
                (year, commodity, annual * (1 + (year - 2022) * 0.02), unit, "test"),
            )

    db.commit()
    return db
