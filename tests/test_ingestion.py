"""Tests for data ingestion — URL builders, Excel parsing, DB writes."""
from __future__ import annotations

from datetime import date

import pytest

from src.ingestion.ice_stocks import _build_ice_url


class TestBuildIceUrl:
    """Test ICE certified stock URL builder."""

    def test_coffee_url(self):
        """Coffee URL should use coffee_c report name."""
        url = _build_ice_url("coffee", date(2024, 3, 15))
        assert "coffee_c" in url
        assert "20240315" in url
        assert url.startswith("https://www.ice.com/publicdocs/")

    def test_cocoa_url(self):
        """Cocoa URL should use cocoa report name."""
        url = _build_ice_url("cocoa", date(2024, 6, 1))
        assert "cocoa" in url
        assert "20240601" in url

    def test_sugar_url(self):
        """Sugar URL should use sugar_no_11 report name."""
        url = _build_ice_url("sugar", date(2024, 12, 31))
        assert "sugar_no_11" in url
        assert "20241231" in url

    def test_url_format(self):
        """URL should follow the expected pattern."""
        url = _build_ice_url("coffee", date(2024, 1, 2))
        expected = (
            "https://www.ice.com/publicdocs/futures_us_reports"
            "/coffee_c/coffee_c_cert_stock_20240102.xls"
        )
        assert url == expected


class TestDatabaseOperations:
    """Test database CRUD operations with in-memory DB."""

    def test_upsert_inventory(self, db):
        """Upsert should insert new rows."""
        from src.database import upsert_inventory

        upsert_inventory(db, "2024-01-02", "copper", "LME", 200000.0, "tonnes")
        db.commit()

        row = db.execute(
            "SELECT * FROM inventory WHERE commodity = 'copper'"
        ).fetchone()
        assert row is not None
        assert row["stock_level"] == 200000.0

    def test_upsert_inventory_update(self, db):
        """Upsert should update existing rows."""
        from src.database import upsert_inventory

        upsert_inventory(db, "2024-01-02", "copper", "LME", 200000.0, "tonnes")
        upsert_inventory(db, "2024-01-02", "copper", "LME", 210000.0, "tonnes")
        db.commit()

        rows = db.execute(
            "SELECT * FROM inventory WHERE commodity = 'copper' AND date = '2024-01-02'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["stock_level"] == 210000.0

    def test_upsert_futures(self, db):
        """Upsert futures with M1, M2, and spread."""
        from src.database import upsert_futures_price

        upsert_futures_price(
            db, "2024-01-02", "copper", 8500.0, 8520.0, -20.0, "chris"
        )
        db.commit()

        row = db.execute(
            "SELECT * FROM futures_prices WHERE commodity = 'copper'"
        ).fetchone()
        assert row["m1_close"] == 8500.0
        assert row["m2_close"] == 8520.0
        assert row["spread"] == -20.0

    def test_upsert_futures_yfinance_only(self, db):
        """Upsert futures with M1 only (yfinance fallback)."""
        from src.database import upsert_futures_price

        upsert_futures_price(
            db, "2024-01-02", "coffee", 180.0, None, None, "yfinance"
        )
        db.commit()

        row = db.execute(
            "SELECT * FROM futures_prices WHERE commodity = 'coffee'"
        ).fetchone()
        assert row["m1_close"] == 180.0
        assert row["m2_close"] is None
        assert row["spread"] is None

    def test_upsert_consumption(self, db):
        """Upsert annual consumption data."""
        from src.database import upsert_consumption

        upsert_consumption(db, 2024, "copper", 26000000.0, "tonnes", "usgs")

        row = db.execute(
            "SELECT * FROM consumption WHERE commodity = 'copper' AND year = 2024"
        ).fetchone()
        assert row["annual_consumption"] == 26000000.0

    def test_upsert_daily_analytics_partial(self, db):
        """Partial upsert should preserve existing non-null fields."""
        from src.database import upsert_daily_analytics

        # First write: z-scores
        upsert_daily_analytics(
            db, "2024-01-02", "copper", zscore_1y=-1.5, zscore_3y=-0.8
        )
        db.commit()

        # Second write: spread (should not overwrite z-scores)
        upsert_daily_analytics(
            db, "2024-01-02", "copper", spread=-20.0, regime="contango"
        )
        db.commit()

        row = db.execute(
            "SELECT * FROM daily_analytics WHERE commodity = 'copper' AND date = '2024-01-02'"
        ).fetchone()
        assert row["zscore_1y"] == -1.5  # preserved
        assert row["zscore_3y"] == -0.8  # preserved
        assert row["spread"] == -20.0    # added
        assert row["regime"] == "contango"  # added

    def test_inventory_batch(self, db):
        """Batch upsert should insert multiple rows."""
        from src.database import upsert_inventory_batch

        rows = [
            ("2024-01-02", "copper", "LME", 200000.0, "tonnes"),
            ("2024-01-03", "copper", "LME", 199000.0, "tonnes"),
            ("2024-01-04", "copper", "LME", 198000.0, "tonnes"),
        ]
        upsert_inventory_batch(db, rows)

        count = db.execute(
            "SELECT COUNT(*) FROM inventory WHERE commodity = 'copper'"
        ).fetchone()[0]
        assert count == 3

    def test_futures_batch(self, db):
        """Batch upsert futures should insert multiple rows."""
        from src.database import upsert_futures_batch

        rows = [
            ("2024-01-02", "copper", 8500.0, 8520.0, -20.0, "chris"),
            ("2024-01-03", "copper", 8510.0, 8530.0, -20.0, "chris"),
        ]
        upsert_futures_batch(db, rows)

        count = db.execute(
            "SELECT COUNT(*) FROM futures_prices WHERE commodity = 'copper'"
        ).fetchone()[0]
        assert count == 2
