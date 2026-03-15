"""Railway startup — initialise DB, ingest data, and run processing pipeline.

Runs once before the Streamlit dashboard launches.  Skips ingestion if the
database already contains data (to avoid re-fetching on every restart).
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection, init_db, list_commodities, get_commodity_meta
from src.ingestion.lme_stocks import backfill_lme_stocks
from src.ingestion.ice_stocks import backfill_ice_stocks
from src.ingestion.futures import backfill_futures
from src.ingestion.consumption import backfill_consumption
from src.processing.normaliser import process_inventory_analytics
from src.processing.spreads import process_spread_analytics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def _db_has_data(conn) -> bool:
    """Check if the database already has inventory data."""
    row = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()
    return row[0] > 0


def main() -> None:
    conn = get_connection()
    init_db(conn)

    if _db_has_data(conn):
        logger.info("Database already populated — skipping ingestion")
    else:
        logger.info("Empty database — running full ingestion")
        commodities = list_commodities()
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * 3)

        for commodity in commodities:
            meta = get_commodity_meta(commodity)
            exchange = meta.get("exchange", "")
            logger.info("Ingesting %s (%s)", meta["name"], exchange)

            if exchange == "LME":
                backfill_lme_stocks(conn, commodity, start_date, end_date)
            elif exchange == "ICE":
                backfill_ice_stocks(conn, commodity, start_date, end_date)

            backfill_futures(conn, commodity)
            backfill_consumption(conn, commodity)

    # Always run processing (fast, ensures analytics are up to date)
    logger.info("Running processing pipeline")
    commodities = list_commodities()
    for commodity in commodities:
        process_inventory_analytics(conn, commodity)
        process_spread_analytics(conn, commodity)

    conn.close()
    logger.info("Startup complete — ready for dashboard")


if __name__ == "__main__":
    main()
