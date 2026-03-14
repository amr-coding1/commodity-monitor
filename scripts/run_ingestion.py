"""Run data ingestion for all configured commodities."""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection, init_db, list_commodities, get_commodity_meta
from src.ingestion.lme_stocks import backfill_lme_stocks
from src.ingestion.ice_stocks import backfill_ice_stocks
from src.ingestion.futures import backfill_futures
from src.ingestion.consumption import backfill_consumption

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    conn = get_connection()
    init_db(conn)

    commodities = list_commodities()
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 3)

    logger.info("Starting ingestion for %d commodities", len(commodities))
    logger.info("Date range: %s to %s", start_date, end_date)

    for commodity in commodities:
        meta = get_commodity_meta(commodity)
        exchange = meta.get("exchange", "")
        logger.info("── %s (%s) ──", meta["name"], exchange)

        # Inventory
        if exchange == "LME":
            n = backfill_lme_stocks(conn, commodity, start_date, end_date)
            logger.info("  Inventory: %d rows", n)
        elif exchange == "ICE":
            n = backfill_ice_stocks(conn, commodity, start_date, end_date)
            logger.info("  Inventory: %d rows", n)

        # Futures
        n = backfill_futures(conn, commodity)
        logger.info("  Futures: %d rows", n)

        # Consumption
        n = backfill_consumption(conn, commodity)
        logger.info("  Consumption: %d rows", n)

    conn.close()
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
