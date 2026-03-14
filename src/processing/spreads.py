"""Calendar spread computation and regime classification."""
from __future__ import annotations

import logging

import pandas as pd

from src.database import (
    get_futures_series,
    get_regime_thresholds,
    upsert_daily_analytics,
)

logger = logging.getLogger(__name__)


def classify_spread_regime(
    spread: float | None,
    m1: float | None,
    flat_threshold_pct: float = 0.002,
) -> str:
    """Classify the calendar spread regime.

    Args:
        spread: M1 - M2 price difference.
        m1: Front-month price (for percentage calculation).
        flat_threshold_pct: Spread as % of M1 below which regime is 'flat'.

    Returns:
        One of 'backwardation', 'contango', or 'flat'.
    """
    if spread is None or m1 is None or m1 == 0:
        return "unknown"

    spread_pct = spread / m1

    if abs(spread_pct) < flat_threshold_pct:
        return "flat"
    elif spread > 0:
        return "backwardation"
    else:
        return "contango"


def classify_signal_alignment(
    zscore: float | None,
    regime: str,
    zscore_tight: float = -1.0,
    zscore_surplus: float = 1.0,
) -> str:
    """Classify whether inventory z-score and spread regime agree.

    Aligned:
        - Low stocks (z < tight) + backwardation → tight market, signals agree
        - High stocks (z > surplus) + contango → surplus market, signals agree
    Divergent:
        - Low stocks + contango, or high stocks + backwardation
    Neutral:
        - Z-score between thresholds, or flat spread

    Returns:
        One of 'aligned', 'divergent', or 'neutral'.
    """
    if zscore is None or regime in ("unknown", "flat"):
        return "neutral"

    is_tight_stocks = zscore < zscore_tight
    is_surplus_stocks = zscore > zscore_surplus

    if is_tight_stocks and regime == "backwardation":
        return "aligned"
    elif is_surplus_stocks and regime == "contango":
        return "aligned"
    elif is_tight_stocks and regime == "contango":
        return "divergent"
    elif is_surplus_stocks and regime == "backwardation":
        return "divergent"
    else:
        return "neutral"


def process_spread_analytics(conn, commodity: str) -> int:
    """Compute spread metrics and regime for a commodity, write to daily_analytics.

    Reads futures prices and existing z-scores from daily_analytics
    to determine signal alignment.

    Returns number of rows written.
    """
    futures_df = get_futures_series(conn, commodity)
    if futures_df.empty:
        logger.warning("No futures data for %s", commodity)
        return 0

    thresholds = get_regime_thresholds()
    flat_pct = thresholds.get("flat_spread_pct", 0.002)
    z_tight = thresholds.get("zscore_tight", -1.0)
    z_surplus = thresholds.get("zscore_surplus", 1.0)

    # Load existing analytics to get z-scores for alignment check
    existing = pd.read_sql_query(
        "SELECT date, zscore_1y FROM daily_analytics WHERE commodity = ?",
        conn,
        params=[commodity],
    )
    zscore_lookup: dict[str, float | None] = {}
    if not existing.empty:
        for _, row in existing.iterrows():
            zscore_lookup[row["date"]] = (
                float(row["zscore_1y"]) if pd.notna(row["zscore_1y"]) else None
            )

    count = 0
    for dt in futures_df.index:
        dt_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
        row = futures_df.loc[dt]

        m1 = float(row["m1_close"]) if pd.notna(row["m1_close"]) else None
        m2 = float(row["m2_close"]) if pd.notna(row.get("m2_close")) else None
        spread = float(row["spread"]) if pd.notna(row.get("spread")) else None

        # If no M2/spread from data, can't classify
        if spread is None and m1 is not None and m2 is not None:
            spread = m1 - m2

        spread_pct = None
        if spread is not None and m1 and m1 != 0:
            spread_pct = spread / m1

        regime = classify_spread_regime(spread, m1, flat_pct)
        zscore = zscore_lookup.get(dt_str)
        alignment = classify_signal_alignment(zscore, regime, z_tight, z_surplus)

        upsert_daily_analytics(
            conn,
            dt_str,
            commodity,
            spread=spread,
            spread_pct=spread_pct,
            regime=regime,
            signal_alignment=alignment,
        )
        count += 1

    conn.commit()
    logger.info("Wrote %d spread analytics rows for %s", count, commodity)
    return count
