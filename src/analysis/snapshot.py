"""Current market state snapshot across all commodities."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from src.database import get_connection, get_latest_analytics, list_commodities

logger = logging.getLogger(__name__)


@dataclass
class CommoditySnapshot:
    """Current state for a single commodity."""

    commodity: str
    date: str | None = None
    zscore_1y: float | None = None
    zscore_3y: float | None = None
    stocks_to_use: float | None = None
    spread: float | None = None
    spread_pct: float | None = None
    regime: str | None = None
    signal_alignment: str | None = None
    overall_regime: str | None = None

    @staticmethod
    def _clean(val: float | None, decimals: int) -> float | None:
        """Round a value, converting NaN to None."""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return round(val, decimals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "commodity": self.commodity,
            "date": self.date,
            "zscore_1y": self._clean(self.zscore_1y, 2),
            "zscore_3y": self._clean(self.zscore_3y, 2),
            "stocks_to_use": self._clean(self.stocks_to_use, 1),
            "spread": self._clean(self.spread, 4),
            "spread_pct": self._clean(self.spread_pct, 4),
            "regime": self.regime,
            "signal_alignment": self.signal_alignment,
            "overall_regime": self.overall_regime,
        }

    @property
    def is_tight(self) -> bool:
        return self.overall_regime == "tight"

    @property
    def is_surplus(self) -> bool:
        return self.overall_regime == "surplus"

    @property
    def is_divergent(self) -> bool:
        return self.signal_alignment == "divergent"


def classify_overall_regime(
    zscore: float | None,
    regime: str | None,
) -> str:
    """Classify the overall market regime from z-score and spread regime.

    Returns:
        'tight': Low stocks and/or backwardation
        'surplus': High stocks and/or contango
        'balanced': Neither tight nor surplus
        'unknown': Insufficient data
    """
    if zscore is None and regime is None:
        return "unknown"

    # Primary signal: z-score
    z_signal = "neutral"
    if zscore is not None:
        if zscore < -1.0:
            z_signal = "tight"
        elif zscore > 1.0:
            z_signal = "surplus"

    # Secondary signal: spread regime
    s_signal = "neutral"
    if regime == "backwardation":
        s_signal = "tight"
    elif regime == "contango":
        s_signal = "surplus"

    # Combine
    if z_signal == "tight" or s_signal == "tight":
        if z_signal == "surplus" or s_signal == "surplus":
            return "balanced"  # conflicting signals
        return "tight"
    elif z_signal == "surplus" or s_signal == "surplus":
        return "surplus"
    else:
        return "balanced"


def get_market_snapshot(conn=None) -> list[CommoditySnapshot]:
    """Get current market state for all commodities.

    Returns list of CommoditySnapshot, one per commodity.
    """
    conn = conn or get_connection()
    df = get_latest_analytics(conn)
    snapshots = []

    if df.empty:
        for commodity in list_commodities():
            snapshots.append(CommoditySnapshot(commodity=commodity))
        return snapshots

    for _, row in df.iterrows():
        zscore = float(row["zscore_1y"]) if row["zscore_1y"] is not None else None
        regime = row["regime"] if row["regime"] is not None else None

        snapshot = CommoditySnapshot(
            commodity=row["commodity"],
            date=row["date"],
            zscore_1y=zscore,
            zscore_3y=float(row["zscore_3y"]) if row["zscore_3y"] is not None else None,
            stocks_to_use=float(row["stocks_to_use"]) if row["stocks_to_use"] is not None else None,
            spread=float(row["spread"]) if row["spread"] is not None else None,
            spread_pct=float(row["spread_pct"]) if row["spread_pct"] is not None else None,
            regime=regime,
            signal_alignment=row["signal_alignment"],
            overall_regime=classify_overall_regime(zscore, regime),
        )
        snapshots.append(snapshot)

    # Fill in any missing commodities
    seen = {s.commodity for s in snapshots}
    for commodity in list_commodities():
        if commodity not in seen:
            snapshots.append(CommoditySnapshot(commodity=commodity))

    return sorted(snapshots, key=lambda s: s.commodity)
