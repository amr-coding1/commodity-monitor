"""Cross-commodity comparison — z-score matrices, correlations, sensitivity."""
from __future__ import annotations

import logging

import pandas as pd

from src.analysis.storage_economics import (
    StorageEconomicsAnalyser,
    pearson_with_ci,
)
from src.database import get_analytics_series, get_connection, list_commodities

logger = logging.getLogger(__name__)


def build_zscore_matrix(conn=None) -> pd.DataFrame:
    """Build a wide-format DataFrame of inventory z-scores across commodities.

    Returns DataFrame with date index and commodity columns.
    """
    conn = conn or get_connection()
    frames = {}
    for commodity in list_commodities():
        df = get_analytics_series(conn, commodity)
        if not df.empty and "zscore_1y" in df.columns:
            frames[commodity] = df["zscore_1y"].astype(float)

    if not frames:
        return pd.DataFrame()

    matrix = pd.DataFrame(frames)
    matrix.index.name = "date"
    return matrix


def compute_cross_correlation_matrix(conn=None) -> pd.DataFrame:
    """Compute pairwise Pearson correlations of inventory z-scores.

    Returns a square DataFrame with commodity names as both index and columns.
    """
    matrix = build_zscore_matrix(conn)
    if matrix.empty:
        return pd.DataFrame()

    commodities = list(matrix.columns)
    n = len(commodities)
    corr_data = pd.DataFrame(
        index=commodities, columns=commodities, dtype=float
    )

    for i in range(n):
        corr_data.iloc[i, i] = 1.0
        for j in range(i + 1, n):
            r, p, _, _ = pearson_with_ci(
                matrix[commodities[i]], matrix[commodities[j]]
            )
            corr_data.iloc[i, j] = r
            corr_data.iloc[j, i] = r

    return corr_data


def compute_sensitivity_comparison(conn=None) -> pd.DataFrame:
    """Compare inventory-spread sensitivity (OLS beta) across commodities.

    Returns DataFrame with columns [commodity, beta, r_squared, p_value, n].
    """
    conn = conn or get_connection()
    analyser = StorageEconomicsAnalyser(conn)
    rows = []

    for commodity in list_commodities():
        result = analyser.analyse_commodity(commodity)
        if result.sensitivity:
            s = result.sensitivity
            rows.append({
                "commodity": commodity,
                "beta": s.beta,
                "r_squared": s.r_squared,
                "p_value": s.p_value,
                "n": s.n,
            })
        else:
            rows.append({
                "commodity": commodity,
                "beta": None,
                "r_squared": None,
                "p_value": None,
                "n": 0,
            })

    return pd.DataFrame(rows)


def compute_regime_summary(conn=None) -> pd.DataFrame:
    """Summarise current regime state across all commodities.

    Returns DataFrame with [commodity, zscore_1y, regime, signal_alignment,
    stocks_to_use, spread_pct].
    """
    conn = conn or get_connection()
    rows = []

    for commodity in list_commodities():
        df = get_analytics_series(conn, commodity)
        if df.empty:
            continue

        latest = df.iloc[-1]
        rows.append({
            "commodity": commodity,
            "zscore_1y": latest.get("zscore_1y"),
            "regime": latest.get("regime"),
            "signal_alignment": latest.get("signal_alignment"),
            "stocks_to_use": latest.get("stocks_to_use"),
            "spread_pct": latest.get("spread_pct"),
        })

    return pd.DataFrame(rows)
