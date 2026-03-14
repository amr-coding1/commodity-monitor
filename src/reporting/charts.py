"""Chart generation — matplotlib figures for analysis output."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.database import (
    get_analytics_series,
    get_connection,
    get_futures_series,
    get_inventory_series,
    list_commodities,
)
from src.analysis.cross_commodity import (
    build_zscore_matrix,
    compute_cross_correlation_matrix,
    compute_sensitivity_comparison,
)

logger = logging.getLogger(__name__)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
COLOURS = {
    "copper": "#B87333",
    "aluminium": "#A8A9AD",
    "coffee": "#6F4E37",
    "cocoa": "#3B1F0B",
    "sugar": "#F5F5DC",
}
REGIME_COLOURS = {
    "backwardation": "#d32f2f",
    "contango": "#1976d2",
    "flat": "#9e9e9e",
    "unknown": "#e0e0e0",
}


def plot_inventory_spread_dual_axis(
    conn,
    commodity: str,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Dual-axis chart: inventory z-score (left) vs calendar spread % (right)."""
    analytics = get_analytics_series(conn, commodity)
    if analytics.empty:
        fig, ax = plt.subplots()
        ax.set_title(f"{commodity} — no data")
        return fig

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: z-score
    z = analytics["zscore_1y"].astype(float)
    ax1.plot(analytics.index, z, color="#1976d2", linewidth=1.2, label="Inventory Z-Score (1Y)")
    ax1.axhline(y=-1.0, color="#d32f2f", linestyle="--", alpha=0.5, label="Tight threshold")
    ax1.axhline(y=1.0, color="#2e7d32", linestyle="--", alpha=0.5, label="Surplus threshold")
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax1.set_ylabel("Inventory Z-Score", color="#1976d2")
    ax1.tick_params(axis="y", labelcolor="#1976d2")

    # Right axis: spread %
    ax2 = ax1.twinx()
    spread_pct = analytics["spread_pct"].astype(float) * 100
    ax2.plot(analytics.index, spread_pct, color="#d32f2f", linewidth=1.2, alpha=0.7, label="Calendar Spread %")
    ax2.axhline(y=0, color="#d32f2f", linestyle="-", alpha=0.2)
    ax2.set_ylabel("M1−M2 Spread (%)", color="#d32f2f")
    ax2.tick_params(axis="y", labelcolor="#d32f2f")

    ax1.set_title(f"{commodity.title()} — Inventory Z-Score vs Calendar Spread")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved dual-axis chart to %s", save_path)
    return fig


def plot_stock_spread_scatter(
    conn,
    commodity: str,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Scatter plot: inventory z-score vs spread %, coloured by regime."""
    analytics = get_analytics_series(conn, commodity)
    if analytics.empty:
        fig, ax = plt.subplots()
        ax.set_title(f"{commodity} — no data")
        return fig

    fig, ax = plt.subplots(figsize=(8, 6))

    for regime, colour in REGIME_COLOURS.items():
        mask = analytics["regime"] == regime
        if mask.any():
            subset = analytics[mask]
            ax.scatter(
                subset["zscore_1y"].astype(float),
                subset["spread_pct"].astype(float) * 100,
                c=colour,
                label=regime.title(),
                alpha=0.6,
                s=15,
                edgecolors="none",
            )

    ax.axvline(x=-1.0, color="#d32f2f", linestyle="--", alpha=0.4)
    ax.axvline(x=1.0, color="#2e7d32", linestyle="--", alpha=0.4)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("Inventory Z-Score (1Y)")
    ax.set_ylabel("Calendar Spread (%)")
    ax.set_title(f"{commodity.title()} — Stock Level vs Spread")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_tightness_heatmap(
    conn,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Heatmap of inventory z-scores over time for all commodities."""
    matrix = build_zscore_matrix(conn)
    if matrix.empty:
        fig, ax = plt.subplots()
        ax.set_title("No data")
        return fig

    # Resample to weekly for cleaner heatmap
    weekly = matrix.resample("W").last()

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        weekly.T,
        ax=ax,
        cmap="RdYlGn_r",
        center=0,
        vmin=-3,
        vmax=3,
        xticklabels=False,
        cbar_kws={"label": "Inventory Z-Score"},
    )

    # Add date labels at intervals
    n_labels = 12
    step = max(1, len(weekly) // n_labels)
    tick_positions = list(range(0, len(weekly), step))
    tick_labels = [weekly.index[i].strftime("%b %Y") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_title("Cross-Commodity Inventory Tightness")
    ax.set_ylabel("")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sensitivity_bar(
    conn,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Bar chart comparing inventory-spread sensitivity (beta) across commodities."""
    df = compute_sensitivity_comparison(conn)
    df = df.dropna(subset=["beta"])

    if df.empty:
        fig, ax = plt.subplots()
        ax.set_title("No sensitivity data")
        return fig

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = [COLOURS.get(c, "#666666") for c in df["commodity"]]
    bars = ax.bar(df["commodity"].str.title(), df["beta"], color=colours, edgecolor="black", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Sensitivity (β: spread % per z-score unit)")
    ax.set_title("Inventory→Spread Sensitivity by Commodity")

    # Annotate with R²
    for bar, (_, row) in zip(bars, df.iterrows()):
        r2 = row["r_squared"]
        if r2 is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"R²={r2:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_matrix(
    conn,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Heatmap of cross-commodity z-score correlations."""
    corr = compute_cross_correlation_matrix(conn)
    if corr.empty:
        fig, ax = plt.subplots()
        ax.set_title("No data")
        return fig

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr.astype(float),
        ax=ax,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=[c.title() for c in corr.columns],
        yticklabels=[c.title() for c in corr.index],
    )
    ax.set_title("Cross-Commodity Inventory Z-Score Correlations")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_zscore_overlay(
    conn,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Overlay time series of normalised z-scores for all commodities."""
    matrix = build_zscore_matrix(conn)
    if matrix.empty:
        fig, ax = plt.subplots()
        ax.set_title("No data")
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))
    for commodity in matrix.columns:
        colour = COLOURS.get(commodity, "#666666")
        ax.plot(
            matrix.index,
            matrix[commodity],
            label=commodity.title(),
            color=colour,
            linewidth=1.2,
        )

    ax.axhline(y=-1.0, color="#d32f2f", linestyle="--", alpha=0.4, label="Tight")
    ax.axhline(y=1.0, color="#2e7d32", linestyle="--", alpha=0.4, label="Surplus")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax.set_ylabel("Inventory Z-Score (1Y)")
    ax.set_title("Normalised Inventory Levels — All Commodities")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
