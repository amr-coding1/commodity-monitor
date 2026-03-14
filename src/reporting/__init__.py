"""Reporting — chart generation and auto-commentary."""
from src.reporting.charts import (
    plot_correlation_matrix,
    plot_inventory_spread_dual_axis,
    plot_sensitivity_bar,
    plot_stock_spread_scatter,
    plot_tightness_heatmap,
    plot_zscore_overlay,
)
from src.reporting.commentary import generate_commentary

__all__ = [
    "generate_commentary",
    "plot_correlation_matrix",
    "plot_inventory_spread_dual_axis",
    "plot_sensitivity_bar",
    "plot_stock_spread_scatter",
    "plot_tightness_heatmap",
    "plot_zscore_overlay",
]
