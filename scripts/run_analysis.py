"""Run processing pipeline and analysis, generate charts and commentary."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection, init_db, list_commodities
from src.processing.normaliser import process_inventory_analytics
from src.processing.spreads import process_spread_analytics
from src.analysis.storage_economics import StorageEconomicsAnalyser
import matplotlib.pyplot as plt

from src.reporting.charts import (
    plot_correlation_matrix,
    plot_inventory_spread_dual_axis,
    plot_sensitivity_bar,
    plot_stock_spread_scatter,
    plot_tightness_heatmap,
    plot_zscore_overlay,
)
from src.reporting.commentary import generate_commentary
from src.settings import CHARTS_DIR, ROOT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    conn = get_connection()
    init_db(conn)

    commodities = list_commodities()

    # ── Step 1: Processing ───────────────────────────────────────────────
    logger.info("=== Processing ===")
    for commodity in commodities:
        logger.info("Processing %s", commodity)
        process_inventory_analytics(conn, commodity)
        process_spread_analytics(conn, commodity)

    # ── Step 2: Analysis ─────────────────────────────────────────────────
    logger.info("=== Analysis ===")
    analyser = StorageEconomicsAnalyser(conn)
    results = analyser.run_all()

    for commodity, result in results.items():
        rd = result.to_dict()
        corrs = rd.get("correlations", [])
        if corrs:
            best = max(corrs, key=lambda c: abs(c["r"]))
            logger.info(
                "  %s: best lag=%dw, r=%.3f (p=%.4f)",
                commodity, best["lag_weeks"], best["r"], best["p_value"],
            )
        if result.walk_forward:
            wf = result.walk_forward
            logger.info(
                "  %s: walk-forward IS=%.3f, OOS=%.3f, stable=%s",
                commodity, wf.in_sample_r, wf.out_of_sample_r, wf.stable,
            )

    # ── Step 3: Charts ───────────────────────────────────────────────────
    logger.info("=== Generating Charts ===")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for commodity in commodities:
        fig = plot_inventory_spread_dual_axis(
            conn, commodity, save_path=CHARTS_DIR / f"{commodity}_dual_axis.png"
        )
        plt.close(fig)
        fig = plot_stock_spread_scatter(
            conn, commodity, save_path=CHARTS_DIR / f"{commodity}_scatter.png"
        )
        plt.close(fig)

    for chart_fn, name in [
        (plot_tightness_heatmap, "tightness_heatmap.png"),
        (plot_sensitivity_bar, "sensitivity_bar.png"),
        (plot_correlation_matrix, "correlation_matrix.png"),
        (plot_zscore_overlay, "zscore_overlay.png"),
    ]:
        fig = chart_fn(conn, save_path=CHARTS_DIR / name)
        plt.close(fig)

    logger.info("Charts saved to %s", CHARTS_DIR)

    # ── Step 4: Commentary ───────────────────────────────────────────────
    logger.info("=== Generating Commentary ===")
    report_path = ROOT_DIR / "reports" / "commentary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_commentary(conn, output_path=report_path)

    conn.close()
    logger.info("Analysis pipeline complete")


if __name__ == "__main__":
    main()
