"""Auto-generated commentary using Jinja2 templates."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.analysis.cross_commodity import compute_sensitivity_comparison
from src.analysis.snapshot import get_market_snapshot
from src.database import get_connection
from src.settings import TEMPLATES_DIR

logger = logging.getLogger(__name__)


def generate_commentary(
    conn=None,
    output_path: Path | str | None = None,
) -> str:
    """Generate markdown commentary from current market state.

    Args:
        conn: Database connection (created if None).
        output_path: Optional path to write the report.

    Returns:
        Rendered markdown string.
    """
    conn = conn or get_connection()

    snapshots = get_market_snapshot(conn)
    sensitivity_df = compute_sensitivity_comparison(conn)
    sensitivity_rows = sensitivity_df.to_dict("records") if not sensitivity_df.empty else []

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("commentary.md.j2")

    rendered = template.render(
        generated_date=date.today().isoformat(),
        snapshots=[s.to_dict() for s in snapshots],
        sensitivity=sensitivity_rows,
    )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        logger.info("Commentary written to %s", path)

    return rendered
