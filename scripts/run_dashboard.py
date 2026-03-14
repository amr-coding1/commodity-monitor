"""Launch the Streamlit dashboard."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "dashboard" / "app.py"


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true"],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()
