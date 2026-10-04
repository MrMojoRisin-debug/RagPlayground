"""`uv run ui` -> starts the Streamlit chat app."""

import subprocess
import sys

from . import config


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", str(config.ROOT / "app.py"), *sys.argv[1:]]
    sys.exit(subprocess.call(cmd, cwd=config.ROOT))
