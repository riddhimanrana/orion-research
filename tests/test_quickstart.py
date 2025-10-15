"""Smoke test mirroring the quick-start instructions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_executes() -> None:
    """Ensure the CLI help command runs without triggering heavy downloads."""
    result = subprocess.run(
        [sys.executable, "-m", "orion.cli", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout.lower()