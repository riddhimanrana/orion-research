#!/usr/bin/env python3
"""Convenience wrapper for running CIS and temporal validation tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_TESTS: List[str] = [
    "tests/test_cis_formula.py",
    "tests/unit/test_causal_inference.py",
    "tests/unit/test_motion_tracker.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CIS and temporal regression test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Optional explicit tests to run (defaults cover CIS + temporal components)",
    )
    parser.add_argument(
        "--pytest-args",
        dest="pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the default tests without executing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tests = args.tests or DEFAULT_TESTS

    if args.list:
        for target in tests:
            print(target)
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "pytest", *tests]
    if args.pytest_args:
        command.extend(args.pytest_args)

    print("Running:", " ".join(command))
    result = subprocess.run(command, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
