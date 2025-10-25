#!/usr/bin/env python3
"""Convenience wrapper for running CIS and temporal validation tests.

This script executes the complete CIS and temporal regression test suite,
with filtering support for specific components.

Examples:
    # Run all tests
    python scripts/run_cis_temporal_tests.py

    # Run with verbose output
    python scripts/run_cis_temporal_tests.py --pytest-args -v

    # List available tests
    python scripts/run_cis_temporal_tests.py --list

    # Run only CIS formula tests with verbose output
    python scripts/run_cis_temporal_tests.py tests/test_cis_formula.py --pytest-args -v

    # Run with coverage
    python scripts/run_cis_temporal_tests.py --pytest-args --cov=orion.semantic.causal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_TESTS: List[str] = [
    "tests/test_cis_formula.py",
    "tests/unit/test_causal_inference.py",
    "tests/unit/test_motion_tracker.py::TestMotionData",
    "tests/unit/test_motion_tracker.py::TestMotionTracker",
    "tests/unit/test_motion_tracker.py::TestUtilityFunctions",
]

TEST_GROUPS = {
    "cis": ["tests/test_cis_formula.py"],
    "causal": ["tests/unit/test_causal_inference.py"],
    "motion": ["tests/unit/test_motion_tracker.py::TestMotionData", "tests/unit/test_motion_tracker.py::TestMotionTracker"],
    "utils": ["tests/unit/test_motion_tracker.py::TestUtilityFunctions"],
    "all": DEFAULT_TESTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CIS and temporal regression test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Optional explicit tests to run (defaults: all CIS + temporal components)",
    )
    parser.add_argument(
        "--filter",
        choices=list(TEST_GROUPS.keys()),
        help="Filter tests by category (cis, causal, motion, utils, all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and exit",
    )
    parser.add_argument(
        "--pytest-args",
        dest="pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest (e.g., -v, --cov=...)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure (-x flag to pytest)",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Drop into pdb on failures (--pdb flag to pytest)",
    )
    return parser.parse_args()


def list_tests() -> None:
    """Display available test groups and default tests."""
    print("Available Test Groups:")
    print("-" * 60)
    for group_name, group_tests in TEST_GROUPS.items():
        print(f"\n  {group_name}:")
        for test in group_tests:
            print(f"    - {test}")


def main() -> int:
    args = parse_args()

    if args.list:
        list_tests()
        return 0

    repo_root = Path(__file__).resolve().parents[1]

    # Determine which tests to run
    if args.filter:
        tests = TEST_GROUPS[args.filter]
    elif args.tests:
        tests = args.tests
    else:
        tests = DEFAULT_TESTS

    # Build pytest command
    command = [sys.executable, "-m", "pytest", "-v", *tests]

    # Add optional flags
    if args.fail_fast:
        command.insert(3, "-x")
    if args.pdb:
        command.insert(3, "--pdb")

    # Add custom pytest args
    if args.pytest_args:
        command.extend(args.pytest_args)

    print("\n" + "=" * 70)
    print("CIS & TEMPORAL REGRESSION TEST SUITE")
    print("=" * 70)
    print(f"Tests: {', '.join(tests)}")
    print(f"Command: {' '.join(command)}")
    print("=" * 70 + "\n")

    result = subprocess.run(command, cwd=repo_root)
    
    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ Tests failed with exit code {result.returncode}")
    print("=" * 70)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
