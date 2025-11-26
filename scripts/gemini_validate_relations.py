#!/usr/bin/env python3
"""Automatically validate scene graph relations using the Gemini Vision API."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.graph import (
    GeminiValidationError,
    validate_graph_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported graph samples with Gemini")
    parser.add_argument("--samples", required=True, help="Directory with frame_*.jpg/txt pairs")
    parser.add_argument("--output", required=True, help="Destination JSON file for Gemini feedback")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on samples to evaluate")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name (default: gemini-2.0-flash)")
    parser.add_argument("--api-key", default=None, help="Override GOOGLE_API_KEY environment variable")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between API calls to avoid rate limits")
    args = parser.parse_args()

    sample_dir = Path(args.samples)
    output_path = Path(args.output)

    try:
        payload = validate_graph_samples(
            sample_dir=sample_dir,
            output_path=output_path,
            max_samples=args.max_samples,
            model_name=args.model,
            api_key=args.api_key,
            sleep_seconds=args.sleep,
        )
    except (GeminiValidationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    verdicts = payload.get("verdict_counts", {})
    total = sum(verdicts.values()) or 1
    print(f"\n✓ Saved validation results to {output_path}")
    print("Summary:")
    for verdict in ("CORRECT", "INCORRECT", "AMBIGUOUS"):
        count = verdicts.get(verdict, 0)
        print(f"  {verdict}: {count} ({100.0 * count / total:.1f}%)")


if __name__ == "__main__":
    main()
