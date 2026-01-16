"""Evaluate PVSG SGG predictions (R@K, mR@K)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orion.evaluation.core.runner import load_predictions_jsonl
from orion.evaluation.pvsg.evaluator import evaluate_pvsg
from orion.evaluation.pvsg.loader import load_pvsg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PVSG SGG predictions.")
    parser.add_argument("--pvsg-root", required=True, help="Root directory of PVSG dataset.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--predictions", required=True, help="Path to prediction JSONL.")
    parser.add_argument("--topk", nargs="*", type=int, default=[20, 50, 100])
    parser.add_argument("--output", default="eval_outputs/pvsg_sgg_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_pvsg(args.pvsg_root, split=args.split)
    predictions = load_predictions_jsonl(args.predictions)
    summary = evaluate_pvsg(bundle, predictions, top_ks=args.topk)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2)

    print("PVSG SGG evaluation complete.")
    for key, value in summary.to_dict().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
