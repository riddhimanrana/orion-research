"""Evaluate Action Genome SGA predictions (future-only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orion.evaluation.action_genome.adapter import build_video_graphs
from orion.evaluation.action_genome.evaluator import evaluate_sga
from orion.evaluation.action_genome.loader import load_action_genome
from orion.evaluation.core.runner import load_predictions_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Action Genome SGA predictions.")
    parser.add_argument("--ag-root", required=True, help="Root directory of Action Genome dataset.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", required=True, help="Path to prediction JSONL.")
    parser.add_argument("--fraction", type=float, default=0.9)
    parser.add_argument("--topk", nargs="*", type=int, default=[10, 20, 50])
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output", default="eval_outputs/ag_sga_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_action_genome(args.ag_root, split=args.split, max_videos=args.max_videos)
    gt_graphs = build_video_graphs(bundle.samples)
    pred_graphs = load_predictions_jsonl(args.predictions)

    result = evaluate_sga(gt_graphs, pred_graphs, top_ks=args.topk, fraction=args.fraction)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fraction": result.fraction, **result.metrics.to_dict()}
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("Action Genome SGA evaluation complete.")
    for key, value in payload.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
