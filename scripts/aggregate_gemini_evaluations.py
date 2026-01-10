#!/usr/bin/env python3
"""Aggregate multiple gemini_evaluation.json files into a single summary.

Inputs:
- One or more Orion results directories (each containing gemini_evaluation.json)

Outputs:
- A Markdown summary table + optional JSON aggregate.

This is useful for evaluating K=10 videos from a larger benchmark run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Row:
    name: str
    precision: float
    recall: float
    f1: float
    label_accuracy: float
    frames_sampled: int
    detections: int
    model: str


def _load_one(results_dir: Path) -> Optional[Row]:
    p = results_dir / "gemini_evaluation.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text())
    return Row(
        name=results_dir.name,
        precision=float(obj.get("detection_precision", 0.0)),
        recall=float(obj.get("detection_recall", 0.0)),
        f1=float(obj.get("detection_f1", 0.0)),
        label_accuracy=float(obj.get("label_accuracy", 0.0)),
        frames_sampled=int(obj.get("total_frames_sampled", 0)),
        detections=int(obj.get("total_detections", 0)),
        model=str(obj.get("model_used", "")),
    )


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Gemini evaluation reports")
    ap.add_argument("results", nargs="+", type=Path, help="Results dirs containing gemini_evaluation.json")
    ap.add_argument("--out-md", type=Path, default=Path("gemini_eval_summary.md"))
    ap.add_argument("--out-json", type=Path, help="Optional JSON aggregate output")

    args = ap.parse_args()

    rows: List[Row] = []
    for d in args.results:
        r = _load_one(d)
        if r:
            rows.append(r)

    if not rows:
        raise SystemExit("No gemini_evaluation.json found in provided directories")

    # Aggregate
    agg = {
        "count": len(rows),
        "mean_precision": _mean([r.precision for r in rows]),
        "mean_recall": _mean([r.recall for r in rows]),
        "mean_f1": _mean([r.f1 for r in rows]),
        "mean_label_accuracy": _mean([r.label_accuracy for r in rows]),
        "models": sorted({r.model for r in rows if r.model}),
    }

    # Markdown output
    lines: List[str] = []
    lines.append("# Gemini evaluation summary")
    lines.append("")
    lines.append(f"Videos evaluated: **{agg['count']}**")
    lines.append("")
    lines.append("## Mean metrics")
    lines.append("")
    lines.append(f"- Precision: {agg['mean_precision']:.1%}")
    lines.append(f"- Recall: {agg['mean_recall']:.1%}")
    lines.append(f"- F1: {agg['mean_f1']:.1%}")
    lines.append(f"- Label accuracy: {agg['mean_label_accuracy']:.1%}")
    lines.append("")

    lines.append("## Per-video")
    lines.append("")
    lines.append("| Results dir | Precision | Recall | F1 | Label Acc | Frames | Detections | Model |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda x: x.name):
        lines.append(
            f"| {r.name} | {r.precision:.1%} | {r.recall:.1%} | {r.f1:.1%} | {r.label_accuracy:.1%} | {r.frames_sampled} | {r.detections} | {r.model} |"
        )

    args.out_md.write_text("\n".join(lines) + "\n")

    if args.out_json:
        args.out_json.write_text(json.dumps({"aggregate": agg, "rows": [r.__dict__ for r in rows]}, indent=2))

    print(f"Wrote {args.out_md}")
    if args.out_json:
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
