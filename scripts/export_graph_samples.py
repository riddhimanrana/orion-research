#!/usr/bin/env python3
"""Export sample scene graph frames with detailed annotations for manual validation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.graph import export_graph_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotated scene graph samples")
    parser.add_argument("--results", required=True, help="results/<episode> directory containing scene_graph.jsonl")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--output", required=True, help="Directory to store samples")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of frames to export")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Allow frames without edges to be sampled (default: require at least one relation)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    video_path = Path(args.video)
    output_dir = Path(args.output)
    graph_path = results_dir / "scene_graph.jsonl"

    exports = export_graph_samples(
        graph_path=graph_path,
        video_path=video_path,
        output_dir=output_dir,
        max_samples=args.max_samples,
        require_edges=not args.include_empty,
    )

    if not exports:
        print("No eligible frames found; nothing exported.")
        return

    print(f"\nExported {len(exports)} frames to {output_dir}")
    total_edges = sum(sample.edge_count for sample in exports)
    print(f"  Avg edges/frame: {total_edges / len(exports):.2f}")
    relation_totals = {}
    for sample in exports:
        for rel, count in sample.relations.items():
            relation_totals[rel] = relation_totals.get(rel, 0) + count
    if relation_totals:
        print("  Relations:")
        for rel, count in sorted(relation_totals.items(), key=lambda item: item[0]):
            print(f"    {rel}: {count}")


if __name__ == "__main__":
    main()
