"""Aggregate object and relation statistics from Orion runs on AG videos.

This script scans ``results/ag_eval/<video_id>/`` (or a custom root), reads each
``scene_graph.jsonl``/``entities.json`` pair, and reports:

* Top-k object classes seen across all runs
* Top-k relation predicates
* Optional per-video breakdowns

Usage::

    conda activate orion
    python scripts/aggregate_ag_eval_stats.py --results-root results/ag_eval
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _iter_scene_graphs(scene_path: Path) -> Iterable[dict]:
    with scene_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line in {scene_path}: {exc}")


def _normalize_label(entry: dict, key: str, fallback: str) -> str:
    value = entry.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    for alt in ("class", "label", "name", "predicate"):
        val = entry.get(alt)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return fallback


def _accumulate_counts(scene_path: Path, object_counts: Counter, relation_counts: Counter) -> None:
    for graph in _iter_scene_graphs(scene_path):
        for obj in graph.get("objects", []):
            label = _normalize_label(obj, "class_name", "unknown")
            object_counts[label] += 1
        for rel in graph.get("relations", []):
            label = _normalize_label(rel, "predicate", "unknown")
            relation_counts[label] += 1


def _summarize_entities(entities_path: Path) -> Dict[str, int]:
    try:
        payload = json.loads(entities_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        print(f"[WARN] Failed to parse {entities_path}: {exc}")
        return {}

    per_class = Counter()
    for entry in payload.get("entities", []):
        label = str(entry.get("class", "unknown")).strip().lower()
        if not label:
            label = "unknown"
        per_class[label] += 1
    return dict(per_class)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Orion stats for Action Genome videos")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/ag_eval"),
        help="Root directory containing per-video Orion outputs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top items to display for objects/relations",
    )
    parser.add_argument(
        "--per-video",
        action="store_true",
        help="Print per-video summaries",
    )
    parser.add_argument(
        "--include-entities",
        action="store_true",
        help="Also aggregate counts from entities.json (per video)",
    )

    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Results root not found: {root}")

    object_counts: Counter = Counter()
    relation_counts: Counter = Counter()
    per_video_entity_counts: Dict[str, Dict[str, int]] = {}

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        scene_graph = subdir / "scene_graph.jsonl"
        if not scene_graph.is_file():
            print(f"[WARN] Missing scene_graph.jsonl in {subdir}")
            continue
        _accumulate_counts(scene_graph, object_counts, relation_counts)

        if args.include_entities:
            per_video_entity_counts[subdir.name] = _summarize_entities(subdir / "entities.json")

    if not object_counts:
        print("No scene graphs found. Did you run scripts/run_orion_on_ag_videos.py?")
        return

    print("\nTop object classes:")
    for cls, count in object_counts.most_common(args.top_k):
        print(f"  {cls:<20} {count:6d}")

    print("\nTop relation predicates:")
    for pred, count in relation_counts.most_common(args.top_k):
        print(f"  {pred:<20} {count:6d}")

    if args.include_entities and per_video_entity_counts and args.per_video:
        print("\nPer-video entity counts:")
        for video_id, counts in sorted(per_video_entity_counts.items()):
            total = sum(counts.values())
            top = ", ".join(f"{cls}:{cnt}" for cls, cnt in Counter(counts).most_common(5))
            print(f"  {video_id}: {total} entities ({top})")


if __name__ == "__main__":
    main()
