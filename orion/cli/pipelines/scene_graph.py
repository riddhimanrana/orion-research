#!/usr/bin/env python3
"""
Run Scene Graph Generation
====================================

Generates scene_graph.jsonl and graph_summary.json from results/<episode>/ memory.json and tracks.jsonl

Usage:
  python -m orion.cli.pipelines.scene_graph --results results/test_validation
"""

import argparse
import logging
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from orion.graph import (
    build_scene_graphs,
    save_scene_graphs,
    build_graph_summary,
    save_graph_summary,
    load_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Scene graph generation with spatial relations")
    ap.add_argument('--results', type=str, required=True, help='Path to results/<episode> directory')
    ap.add_argument('--relations', nargs='+', default=['near', 'on', 'held_by'], help='Relations to include')
    ap.add_argument('--near-dist', type=float, default=0.08, help='Normalized centroid distance for near')
    ap.add_argument('--on-overlap', type=float, default=0.3, help='Min horizontal overlap for on')
    ap.add_argument('--on-vgap', type=float, default=0.02, help='Max normalized vertical gap for on')
    ap.add_argument('--on-subj-overlap', type=float, default=0.5, help='Min overlap relative to subject width for on')
    ap.add_argument('--on-subj-within', type=float, default=0.0, help='Min fraction of subject x-span within object x-span (0=disable)')
    ap.add_argument('--on-small-area', type=float, default=0.05, help='Subject area fraction (of frame) considered small')
    ap.add_argument('--on-small-overlap', type=float, default=0.15, help='Relaxed min overlap for small subjects')
    ap.add_argument('--on-small-vgap', type=float, default=0.06, help='Relaxed max vertical gap for small subjects')
    ap.add_argument('--held-iou', type=float, default=0.3, help='IoU threshold for held_by')
    ap.add_argument('--iou-exclude', type=float, default=0.1, help='IoU exclusion for near (avoid overlaps)')
    ap.add_argument('--near-small-dist', type=float, default=0.06, help='Near distance for small objects')
    ap.add_argument('--near-small-area', type=float, default=0.05, help='Area fraction to consider an object small for near/adaptive heuristics')
    ap.add_argument('--exclude-on-pair', action='append', help='Class-pair to exclude from on relation, format "classA:classB". Can be repeated.')
    ap.add_argument('--use-pose-held', action='store_true', help='Use person keypoints/hand keypoints to prefer holders when available')
    ap.add_argument('--pose-hand-dist', type=float, default=0.03, help='Normalized distance threshold for hand proximity to object centroid')
    ap.add_argument('--no-class-filter', action='store_true', help='Disable class-based filtering')
    args = ap.parse_args()

    results_dir = Path(args.results)
    memory_path = results_dir / 'memory.json'
    tracks_path = results_dir / 'tracks.jsonl'
    graph_path = results_dir / 'scene_graph.jsonl'
    summary_path = results_dir / 'graph_summary.json'

    if not memory_path.exists():
        raise FileNotFoundError(f"Missing memory.json: {memory_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing tracks.jsonl: {tracks_path}")

    logger.info("==============================================================")
    logger.info("SCENE GRAPH GENERATION")
    logger.info("==============================================================")
    logger.info(f"Results dir: {results_dir}")

    memory = load_memory(memory_path)
    with tracks_path.open('r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    graphs = build_scene_graphs(
        memory,
        tracks,
        relations=args.relations,
        near_dist_norm=args.near_dist,
        near_small_dist_norm=args.near_small_dist,
        near_small_area=args.near_small_area,
        on_h_overlap=args.on_overlap,
        on_vgap_norm=args.on_vgap,
        on_subj_overlap_min=args.on_subj_overlap,
        on_subj_within_obj=args.on_subj_within,
        on_small_subject_area=args.on_small_area,
        on_small_overlap=args.on_small_overlap,
        on_small_vgap=args.on_small_vgap,
        exclude_on_pairs=[tuple(p.split(":")) for p in (args.exclude_on_pair or [])],
        use_pose_for_held=args.use_pose_held,
        pose_hand_dist=args.pose_hand_dist,
        iou_exclude=args.iou_exclude,
        held_by_iou=args.held_iou,
        enable_class_filtering=not args.no_class_filter,
    )

    save_scene_graphs(graphs, graph_path)
    logger.info(f"✓ Wrote {graph_path} ({len(graphs)} frames)")

    summary = build_graph_summary(graphs)
    save_graph_summary(summary, summary_path)
    logger.info(f"✓ Wrote {summary_path}")
    logger.info(f"  Total frames: {summary['total_frames']}")
    logger.info(f"  Avg nodes/frame: {summary['avg_nodes_per_frame']}")
    logger.info(f"  Avg edges/frame: {summary['avg_edges_per_frame']}")
    logger.info(f"  Relation counts: {summary['relation_counts']}")


if __name__ == '__main__':
    main()
