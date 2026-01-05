#!/usr/bin/env python3
"""End-to-end perception showcase with overlay + Memgraph export."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.cli.pipelines.tracks import process_video_to_tracks
from orion.config import ensure_results_dir, get_episode_video_path
from orion.graph import (
    build_graph_summary,
    build_scene_graphs,
    load_memory,
    load_tracks,
    save_graph_summary,
    save_scene_graphs,
)
from orion.graph.backends.exporter import MemgraphExportResult, export_results_to_memgraph
from orion.perception.reid.matcher import build_memory_from_tracks
from orion.perception.viz_overlay import OverlayOptions, render_insight_overlay

logger = logging.getLogger("orion.showcase")


def _resolve_video_path(args: argparse.Namespace) -> Path:
    if args.video:
        return Path(args.video)
    video = get_episode_video_path(args.episode)
    if video is None:
        raise FileNotFoundError(
            f"Could not locate video for episode '{args.episode}'. Provide --video explicitly."
        )
    return video


def _phase1(args: argparse.Namespace, video_path: Path, results_dir: Path) -> Dict:
    tracks_path = results_dir / "tracks.jsonl"
    meta_path = results_dir / "run_metadata.json"
    if args.skip_phase1:
        if not tracks_path.exists():
            raise FileNotFoundError("tracks.jsonl missing; cannot skip Phase 1")
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    if tracks_path.exists() and not args.force_phase1:
        logger.info("[Phase 1] Reusing existing tracks in %s", results_dir)
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    logger.info("[Phase 1] Running detection + tracking")
    return process_video_to_tracks(
        video_path=str(video_path),
        episode_id=args.episode,
        target_fps=args.fps,
        yolo_model=args.yolo_model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        max_age=args.max_age,
        device=args.device,
        save_viz=args.save_viz,
        enable_hand_detector=args.detect_hands,
        hand_max_hands=args.hand_max,
        hand_detection_confidence=args.hand_det_conf,
        hand_tracking_confidence=args.hand_track_conf,
    )


def _phase2(args: argparse.Namespace, video_path: Path, results_dir: Path) -> Dict:
    memory_path = results_dir / "memory.json"
    if args.skip_memory:
        if not memory_path.exists():
            raise FileNotFoundError("memory.json missing; cannot skip Phase 2")
        return json.loads(memory_path.read_text())

    if memory_path.exists() and not args.force_memory:
        logger.info("[Phase 2] Reusing memory.json")
        return json.loads(memory_path.read_text())

    tracks_path = results_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError("tracks.jsonl missing; run Phase 1 first")

    logger.info("[Phase 2] Building memory.json")
    build_memory_from_tracks(
        episode_id=args.episode,
        video_path=video_path,
        tracks_path=tracks_path,
        results_dir=results_dir,
        cosine_threshold=args.reid_threshold,
        max_crops_per_track=args.max_crops_per_track,
        class_thresholds=None,
    )
    return json.loads(memory_path.read_text())


def _phase3_graph(args: argparse.Namespace, results_dir: Path) -> Dict:
    graph_path = results_dir / "scene_graph.jsonl"
    summary_path = results_dir / "graph_summary.json"
    if args.skip_graph:
        if not summary_path.exists():
            raise FileNotFoundError("graph_summary.json missing; cannot skip graph stage")
        return json.loads(summary_path.read_text())

    if graph_path.exists() and summary_path.exists() and not args.force_graph:
        logger.info("[Graph] Reusing existing scene graph")
        return json.loads(summary_path.read_text())

    memory_path = results_dir / "memory.json"
    tracks_path = results_dir / "tracks.jsonl"
    if not memory_path.exists() or not tracks_path.exists():
        raise FileNotFoundError("memory.json or tracks.jsonl missing for graph stage")

    memory = load_memory(memory_path)
    tracks = load_tracks(tracks_path)
    logger.info("[Graph] Building scene graph (%d tracks)", len(tracks))
    graphs = build_scene_graphs(
        memory,
        tracks,
        relations=args.relations,
        near_dist_norm=args.near_dist,
        on_h_overlap=args.on_overlap,
        on_vgap_norm=args.on_vgap,
        on_subj_overlap_min=args.on_subj_overlap,
        on_subj_within_obj=args.on_subj_within,
        on_small_subject_area=args.on_small_area,
        on_small_overlap=args.on_small_overlap,
        on_small_vgap=args.on_small_vgap,
        held_by_iou=args.held_iou,
        near_small_dist_norm=args.near_small_dist,
        near_small_area=args.near_small_area,
        exclude_on_pairs=[tuple(pair.split(":")) for pair in (args.exclude_on_pair or [])],
        use_pose_for_held=args.use_pose_held,
        pose_hand_dist=args.pose_hand_dist,
        enable_class_filtering=not args.no_class_filter,
    )
    save_scene_graphs(graphs, graph_path)
    summary = build_graph_summary(graphs)
    save_graph_summary(summary, summary_path)
    return summary


def _render_overlay(args: argparse.Namespace, video_path: Path, results_dir: Path) -> Optional[Path]:
    if args.no_overlay:
        return None
    logger.info("[Overlay] Rendering insight overlay")
    options = OverlayOptions(
        max_relations=args.overlay_max_relations,
        message_linger_seconds=args.overlay_message_seconds,
        max_state_messages=args.overlay_max_messages,
        gap_frames_for_refind=args.overlay_refind_gap,
        overlay_basename=(Path(args.overlay_output).name if args.overlay_output else "video_overlay_insights.mp4"),
    )
    output_path = Path(args.overlay_output) if args.overlay_output else None
    return render_insight_overlay(video_path=video_path, results_dir=results_dir, output_path=output_path, options=options)


def _export_memgraph(args: argparse.Namespace, video_path: Path, results_dir: Path) -> Optional[MemgraphExportResult]:
    if not args.memgraph:
        return None
    logger.info("[Memgraph] Exporting results to %s:%d", args.memgraph_host, args.memgraph_port)
    return export_results_to_memgraph(
        results_dir=results_dir,
        video_path=video_path,
        host=args.memgraph_host,
        port=args.memgraph_port,
        clear_existing=args.memgraph_clear,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run perception showcase on a video")
    parser.add_argument("--episode", required=True, help="Episode ID for saving results")
    parser.add_argument("--video", help="Override video path (defaults to episode video)")
    parser.add_argument("--fps", type=float, default=4.0, help="Target FPS for detection sampling")
    parser.add_argument("--yolo-model", default="yolo11m", choices=["yolo11n", "yolo11s", "yolo11m", "yolo11x"])
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="Tracker IoU threshold")
    parser.add_argument("--max-age", type=int, default=30, help="Tracker max age")
    parser.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--save-viz", action="store_true", help="Persist detector debug outputs")
    parser.add_argument("--detect-hands", action="store_true")
    parser.add_argument("--hand-max", type=int, default=2)
    parser.add_argument("--hand-det-conf", type=float, default=0.5)
    parser.add_argument("--hand-track-conf", type=float, default=0.3)
    parser.add_argument("--reid-threshold", type=float, default=0.70)
    parser.add_argument("--max-crops-per-track", type=int, default=5)

    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--force-phase1", action="store_true")
    parser.add_argument("--force-memory", action="store_true")
    parser.add_argument("--force-graph", action="store_true")

    parser.add_argument("--relations", nargs="+", default=["near", "on", "held_by"])
    parser.add_argument("--near-dist", type=float, default=0.08)
    parser.add_argument("--near-small-dist", type=float, default=0.06)
    parser.add_argument("--near-small-area", type=float, default=0.05)
    parser.add_argument("--on-overlap", type=float, default=0.3)
    parser.add_argument("--on-vgap", type=float, default=0.02)
    parser.add_argument("--on-subj-overlap", type=float, default=0.5)
    parser.add_argument("--on-subj-within", type=float, default=0.0)
    parser.add_argument("--on-small-area", type=float, default=0.05)
    parser.add_argument("--on-small-overlap", type=float, default=0.15)
    parser.add_argument("--on-small-vgap", type=float, default=0.06)
    parser.add_argument("--held-iou", type=float, default=0.3)
    parser.add_argument("--pose-hand-dist", type=float, default=0.03)
    parser.add_argument("--use-pose-held", action="store_true")
    parser.add_argument("--no-class-filter", action="store_true")
    parser.add_argument("--exclude-on-pair", action="append", help="Exclude class pair from 'on' relation (classA:classB)")

    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay rendering")
    parser.add_argument("--overlay-output", help="Explicit overlay path")
    parser.add_argument("--overlay-message-seconds", type=float, default=1.75)
    parser.add_argument("--overlay-max-messages", type=int, default=5)
    parser.add_argument("--overlay-refind-gap", type=int, default=45)
    parser.add_argument("--overlay-max-relations", type=int, default=4)

    parser.add_argument("--memgraph", action="store_true", help="Write outputs to Memgraph")
    parser.add_argument("--memgraph-host", default="127.0.0.1")
    parser.add_argument("--memgraph-port", type=int, default=7687)
    parser.add_argument("--memgraph-clear", action="store_true", help="Clear graph before ingesting")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    results_dir = ensure_results_dir(args.episode)
    video_path = _resolve_video_path(args)

    phase1_meta = _phase1(args, video_path, results_dir)
    memory_data = _phase2(args, video_path, results_dir)
    graph_summary = _phase3_graph(args, results_dir)
    overlay_path = _render_overlay(args, video_path, results_dir)
    memgraph_result = _export_memgraph(args, video_path, results_dir)

    logger.info("\n=== SHOWCASE SUMMARY ===")
    logger.info("Episode: %s", args.episode)
    logger.info("Video: %s", video_path)
    if phase1_meta:
        stats = phase1_meta.get("statistics", {})
        logger.info("Tracks: %s frames, %s unique", stats.get("frames_processed"), stats.get("unique_tracks"))
    logger.info("Memory objects: %d", len(memory_data.get("objects", [])))
    logger.info(
        "Graph: frames=%s edges/frame=%.2f",
        graph_summary.get("total_frames"),
        graph_summary.get("avg_edges_per_frame", 0.0),
    )
    if overlay_path:
        logger.info("Overlay: %s", overlay_path)
    if memgraph_result:
        logger.info(
            "Memgraph: %d observations, %d relations → %s:%d",
            memgraph_result.observations_written,
            memgraph_result.relations_written,
            memgraph_result.output_host,
            memgraph_result.output_port,
        )
if __name__ == "__main__":
    main()
