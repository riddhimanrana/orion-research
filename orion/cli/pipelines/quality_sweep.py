#!/usr/bin/env python3
"""Run a full perception quality sweep (Phase 1 → Phase 2 → graph → QA)."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from orion.config import (
    ensure_results_dir,
    get_episode_video_path,
)
from orion.cli.pipelines.tracks import process_video_to_tracks
from orion.graph import (
    GeminiValidationError,
    build_graph_summary,
    build_scene_graphs,
    export_graph_samples,
    load_memory,
    load_tracks,
    save_graph_summary,
    save_scene_graphs,
    validate_graph_samples,
)
from orion.perception.reid.matcher import build_memory_from_tracks

logger = logging.getLogger("orion.quality")


@dataclass
class EpisodeSpec:
    episode_id: str
    video_path: Path
    results_dir: Path


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _parse_episode_overrides(entries: List[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid episode override '{entry}'. Use episode_id=/path/to/video")
        key, value = entry.split("=", 1)
        mapping[key.strip()] = Path(value.strip())
    return mapping


def _resolve_episode_specs(args: argparse.Namespace) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    if args.plan:
        plan_path = Path(args.plan)
        plan_data = json.loads(plan_path.read_text())
        for item in plan_data:
            episode_id = item["episode"]
            video_path = Path(item["video"])
            results_dir = ensure_results_dir(episode_id)
            specs.append(EpisodeSpec(episode_id, video_path, results_dir))
        return specs

    if not args.episode:
        raise ValueError("Specify --episode or provide --plan")

    overrides = _parse_episode_overrides(args.episode_video or [])
    for ep in args.episode:
        if ep in overrides:
            video_path = overrides[ep]
        elif args.video and len(args.episode) == 1:
            video_path = Path(args.video)
        else:
            video = get_episode_video_path(ep)
            if video is None:
                raise FileNotFoundError(
                    f"Could not locate video for episode '{ep}'. Provide --video or --episode-video override."
                )
            video_path = video
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        results_dir = ensure_results_dir(ep)
        specs.append(EpisodeSpec(ep, video_path, results_dir))
    return specs


def _phase1_stage(spec: EpisodeSpec, args: argparse.Namespace) -> Optional[Dict]:
    tracks_path = spec.results_dir / "tracks.jsonl"
    run_meta_path = spec.results_dir / "run_metadata.json"
    if args.skip_phase1:
        return _load_json(run_meta_path)
    if tracks_path.exists() and not args.force_phase1:
        logger.info("[Phase 1] Reusing existing tracks for %s", spec.episode_id)
        return _load_json(run_meta_path)

    logger.info("[Phase 1] Running detection + tracking for %s", spec.episode_id)
    meta = process_video_to_tracks(
        video_path=str(spec.video_path),
        episode_id=spec.episode_id,
        target_fps=args.fps,
        yolo_model=args.yolo_model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        max_age=args.max_age,
        device=args.device,
        save_viz=False,
        enable_hand_detector=args.detect_hands,
        hand_max_hands=args.hand_max,
        hand_detection_confidence=args.hand_det_conf,
        hand_tracking_confidence=args.hand_track_conf,
    )
    return meta


def _phase2_stage(spec: EpisodeSpec, args: argparse.Namespace) -> Dict:
    memory_path = spec.results_dir / "memory.json"
    if args.skip_phase2:
        data = _load_json(memory_path)
        if data is None:
            raise FileNotFoundError(f"memory.json missing for {spec.episode_id}; cannot skip Phase 2")
        return data

    if memory_path.exists() and not args.force_phase2:
        logger.info("[Phase 2] Reusing memory.json for %s", spec.episode_id)
        return json.loads(memory_path.read_text())

    tracks_path = spec.results_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.jsonl missing for {spec.episode_id}; run Phase 1 first")

    logger.info("[Phase 2] Building memory.json for %s", spec.episode_id)
    build_memory_from_tracks(
        episode_id=spec.episode_id,
        video_path=spec.video_path,
        tracks_path=tracks_path,
        results_dir=spec.results_dir,
        cosine_threshold=args.reid_threshold,
        max_crops_per_track=args.max_crops_per_track,
        class_thresholds=None,
    )
    return json.loads(memory_path.read_text())


def _graph_stage(spec: EpisodeSpec, args: argparse.Namespace) -> Dict:
    summary_path = spec.results_dir / "graph_summary.json"
    graph_path = spec.results_dir / "scene_graph.jsonl"
    if args.skip_graph:
        data = _load_json(summary_path)
        if data is None:
            raise FileNotFoundError(f"graph_summary.json missing for {spec.episode_id}; cannot skip graph stage")
        return data

    if summary_path.exists() and graph_path.exists() and not args.force_graph:
        logger.info("[Graph] Reusing existing scene graph for %s", spec.episode_id)
        return json.loads(summary_path.read_text())

    memory_path = spec.results_dir / "memory.json"
    tracks_path = spec.results_dir / "tracks.jsonl"
    if not memory_path.exists() or not tracks_path.exists():
        raise FileNotFoundError("memory.json or tracks.jsonl missing; cannot build scene graph")

    memory = load_memory(memory_path)
    tracks = load_tracks(tracks_path)

    logger.info("[Graph] Building scene graph (%s) with %d tracks", spec.episode_id, len(tracks))
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
        iou_exclude=args.iou_exclude,
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


def _samples_stage(spec: EpisodeSpec, args: argparse.Namespace) -> Optional[Dict]:
    if args.skip_samples:
        return None
    sample_dir = spec.results_dir / args.samples_dir_name
    if sample_dir.exists() and args.force_samples:
        shutil.rmtree(sample_dir)

    graph_path = spec.results_dir / "scene_graph.jsonl"
    if not graph_path.exists():
        raise FileNotFoundError(f"scene_graph.jsonl missing for {spec.episode_id}; run graph stage first")

    logger.info("[Samples] Exporting up to %d frames for %s", args.max_graph_samples, spec.episode_id)
    samples = export_graph_samples(
        graph_path=graph_path,
        video_path=spec.video_path,
        output_dir=sample_dir,
        max_samples=args.max_graph_samples,
        require_edges=not args.include_empty_samples,
    )
    if not samples:
        logger.warning("No eligible graph samples exported for %s", spec.episode_id)
        return None

    relation_totals: Dict[str, int] = {}
    for sample in samples:
        for rel, count in sample.relations.items():
            relation_totals[rel] = relation_totals.get(rel, 0) + count
    avg_edges = sum(s.edge_count for s in samples) / len(samples)
    return {
        "sample_dir": str(sample_dir),
        "count": len(samples),
        "avg_edges_per_frame": round(avg_edges, 2),
        "relation_totals": relation_totals,
    }


def _gemini_stage(spec: EpisodeSpec, args: argparse.Namespace) -> Optional[Dict]:
    if not args.run_gemini:
        return None
    sample_dir = spec.results_dir / args.samples_dir_name
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory missing ({sample_dir}); run samples stage first")

    output_path = spec.results_dir / args.gemini_output_name
    try:
            payload = validate_graph_samples(
            sample_dir=sample_dir,
            output_path=output_path,
            max_samples=args.max_gemini_samples,
            model_name=args.gemini_model,
            api_key=args.gemini_api_key,
            sleep_seconds=args.gemini_sleep,
        )
    except GeminiValidationError as exc:
        logger.error("Gemini validation failed for %s: %s", spec.episode_id, exc)
        return None
    return {
        "output": str(output_path),
        "total_samples": payload.get("total_samples"),
        "verdict_counts": payload.get("verdict_counts", {}),
        "validated_at": payload.get("validated_at"),
    }


def _write_quality_report(spec: EpisodeSpec, report: Dict) -> Path:
    path = spec.results_dir / "quality_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def _build_report(
    spec: EpisodeSpec,
    phase1_meta: Optional[Dict],
    memory_data: Dict,
    graph_summary: Dict,
    samples_info: Optional[Dict],
    gemini_info: Optional[Dict],
) -> Dict:
    objects = memory_data.get("objects", [])
    report = {
        "episode": spec.episode_id,
        "video": str(spec.video_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase1": phase1_meta,
        "memory": {
            "object_count": len(objects),
            "statistics": memory_data.get("statistics", {}),
        },
        "graph": graph_summary,
        "samples": samples_info,
        "gemini": gemini_info,
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run perception quality sweep")
    parser.add_argument("--episode", action="append", help="Episode ID (repeatable)")
    parser.add_argument("--episode-video", action="append", help="episode_id=/path override", default=[])
    parser.add_argument("--video", help="Video override when a single episode is provided")
    parser.add_argument("--plan", help="Path to JSON plan file [{\"episode\": .., \"video\": ..}]")

    # Phase 1 options
    parser.add_argument("--fps", type=float, default=4.0, help="Target FPS for detection sampling")
    parser.add_argument("--yolo-model", default="yolo11m", choices=["yolo11n", "yolo11s", "yolo11m", "yolo11x"])
    parser.add_argument("--conf", dest="confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="Tracker IoU threshold")
    parser.add_argument("--max-age", type=int, default=30, help="Tracker max age")
    parser.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"], help="Inference device")
    parser.add_argument("--detect-hands", action="store_true", help="Enable MediaPipe hand detections")
    parser.add_argument("--hand-max", type=int, default=2)
    parser.add_argument("--hand-det-conf", type=float, default=0.5)
    parser.add_argument("--hand-track-conf", type=float, default=0.3)

    parser.add_argument("--force-phase1", action="store_true")
    parser.add_argument("--force-phase2", action="store_true")
    parser.add_argument("--force-graph", action="store_true")
    parser.add_argument("--force-samples", action="store_true")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-samples", action="store_true")

    parser.add_argument("--reid-threshold", type=float, default=0.70)
    parser.add_argument("--max-crops-per-track", type=int, default=5)

    # Graph parameters (matching run_scene_graph)
    parser.add_argument("--relations", nargs="+", default=["near", "on", "held_by"], help="Relations to compute")
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
    parser.add_argument("--iou-exclude", type=float, default=0.1)
    parser.add_argument("--pose-hand-dist", type=float, default=0.03)
    parser.add_argument("--use-pose-held", action="store_true")
    parser.add_argument("--no-class-filter", action="store_true")
    parser.add_argument("--exclude-on-pair", action="append", help="Class pair to exclude (classA:classB)")

    # Sample + Gemini options
    parser.add_argument("--samples-dir-name", default="graph_samples")
    parser.add_argument("--max-graph-samples", type=int, default=10)
    parser.add_argument("--include-empty-samples", action="store_true", help="Allow sampling frames without edges")

    parser.add_argument("--run-gemini", action="store_true")
    parser.add_argument("--max-gemini-samples", type=int, default=None)
    parser.add_argument("--gemini-model", default="gemini-2.0-flash")
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--gemini-sleep", type=float, default=1.0)
    parser.add_argument("--gemini-output-name", default="gemini_feedback.json")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    try:
        specs = _resolve_episode_specs(args)
    except Exception as exc:
        parser.error(str(exc))

    for spec in specs:
        logger.info("=== Quality sweep for %s ===", spec.episode_id)
        phase1_meta = _phase1_stage(spec, args)
        memory_data = _phase2_stage(spec, args)
        graph_summary = _graph_stage(spec, args)
        samples_info = _samples_stage(spec, args)
        gemini_info = _gemini_stage(spec, args)

        report = _build_report(spec, phase1_meta, memory_data, graph_summary, samples_info, gemini_info)
        report_path = _write_quality_report(spec, report)
        logger.info("✓ Wrote %s", report_path)
        if graph_summary:
            logger.info(
                "Graph stats: frames=%d, edges/frame=%.2f",
                graph_summary.get("total_frames", 0),
                graph_summary.get("avg_edges_per_frame", 0.0),
            )
        if samples_info:
            logger.info("Samples exported: %s", samples_info)
        if gemini_info:
            logger.info("Gemini verdict counts: %s", gemini_info.get("verdict_counts"))


if __name__ == "__main__":
    main()
