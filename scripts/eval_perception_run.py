"""
Evaluate Perception Run
=======================

Runs the Phase 1 perception pipeline on a given video and prints a concise
summary of what happened: detections, entities, timings per phase, and
(optional) tracking metrics if enabled.

Usage:
    python scripts/eval_perception_run.py --video data/examples/video.mp4 --mode quick
    python scripts/eval_perception_run.py --video /path/to/video.mp4 --mode accurate --tracking

Modes:
  - quick:    lower compute (fewer frames, higher confidence)
  - balanced: default configuration
  - accurate: DINO + higher quality settings

Outputs a JSON to results/perception_run.json unless --no-save is passed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from orion.perception.config import get_fast_config, get_balanced_config, get_accurate_config
from orion.perception.engine import run_perception


def build_config(mode: str, tracking: bool, target_fps: float | None, conf: float | None):
    if mode == "quick":
        cfg = get_fast_config()
        # speed up further for quick evals
        cfg.target_fps = 0.5 if target_fps is None else float(target_fps)
        cfg.detection.confidence_threshold = 0.5 if conf is None else float(conf)
    elif mode == "balanced":
        cfg = get_balanced_config()
        if target_fps is not None:
            cfg.target_fps = float(target_fps)
        if conf is not None:
            cfg.detection.confidence_threshold = float(conf)
    elif mode == "accurate":
        cfg = get_accurate_config()
        if target_fps is not None:
            cfg.target_fps = float(target_fps)
        if conf is not None:
            cfg.detection.confidence_threshold = float(conf)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    cfg.enable_tracking = bool(tracking)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--mode", default="balanced", choices=["quick", "balanced", "accurate"], help="Preset config")
    ap.add_argument("--tracking", action="store_true", help="Enable per-frame tracking metrics")
    ap.add_argument("--target-fps", type=float, default=None, help="Override target FPS for sampling")
    ap.add_argument("--conf", type=float, default=None, help="Override YOLO confidence threshold")
    ap.add_argument("--no-save", action="store_true", help="Do not write results/perception_run.json")
    args = ap.parse_args()

    cfg = build_config(args.mode, args.tracking, args.target_fps, args.conf)

    res = run_perception(args.video, config=cfg)

    metrics = res.metrics or {}
    timings = metrics.get("timings", {})

    summary = {
        "video": args.video,
        "mode": args.mode,
        "yolo_model": metrics.get("yolo_model"),
        "embedding_backend": metrics.get("embedding_backend"),
        "embedding_dim": metrics.get("embedding_dim"),
        "unique_entities": res.unique_entities,
        "total_detections": res.total_detections,
        "sampled_frames": metrics.get("sampled_frames"),
        "detections_per_sampled_frame": metrics.get("detections_per_sampled_frame"),
        "timings_seconds": {
            "detection": round(float(timings.get("detection_seconds", 0.0)), 3),
            "embedding": round(float(timings.get("embedding_seconds", 0.0)), 3),
            "tracking": round(float(timings.get("tracking_seconds", 0.0)), 3),
            "clustering": round(float(timings.get("clustering_seconds", 0.0)), 3),
            "description": round(float(timings.get("description_seconds", 0.0)), 3),
            "total": round(float(timings.get("total_seconds", res.processing_time_seconds)), 3),
        },
        "tracking_metrics": {
            k: metrics.get(k)
            for k in ("total_tracks", "confirmed_tracks", "active_tracks", "id_switches")
            if k in metrics
        },
    }

    # Pretty print concise storyline
    print("\n=== Perception Run Summary ===")
    print(f"Video: {summary['video']}")
    print(f"Mode: {summary['mode']}  |  YOLO: {summary['yolo_model']}  |  Embedding: {summary['embedding_backend']} ({summary['embedding_dim']})")
    print(f"From frames → detections → embeddings → tracks → entities")
    print(f"Sampled frames: {summary['sampled_frames']}, Detections: {summary['total_detections']}, Entities: {summary['unique_entities']}")
    t = summary["timings_seconds"]
    print(
        "Timings (s): "
        f"detect={t['detection']}, embed={t['embedding']}, track={t['tracking']}, "
        f"cluster={t['clustering']}, describe={t['description']}, total={t['total']}"
    )
    if summary["tracking_metrics"]:
        print(f"Tracking: {summary['tracking_metrics']}")

    if not args.no_save:
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "perception_run.json"
        with out_file.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {out_file}")


if __name__ == "__main__":
    main()
