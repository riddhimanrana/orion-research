#!/usr/bin/env python3
"""Run Orion showcase pipeline over a list of videos.

This is a thin batch wrapper around `orion.cli.pipelines.showcase`.

Why this exists:
- reproducible batch runs
- resumable (skips videos with existing tracks/memory unless forced)
- writes one results folder per video (episode id derived from filename)

This script does NOT assume anything about Ego4D specifically; it can be used
for any dataset.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List


def _sanitize_episode_id(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:180]


def _read_lines(p: Path) -> List[str]:
    lines = []
    for raw in p.read_text().splitlines():
        v = raw.strip()
        if not v or v.startswith("#"):
            continue
        lines.append(v)
    return lines


def _run_one(
    video_path: Path,
    episode_prefix: str,
    fps: float,
    device: str,
    yolo_model: str,
    confidence: float,
    iou: float,
    max_age: int,
    reid_threshold: float,
    max_crops_per_track: int,
    skip_graph: bool,
    no_overlay: bool,
    force_phase1: bool,
    force_memory: bool,
    force_graph: bool,
) -> int:
    episode_id = _sanitize_episode_id(f"{episode_prefix}{video_path.stem}")

    cmd = [
        sys.executable,
        "-m",
        "orion.cli.pipelines.showcase",
        "--episode",
        episode_id,
        "--video",
        str(video_path),
        "--fps",
        str(fps),
        "--device",
        device,
        "--yolo-model",
        yolo_model,
        "--confidence",
        str(confidence),
        "--iou",
        str(iou),
        "--max-age",
        str(max_age),
        "--reid-threshold",
        str(reid_threshold),
        "--max-crops-per-track",
        str(max_crops_per_track),
    ]

    if skip_graph:
        cmd.append("--skip-graph")
    if no_overlay:
        cmd.append("--no-overlay")
    if force_phase1:
        cmd.append("--force-phase1")
    if force_memory:
        cmd.append("--force-memory")
    if force_graph:
        cmd.append("--force-graph")

    p = subprocess.run(cmd)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-run Orion showcase on many videos")
    ap.add_argument("--video-list", type=Path, required=True, help="Text file: one video path per line")
    ap.add_argument("--episode-prefix", default="ego4d_", help="Prefix for episode IDs")

    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--yolo-model", default="yolo11m", choices=["yolo11n", "yolo11s", "yolo11m", "yolo11x"])
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-age", type=int, default=30)

    ap.add_argument("--reid-threshold", type=float, default=0.70)
    ap.add_argument("--max-crops-per-track", type=int, default=5)

    ap.add_argument("--skip-graph", action="store_true", help="Skip scene graph stage (faster)")
    ap.add_argument("--no-overlay", action="store_true", help="Skip overlay rendering (faster)")

    ap.add_argument("--force-phase1", action="store_true")
    ap.add_argument("--force-memory", action="store_true")
    ap.add_argument("--force-graph", action="store_true")

    args = ap.parse_args()

    videos = [Path(p) for p in _read_lines(args.video_list)]
    missing = [p for p in videos if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} video files (first: {missing[0]})")

    failures: List[Path] = []
    for i, vp in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {vp}")
        rc = _run_one(
            video_path=vp,
            episode_prefix=args.episode_prefix,
            fps=args.fps,
            device=args.device,
            yolo_model=args.yolo_model,
            confidence=args.confidence,
            iou=args.iou,
            max_age=args.max_age,
            reid_threshold=args.reid_threshold,
            max_crops_per_track=args.max_crops_per_track,
            skip_graph=args.skip_graph,
            no_overlay=args.no_overlay,
            force_phase1=args.force_phase1,
            force_memory=args.force_memory,
            force_graph=args.force_graph,
        )
        if rc != 0:
            failures.append(vp)

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
