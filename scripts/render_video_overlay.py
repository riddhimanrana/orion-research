#!/usr/bin/env python3
"""Render annotated perception overlays with memory + spatial insights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.perception.viz_overlay import OverlayOptions, render_insight_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Orion perception overlay")
    parser.add_argument("--video", required=True, help="Source video path")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--output", help="Output mp4 path (defaults to results/video_overlay_insights.mp4)")
    parser.add_argument("--state-display", type=float, default=1.75, help="Seconds to keep narrative messages visible")
    parser.add_argument("--max-messages", type=int, default=5, help="Maximum concurrent narrative messages")
    parser.add_argument("--max-relations", type=int, default=4, help="Max relations listed per frame")
    parser.add_argument("--refind-gap", type=int, default=45, help="Frames that qualify as a refind event")
    parser.add_argument("--frame-offset", type=int, default=0, help="Shift overlays by this many frames (positive = overlays appear later)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    options = OverlayOptions(
        max_relations=args.max_relations,
        message_linger_seconds=args.state_display,
        max_state_messages=args.max_messages,
        gap_frames_for_refind=args.refind_gap,
        overlay_basename=(Path(args.output).name if args.output else "video_overlay_insights.mp4"),
        frame_offset=args.frame_offset,
    )
    output_path = Path(args.output) if args.output else None
    rendered_path = render_insight_overlay(
        video_path=video_path,
        results_dir=results_dir,
        output_path=output_path,
        options=options,
    )
    print(f"✓ Overlay video saved to {rendered_path}")


if __name__ == "__main__":
    main()
