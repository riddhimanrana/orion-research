#!/usr/bin/env python3
"""Generate frame-level overlays that visualize DINO/YOLO detections.

The script reads a tracks JSONL file (each line = detection/tracked object)
and renders bounding boxes plus labels on the corresponding video frame.
This is helpful for auditing specific categories that appear over-counted.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2

Detection = Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the source video that was analyzed.",
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Path to the tracks JSONL emitted by the perception run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where annotated frames will be written.",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help=(
            "Category to visualize (repeatable). If omitted, all categories are used."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=4,
        help="Maximum number of frames to export (default: 4).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Ignore detections below this confidence (default: 0.3).",
    )
    parser.add_argument(
        "--highlight-top",
        action="store_true",
        help="Sort frames by total confidence instead of detection count.",
    )
    return parser.parse_args()


def color_for_label(label: str) -> tuple[int, int, int]:
    """Create a deterministic BGR color from the label."""

    seed = hashlib.sha1(label.encode("utf-8")).hexdigest()
    r = int(seed[0:2], 16)
    g = int(seed[2:4], 16)
    b = int(seed[4:6], 16)
    return (b, g, r)


def load_detections(tracks_path: Path, categories: Sequence[str] | None, min_conf: float) -> Dict[int, List[Detection]]:
    frames: Dict[int, List[Detection]] = defaultdict(list)
    cats = set(c.lower() for c in categories) if categories else None

    with tracks_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            det: Detection = json.loads(line)
            if cats and det.get("category", "").lower() not in cats:
                continue
            if float(det.get("confidence", 0.0)) < min_conf:
                continue
            frame_id = int(det["frame_id"])
            frames[frame_id].append(det)
    return frames


def select_frames(frames: Dict[int, List[Detection]], limit: int, highlight_top: bool) -> List[int]:
    if not frames:
        return []

    if highlight_top:
        scored = sorted(
            frames.items(),
            key=lambda item: sum(float(det["confidence"]) for det in item[1]),
            reverse=True,
        )
    else:
        scored = sorted(
            frames.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )

    return [frame_id for frame_id, _ in scored[:limit]]


def grab_frame(cap: cv2.VideoCapture, frame_id: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def draw_detections(frame, detections: Sequence[Detection]):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        label = str(det.get("category", "unknown"))
        conf = float(det.get("confidence", 0.0))
        color = color_for_label(label)
        cv2.rectangle(frame, p1, p2, color, 2)
        caption = f"{label} {conf:.2f}"
        text_origin = (p1[0], max(p1[1] - 5, 20))
        cv2.putText(
            frame,
            caption,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.tracks.exists():
        raise FileNotFoundError(f"Tracks file not found: {args.tracks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_detections(args.tracks, args.categories, args.min_confidence)
    target_frames = select_frames(frames, args.frames, args.highlight_top)

    if not target_frames:
        print("No matching detections found for the given filters.")
        return

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    try:
        for frame_id in target_frames:
            frame = grab_frame(cap, frame_id)
            if frame is None:
                print(f"[warn] Failed to read frame {frame_id}")
                continue
            draw_detections(frame, frames[frame_id])
            out_path = args.output_dir / f"frame_{frame_id:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved {out_path} with {len(frames[frame_id])} detections")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
