#!/usr/bin/env python3
"""
Render Phase 1 overlay video with YOLO-World tracks and optional Gemini validation summaries.

- Draws tracked bounding boxes with label, track id, and confidence
- Samples frames at target FPS (matches detection cadence)
- Optionally overlays Gemini validation (false negatives/positives summary per frame)

Usage:
    python scripts/render_overlay_phase1.py \
        --video data/examples/test.mp4 \
        --results results/phase1_test_v2 \
        --fps 5 \
        --gemini gemini_validation.json \
        --output overlay_test_v2.mp4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def _load_tracks(tracks_path: Path) -> Dict[int, List[dict]]:
    by_frame: Dict[int, List[dict]] = {}
    with tracks_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fid = int(rec.get("frame_id", -1))
            if fid < 0:
                continue
            by_frame.setdefault(fid, []).append(rec)
    return by_frame


def _load_gemini(gemini_path: Optional[Path]) -> Dict[int, dict]:
    if not gemini_path or not gemini_path.exists():
        return {}
    with gemini_path.open("r") as f:
        data = json.load(f)
    out: Dict[int, dict] = {}
    for v in data.get("validations", []):
        fid = v.get("frame_id")
        if fid is None:
            continue
        out[int(fid)] = v
    return out


def _color_for_label(label: str) -> Tuple[int, int, int]:
    h = hash(label) & 0xFFFFFF
    return (int(h % 200), int((h // 200) % 200), int((h // 40000) % 200))


def _draw_label(img, x1: int, y1: int, text: str, color: Tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 3), font, scale, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_gemini_panel(img, validation: dict):
    h, w = img.shape[:2]
    panel_h = 120
    cv2.rectangle(img, (0, 0), (w, panel_h), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 24
    cv2.putText(img, "Gemini (detector audit)", (10, y), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y += 22
    acc = validation.get("accuracy_score", "?")
    cv2.putText(img, f"Accuracy: {acc}", (10, y), font, 0.55, (200, 255, 200), 2, cv2.LINE_AA)
    y += 20
    fp = validation.get("false_positives", [])
    fn = validation.get("false_negatives", [])
    fp_str = (", ".join(fp[:3]) + (" ..." if len(fp) > 3 else "")) or "None"
    fn_str = (", ".join(fn[:3]) + (" ..." if len(fn) > 3 else "")) or "None"
    cv2.putText(img, f"False Positives: {fp_str[:80]}", (10, y), font, 0.5, (180, 180, 255), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(img, f"False Negatives: {fn_str[:80]}", (10, y), font, 0.5, (180, 255, 255), 1, cv2.LINE_AA)
    y += 18
    comments = str(validation.get("comments", ""))[:110]
    cv2.putText(img, f"Comments: {comments}", (10, y), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def render_overlay(video_path: Path, results_dir: Path, tracks_file: str, gemini_file: Optional[str], output: str, fps: float):
    tracks_path = results_dir / tracks_file
    gemini_path = results_dir / gemini_file if gemini_file else None

    by_frame = _load_tracks(tracks_path)
    gemini = _load_gemini(gemini_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    interval = max(1, int(round(src_fps / fps)))
    target_frames = set(range(0, total_frames, interval))

    out_path = results_dir / output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    frame_idx = 0
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_frames:
            dets = by_frame.get(frame_idx, [])
            for d in dets:
                bbox = d.get("bbox") or d.get("bbox_2d")
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                label = str(d.get("label", "obj"))
                track_id = d.get("track_id", 0)
                conf = d.get("confidence", 0)
                color = _color_for_label(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                _draw_label(frame, x1, y1, f"{label}#{track_id} {conf:.2f}", color)

            if frame_idx in gemini:
                _draw_gemini_panel(frame, gemini[frame_idx])

            writer.write(frame)
            frames_written += 1

        frame_idx += 1

    cap.release()
    writer.release()

    return out_path, frames_written


def main():
    parser = argparse.ArgumentParser(description="Render overlay video for Phase 1 detections")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--results", required=True, help="Results directory containing tracks.jsonl")
    parser.add_argument("--tracks", default="tracks.jsonl", help="Tracks file name (default: tracks.jsonl)")
    parser.add_argument("--gemini", help="Gemini validation JSON file (optional)")
    parser.add_argument("--output", default="overlay_phase1.mp4", help="Output overlay mp4 filename")
    parser.add_argument("--fps", type=float, default=5.0, help="Output FPS (match detection cadence)")
    args = parser.parse_args()

    video_path = Path(args.video)
    results_dir = Path(args.results)

    out_path, frames_written = render_overlay(
        video_path, results_dir, args.tracks, args.gemini, args.output, args.fps
    )
    print(f"Overlay saved to {out_path} ({frames_written} frames)")


if __name__ == "__main__":
    main()
