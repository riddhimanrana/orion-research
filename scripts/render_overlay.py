#!/usr/bin/env python3
"""
Render video overlay showing YOLO-World detections + tracking + Gemini validation.

Creates a video with:
- Bounding boxes for each detection
- Track IDs and class labels
- Color-coded by track ID
- Optional Gemini assessment panel

Usage:
    python scripts/render_overlay.py --results results/phase1_test_v2 --output overlay.mp4
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def generate_color(track_id: int) -> tuple:
    """Generate consistent color for track ID."""
    np.random.seed(track_id)
    return tuple(np.random.randint(50, 255, 3).tolist())


def draw_bbox(frame, bbox, track_id, label, confidence, color=None):
    """Draw bounding box with label."""
    x1, y1, x2, y2 = [int(b) for b in bbox]
    
    if color is None:
        color = generate_color(track_id)
    
    # Draw bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label_text = f"#{track_id} {label} {confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    
    cv2.rectangle(
        frame, 
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


def draw_info_panel(frame, frame_id, detections, gemini_data=None, get_label_fn=None):
    """Draw information panel on the frame."""
    height, width = frame.shape[:2]
    
    # Default label getter
    if get_label_fn is None:
        get_label_fn = lambda d: d.get('label', d.get('class_name', 'unknown'))
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    panel_height = 200
    cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = height - panel_height + 20
    
    # Frame info
    cv2.putText(
        frame,
        f"Frame: {frame_id}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    y_offset += 25
    
    # Detection count
    cv2.putText(
        frame,
        f"Detections: {len(detections)}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    y_offset += 25
    
    # Class distribution
    if detections:
        labels = [get_label_fn(d) for d in detections]
        from collections import Counter
        counts = Counter(labels)
        classes_text = ", ".join([f"{label}({cnt})" for label, cnt in counts.most_common(5)])
        cv2.putText(
            frame,
            f"Classes: {classes_text[:80]}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        y_offset += 25
    
    # Gemini assessment
    if gemini_data:
        acc = gemini_data.get("accuracy_score", "?")
        if isinstance(acc, (int, float)):
            acc_text = f"{acc:.0%}"
            acc_color = (0, 255, 0) if acc > 0.7 else (0, 165, 255) if acc > 0.4 else (0, 0, 255)
        else:
            acc_text = str(acc)
            acc_color = (200, 200, 200)
        
        cv2.putText(
            frame,
            f"Gemini Accuracy: {acc_text}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            acc_color,
            2
        )
        y_offset += 25
        
        # False negatives
        fn = gemini_data.get("false_negatives", [])
        if fn:
            fn_text = "Missed: " + ", ".join(fn[:3])
            if len(fn) > 3:
                fn_text += f" +{len(fn)-3} more"
            cv2.putText(
                frame,
                fn_text[:80],
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
            y_offset += 25
        
        # False positives
        fp = gemini_data.get("false_positives", [])
        if fp:
            fp_text = "Wrong: " + ", ".join([str(f)[:30] for f in fp[:2]])
            cv2.putText(
                frame,
                fp_text[:80],
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1
            )


def load_tracks(results_dir: Path):
    """Load tracks from tracks.jsonl."""
    tracks = []
    with open(results_dir / "tracks.jsonl") as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    return tracks


def load_gemini_validation(results_dir: Path):
    """Load Gemini validation data if available."""
    gemini_path = results_dir / "gemini_validation.json"
    if not gemini_path.exists():
        return {}
    
    with open(gemini_path) as f:
        data = json.load(f)
    
    # Index by frame_id
    by_frame = {}
    for v in data.get("validations", []):
        frame_id = v.get("frame_id")
        if frame_id is not None:
            by_frame[frame_id] = v
    
    return by_frame


def render_overlay(video_path: str, results_dir: Path, output_path: str, 
                   show_gemini: bool = True, fps_limit: int = None):
    """Render video overlay with detections and tracking."""
    print(f"Loading data from {results_dir}...")
    
    # Load tracks and group by frame
    tracks = load_tracks(results_dir)
    by_frame = defaultdict(list)
    for t in tracks:
        by_frame[t['frame_id']].append(t)
    
    print(f"  Loaded {len(tracks)} track observations across {len(by_frame)} frames")
    
    # Detect track format (support both old and new schemas)
    sample = tracks[0] if tracks else {}
    use_new_schema = 'bbox_2d' in sample
    
    if use_new_schema:
        print("  Using new schema (bbox_2d, class_name)")
        def get_bbox(det): return det.get('bbox_2d', [0, 0, 0, 0])
        def get_label(det): return det.get('class_name', 'unknown')
    else:
        print("  Using legacy schema (bbox, label)")
        def get_bbox(det): return det.get('bbox', [0, 0, 0, 0])
        def get_label(det): return det.get('label', 'unknown')
    
    # Load Gemini validation
    gemini_by_frame = {}
    if show_gemini:
        gemini_by_frame = load_gemini_validation(results_dir)
        print(f"  Loaded Gemini validation for {len(gemini_by_frame)} frames")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {width}x{height}, {video_fps:.1f} FPS, {total_frames} frames")
    
    # Setup output writer
    output_fps = min(video_fps, fps_limit) if fps_limit else video_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    print(f"Rendering to {output_path} at {output_fps:.1f} FPS...")
    print()
    
    frame_idx = 0
    rendered = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame
        detections = by_frame.get(frame_idx, [])
        
        # Draw all bounding boxes
        for det in detections:
            draw_bbox(
                frame,
                get_bbox(det),
                det['track_id'],
                get_label(det),
                det['confidence']
            )
        
        # Draw info panel
        gemini_data = gemini_by_frame.get(frame_idx) if show_gemini else None
        draw_info_panel(frame, frame_idx, detections, gemini_data, get_label)
        
        # Write frame
        out.write(frame)
        rendered += 1
        
        # Progress
        if rendered % 30 == 0:
            pct = (frame_idx / total_frames) * 100
            print(f"  [{pct:5.1f}%] Frame {frame_idx:5d}/{total_frames}", end='\r')
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\n\n{'='*80}")
    print(f"✓ Overlay rendered: {output_path}")
    print(f"  Frames: {rendered}")
    print(f"  Detections visualized: {len(tracks)}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Render video overlay with detections and tracking")
    parser.add_argument("--results", required=True, help="Results directory (e.g., results/phase1_test_v2)")
    parser.add_argument("--output", help="Output video path (default: <results>/overlay.mp4)")
    parser.add_argument("--no-gemini", action="store_true", help="Don't show Gemini validation panel")
    parser.add_argument("--fps-limit", type=int, help="Limit output FPS (default: same as input)")
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    
    # Load metadata to get video path
    with open(results_dir / "episode_meta.json") as f:
        meta = json.load(f)
    
    video_path = meta["video_path"]
    output_path = args.output or str(results_dir / "overlay.mp4")
    
    render_overlay(
        video_path,
        results_dir,
        output_path,
        show_gemini=not args.no_gemini,
        fps_limit=args.fps_limit
    )


if __name__ == "__main__":
    main()
