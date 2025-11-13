"""
Quick Frame Snapshot Visualizer
================================

Extracts a few key frames with visualizations to quickly see what's being detected
without processing the entire video.

Usage:
    python scripts/snapshot_vis.py --video data/examples/video.mp4 --num-frames 5
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

from orion.perception.config import get_accurate_config
from orion.perception.engine import PerceptionEngine


COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def draw_bbox_label(frame, bbox, label, conf, color=(0, 255, 0)):
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--num-frames", type=int, default=5, help="Number of snapshot frames")
    ap.add_argument("--output-dir", default="results/snapshots")
    args = ap.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Quick config for fast snapshots
    cfg = get_accurate_config()
    cfg.target_fps = 0.5  # sample slowly for diverse frames
    cfg.detection.confidence_threshold = 0.4
    
    engine = PerceptionEngine(config=cfg)
    engine._initialize_components()
    
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample evenly spaced frames
    sample_indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
    
    print(f"[Snapshot] Extracting {args.num_frames} frames from {args.video}")
    
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        timestamp = frame_idx / video_fps
        
        # Detect
        detections = engine.observer.detect_objects(frame, frame_idx, timestamp, width, height)
        
        # Draw
        vis = frame.copy()
        for j, det in enumerate(detections):
            color = COLORS[j % len(COLORS)]
            vis = draw_bbox_label(vis, det["bounding_box"], det["object_class"], det["confidence"], color)
        
        # Add frame info
        info_text = f"Frame {frame_idx}/{total_frames} ({timestamp:.1f}s) - {len(detections)} detections"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save
        out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  [{i+1}/{args.num_frames}] Saved {out_path} - {len(detections)} objects")
    
    cap.release()
    print(f"\n[Snapshot] ✓ Saved {args.num_frames} annotated frames to {out_dir}/")
    print(f"[Snapshot] View with: open {out_dir}/")


if __name__ == "__main__":
    main()
