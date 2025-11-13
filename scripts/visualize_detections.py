"""
Visual Evaluation: Draw Detections, IDs, Zones, Trajectories
==============================================================

Visualizes the perception pipeline output on video frames:
- Bounding boxes with class labels and confidence
- Tracking IDs (if available)
- Spatial zones (color-coded backgrounds)
- Object trajectories and motion trails
- Depth heatmap overlay (optional)

Usage:
    python scripts/visualize_detections.py --video data/examples/video.mp4 --mode quick --output results/vis_quick.mp4
    python scripts/visualize_detections.py --video data/examples/video.mp4 --mode accurate --tracking --depth

Outputs an annotated video to results/ showing what the system detected.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from orion.perception.config import get_fast_config, get_balanced_config, get_accurate_config
from orion.perception.engine import PerceptionEngine


# Color palette for classes (BGR format)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
]

# Zone color overlays (BGR, alpha will be applied)
ZONE_COLORS = {
    "center": (200, 200, 200),
    "left": (150, 180, 255),
    "right": (255, 180, 150),
    "top": (200, 255, 200),
    "bottom": (255, 220, 180),
    "top_left": (180, 200, 255),
    "top_right": (255, 200, 180),
    "bottom_left": (180, 220, 255),
    "bottom_right": (255, 220, 200),
}


def draw_zone_overlay(frame: np.ndarray, zone: str, alpha: float = 0.1) -> np.ndarray:
    """Draw semi-transparent zone overlay."""
    overlay = frame.copy()
    color = ZONE_COLORS.get(zone, (200, 200, 200))
    overlay[:] = color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def draw_depth_heatmap(frame: np.ndarray, depth_map: np.ndarray | None, alpha: float = 0.3) -> np.ndarray:
    """Overlay depth as heatmap."""
    if depth_map is None:
        return frame
    
    # Normalize depth to 0-255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    
    # Resize to match frame
    if depth_colored.shape[:2] != frame.shape[:2]:
        depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
    
    return cv2.addWeighted(frame, 1 - alpha, depth_colored, alpha, 0)


def draw_detection(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    track_id: int | None = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box with label."""
    x1, y1, x2, y2 = bbox
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text
    if track_id is not None:
        text = f"ID{track_id} {label} {confidence:.2f}"
    else:
        text = f"{label} {confidence:.2f}"
    
    # Draw label background
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
    
    # Draw label text
    cv2.putText(
        frame, text, (x1, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    
    return frame


def draw_trajectory(
    frame: np.ndarray,
    centroids: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 255),
    max_trail: int = 30,
) -> np.ndarray:
    """Draw motion trail."""
    if len(centroids) < 2:
        return frame
    
    # Draw lines connecting recent centroids
    trail = centroids[-max_trail:]
    for i in range(1, len(trail)):
        pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        pt2 = (int(trail[i][0]), int(trail[i][1]))
        # Fade alpha based on age
        alpha = i / len(trail)
        thickness = max(1, int(2 * alpha))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    return frame


def build_config(mode: str, tracking: bool, enable_3d: bool, target_fps: float | None):
    if mode == "quick":
        cfg = get_fast_config()
        cfg.target_fps = 0.5 if target_fps is None else target_fps
        cfg.detection.confidence_threshold = 0.5
    elif mode == "balanced":
        cfg = get_balanced_config()
        if target_fps:
            cfg.target_fps = target_fps
    elif mode == "accurate":
        cfg = get_accurate_config()
        if target_fps:
            cfg.target_fps = target_fps
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    cfg.enable_tracking = tracking
    cfg.enable_3d = enable_3d or cfg.enable_3d
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--mode", default="quick", choices=["quick", "balanced", "accurate"])
    ap.add_argument("--tracking", action="store_true", help="Enable tracking for IDs")
    ap.add_argument("--depth", action="store_true", help="Overlay depth heatmap")
    ap.add_argument("--zones", action="store_true", help="Show spatial zone overlays")
    ap.add_argument("--trails", action="store_true", help="Draw motion trajectories")
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--output", default=None, help="Output video path (default: results/vis_{mode}.mp4)")
    args = ap.parse_args()
    
    # Prepare output
    if args.output is None:
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True, parents=True)
        args.output = str(out_dir / f"vis_{args.mode}.mp4")
    
    print(f"[VisEval] Processing {args.video} in {args.mode} mode...")
    
    # Build config
    cfg = build_config(args.mode, args.tracking, args.depth, args.target_fps)
    
    # Run perception engine to get detections
    engine = PerceptionEngine(config=cfg)
    
    # Open video for reading frames
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, video_fps, (width, height))
    
    print(f"[VisEval] Video: {width}x{height} @ {video_fps:.2f} FPS, {total_frames} frames")
    
    # Initialize perception components manually for per-frame access
    engine._initialize_components()
    
    # Track trajectories per entity
    trajectories: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    
    # Process frames
    frame_interval = max(1, int(video_fps / cfg.target_fps))
    frame_count = 0
    
    pbar = tqdm(total=total_frames, desc="Visualizing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated = frame.copy()
        
        # Process sampled frames
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            
            # Run detection on this frame
            frame_dets = engine.observer.detect_objects(
                frame, frame_count, timestamp, width, height
            )
            
            # Embed
            if frame_dets:
                frame_dets = engine.embedder._embed_batch(frame_dets)
                for det, emb in zip(frame_dets, frame_dets):
                    det["embedding"] = emb.get("embedding") if isinstance(emb, dict) else emb
            
            # Draw zone overlay if requested
            if args.zones and frame_dets:
                # Use first detection's zone as representative
                zone = frame_dets[0].get("spatial_zone", "center")
                annotated = draw_zone_overlay(annotated, zone, alpha=0.08)
            
            # Draw depth if requested
            if args.depth and engine.observer.perception_engine:
                # Estimate depth for visualization
                try:
                    depth, _ = engine.observer.perception_engine.depth_estimator.estimate(frame)
                    annotated = draw_depth_heatmap(annotated, depth, alpha=0.25)
                except Exception:
                    pass
            
            # Draw detections
            for i, det in enumerate(frame_dets):
                bbox = det["bounding_box"]
                bbox_int = (int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2))
                label = det["object_class"]
                conf = det["confidence"]
                
                # Assign color by class hash
                color = COLORS[hash(label) % len(COLORS)]
                
                # Draw detection
                track_id = None  # TODO: extract from tracker if available
                annotated = draw_detection(
                    annotated, bbox_int, label, conf, track_id, color
                )
                
                # Track trajectory
                if args.trails:
                    centroid = det["centroid"]
                    entity_key = f"{label}_{i}"  # Simple key
                    trajectories[entity_key].append(centroid)
                    annotated = draw_trajectory(annotated, trajectories[entity_key], color)
        
        # Write frame
        out.write(annotated)
        
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    out.release()
    pbar.close()
    
    print(f"\n[VisEval] ✓ Visualization saved to {args.output}")
    print(f"[VisEval] View with: vlc {args.output}")


if __name__ == "__main__":
    main()
