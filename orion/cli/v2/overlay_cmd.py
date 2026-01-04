"""orion overlay - Render video overlay with tracks and scene info"""

import json
import logging
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def run_overlay(args) -> int:
    """Render video overlay with tracking visualization."""
    
    episode_dir = Path("results") / args.episode
    meta_path = episode_dir / "episode_meta.json"
    
    if not meta_path.exists():
        print(f"Episode not found: {args.episode}")
        return 1
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    video_path = meta["video_path"]
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return 1
    
    print(f"\n  OVERLAY RENDERING")
    print(f"  Video: {video_path}")
    print(f"  Style: {args.style}")
    print("  " + "─" * 60)
    
    # Load tracks
    tracks_path = episode_dir / "tracks_filtered.jsonl"
    if not tracks_path.exists():
        tracks_path = episode_dir / "tracks.jsonl"
    
    observations = []
    if tracks_path.exists():
        with open(tracks_path) as f:
            for line in f:
                if line.strip():
                    observations.append(json.loads(line))
    
    # Group by frame
    obs_by_frame = defaultdict(list)
    for obs in observations:
        obs_by_frame[obs["frame_id"]].append(obs)
    
    print(f"  Loaded {len(observations)} observations across {len(obs_by_frame)} frames")
    
    # Load VLM scene captions if available
    vlm_scenes = {}
    vlm_path = episode_dir / "vlm_scene.jsonl"
    if vlm_path.exists():
        with open(vlm_path) as f:
            for line in f:
                if line.strip():
                    scene = json.loads(line)
                    vlm_scenes[scene["frame_id"]] = scene.get("scene_caption", "")
    
    # Load memory for object labels
    memory_labels = {}
    memory_path = episode_dir / "memory.json"
    if memory_path.exists():
        with open(memory_path) as f:
            memory = json.load(f)
        for obj in memory.get("filtered_objects", memory.get("objects", [])):
            memory_labels[obj["id"]] = obj.get("canonical_label", "object")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output path
    output_path = args.output or str(episode_dir / f"overlay_{args.style}.mp4")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate colors for tracks
    track_colors = {}
    def get_color(track_id):
        if track_id not in track_colors:
            np.random.seed(hash(track_id) % 2**32)
            track_colors[track_id] = tuple(int(c) for c in np.random.randint(100, 255, 3))
        return track_colors[track_id]
    
    # Process frames
    frame_idx = 0
    processed = 0
    current_caption = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check for observations at this frame
        if frame_idx in obs_by_frame:
            obs_list = obs_by_frame[frame_idx]
            
            for obs in obs_list:
                track_id = obs.get("memory_object_id", obs["track_id"])
                label = memory_labels.get(track_id, obs.get("label", "object"))
                bbox = obs.get("bbox")
                conf = obs.get("confidence", 1.0)
                color = get_color(track_id)
                
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    if args.style == "v1":
                        # Simple boxes
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label}", (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    elif args.style == "v2":
                        # Boxes with confidence
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        text = f"{label} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1-th-4), (x1+tw+4, y1), color, -1)
                        cv2.putText(frame, text, (x1+2, y1-2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
                    elif args.style == "v3":
                        # Pseudo-3D style
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw ID badge
                        id_short = track_id[:8] if isinstance(track_id, str) else str(track_id)
                        (tw, th), _ = cv2.getTextSize(id_short, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
                        cv2.putText(frame, id_short, (x1+3, y1-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                        
                        # Draw label below
                        cv2.putText(frame, label, (x1, y2+15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            processed += 1
        
        # Update caption from VLM scenes
        if frame_idx in vlm_scenes:
            current_caption = vlm_scenes[frame_idx]
        
        # Draw caption at bottom
        if current_caption and args.style in ["v2", "v3"]:
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, height-60), (width, height), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Draw caption text
            cv2.putText(frame, current_caption[:100], (10, height-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # Draw frame counter
        cv2.putText(frame, f"F:{frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n  ✓ Rendered {frame_idx} frames ({processed} with detections)")
    print(f"  Saved to: {output_path}")
    
    return 0
