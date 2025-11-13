#!/usr/bin/env python3
"""
Phase 2: Visual Re-ID Demo with Gemini Ground Truth
- Visualize tracked objects with bounding boxes and track IDs
- Highlight Re-ID events (when object reappears after occlusion)
- Query Gemini for ground truth object descriptions
- Save annotated frames showing tracking results
"""

import cv2
import sys
import os
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import time
from typing import List, Dict, Tuple


def visualize_tracking_with_reid(
    video_path: str,
    yolo,
    clip,
    output_dir: str = "results/reid_visualization",
    num_frames: int = 300,
    skip_frames: int = 5,
):
    """
    Visualize tracking with Re-ID highlighting
    
    Args:
        video_path: Path to input video
        yolo: YOLO model instance
        clip: CLIP model instance
        output_dir: Directory to save annotated frames
        num_frames: Number of frames to process
        skip_frames: Frame skip interval
    """
    from orion.perception.enhanced_tracker import EnhancedTracker
    
    print("="*80)
    print("PHASE 2: VISUAL Re-ID DEMONSTRATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = EnhancedTracker(
        max_age=15,
        min_hits=2,
        iou_threshold=0.3,
        appearance_threshold=0.55,
        max_gallery_size=10,
        ema_alpha=0.85,
    )
    
    print(f"\nTracker Configuration:")
    print(f"  max_age: {tracker.max_age} frames")
    print(f"  min_hits: {tracker.min_hits} detections")
    print(f"  appearance_threshold: {tracker.appearance_threshold}")
    
    # Color palette for track IDs (consistent colors per ID)
    np.random.seed(42)
    colors = {}
    
    def get_color(track_id: int) -> Tuple[int, int, int]:
        """Get consistent BGR color for track ID"""
        if track_id not in colors:
            colors[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
        return colors[track_id]
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    processed_frames = 0
    
    # Track history for Re-ID detection
    track_last_seen = {}  # track_id -> last frame_idx
    reid_events = []  # List of (frame_idx, track_id, gap_frames)
    track_history = defaultdict(list)  # track_id -> [(frame_idx, bbox), ...]
    
    print(f"\nProcessing {num_frames} frames (skip={skip_frames})...")
    print(f"Output directory: {output_dir}")
    
    saved_frames = []
    
    while processed_frames < num_frames:
        # Skip frames
        for _ in range(skip_frames):
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = yolo(frame, verbose=False)
        
        # Build detections
        detections = []
        embeddings = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = box
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                
                embedding = clip.encode_image(crop)
                embeddings.append(embedding)
                
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    'bbox_2d': box,
                    'bbox_3d': np.array([cx, cy, 1000.0, w, h, 100.0]),
                    'class_name': yolo.names[int(cls)],
                    'confidence': float(conf),
                    'depth_mm': 1000.0,
                })
        
        # Update tracker
        tracks = tracker.update(
            detections, 
            embeddings if embeddings else None, 
            camera_pose=None, 
            frame_idx=frame_idx
        )
        
        # Detect Re-ID events
        current_track_ids = {t.id for t in tracks if t.hits >= tracker.min_hits}
        for track in tracks:
            if track.hits < tracker.min_hits:
                continue
            
            tid = track.id
            
            # Check if this is a re-identification (track reappearing after gap)
            if tid in track_last_seen:
                gap = frame_idx - track_last_seen[tid]
                if gap > tracker.max_age // 2:  # Significant gap
                    reid_events.append({
                        'frame': frame_idx,
                        'track_id': tid,
                        'class': track.class_name,
                        'gap': gap,
                        'bbox': track.bbox_2d,
                    })
                    print(f"  → Re-ID Event: Track {tid} ({track.class_name}) reappeared after {gap} frames")
            
            track_last_seen[tid] = frame_idx
            track_history[tid].append((frame_idx, track.bbox_2d.copy()))
        
        # Annotate frame
        annotated = frame.copy()
        
        # Draw tracks
        for track in tracks:
            if track.hits < tracker.min_hits:
                continue
            
            tid = track.id
            color = get_color(tid)
            x1, y1, x2, y2 = map(int, track.bbox_2d)
            
            # Check if this is a recent Re-ID event
            is_reid = any(
                e['track_id'] == tid and abs(e['frame'] - frame_idx) < 3
                for e in reid_events
            )
            
            # Draw bounding box (thicker for Re-ID events)
            thickness = 4 if is_reid else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track ID and class
            label = f"ID:{tid} {track.class_name}"
            if is_reid:
                label += " [RE-ID]"
            
            # Background for text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Draw track history (trajectory)
            if tid in track_history and len(track_history[tid]) > 1:
                pts = []
                for hist_frame, hist_bbox in track_history[tid][-20:]:  # Last 20 positions
                    cx = int((hist_bbox[0] + hist_bbox[2]) / 2)
                    cy = int((hist_bbox[1] + hist_bbox[3]) / 2)
                    pts.append((cx, cy))
                
                if len(pts) > 1:
                    pts = np.array(pts, dtype=np.int32)
                    cv2.polylines(annotated, [pts], False, color, 2)
        
        # Add frame info overlay
        info_text = f"Frame: {frame_idx} | Tracks: {len([t for t in tracks if t.hits >= tracker.min_hits])} | Re-ID Events: {len(reid_events)}"
        cv2.putText(
            annotated, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        # Save key frames (Re-ID events, every 50th frame, first/last)
        should_save = (
            processed_frames == 0 or
            processed_frames == num_frames - 1 or
            processed_frames % 50 == 0 or
            any(abs(e['frame'] - frame_idx) < 2 for e in reid_events[-5:])  # Recent Re-ID
        )
        
        if should_save:
            output_path = f"{output_dir}/frame_{frame_idx:05d}.jpg"
            cv2.imwrite(output_path, annotated)
            saved_frames.append(frame_idx)
            print(f"  Saved: {output_path}")
        
        processed_frames += 1
        frame_idx += skip_frames + 1
        
        if processed_frames % 50 == 0:
            print(f"  Progress: {processed_frames}/{num_frames} frames processed...")
    
    cap.release()
    
    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Frames processed: {processed_frames}")
    print(f"Frames saved: {len(saved_frames)}")
    print(f"Re-ID events detected: {len(reid_events)}")
    print(f"Unique tracks: {len(track_history)}")
    print(f"Output directory: {output_dir}")
    
    # Re-ID summary
    if reid_events:
        print("\nRe-ID Event Summary:")
        class_counts = Counter(e['class'] for e in reid_events)
        print(f"  By class: {dict(class_counts)}")
        avg_gap = np.mean([e['gap'] for e in reid_events])
        print(f"  Average gap: {avg_gap:.1f} frames")
        print(f"\n  Recent Re-ID events:")
        for event in reid_events[-10:]:
            print(f"    Frame {event['frame']:5d}: Track {event['track_id']:2d} ({event['class']:10s}) - gap: {event['gap']} frames")
    
    return saved_frames, reid_events, track_history


def query_gemini_for_objects(video_path: str, sample_frames: List[int] = [100, 500, 1000]):
    """Query Gemini API for object descriptions in frames"""
    import google.generativeai as genai
    from PIL import Image
    
    print("\n" + "="*80)
    print("GEMINI GROUND TRUTH COMPARISON")
    print("="*80)
    
    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not found in environment")
        return {}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    print(f"\nQuerying Gemini for objects in frames: {sample_frames}")
    
    cap = cv2.VideoCapture(video_path)
    gemini_results = {}
    
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {frame_idx}: Could not read")
            continue
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Query Gemini
        prompt = """List all visible objects in this image. For each object, provide:
1. Object name
2. Approximate location (left/right/center, top/bottom)
3. Brief description

Format as a structured list."""
        
        try:
            response = model.generate_content([prompt, pil_image])
            gemini_results[frame_idx] = response.text
            print(f"\n  Frame {frame_idx}:")
            print(f"    {response.text[:200]}...")
        except Exception as e:
            print(f"  Frame {frame_idx}: Gemini query failed - {e}")
            gemini_results[frame_idx] = None
    
    cap.release()
    
    return gemini_results


if __name__ == "__main__":
    video_path = "data/examples/video.mp4"
    
    # Load models
    print("Loading models...")
    from ultralytics import YOLO
    from orion.managers.model_manager import ModelManager
    
    yolo = YOLO("yolo11s-seg.pt")
    manager = ModelManager.get_instance()
    clip = manager.clip
    
    print("✓ Models loaded\n")
    
    # Phase 2A: Visualize tracking with Re-ID highlights
    saved_frames, reid_events, track_history = visualize_tracking_with_reid(
        video_path,
        yolo,
        clip,
        output_dir="results/reid_visualization",
        num_frames=300,
        skip_frames=5,
    )
    
    # Phase 2B: Query Gemini for ground truth
    gemini_results = query_gemini_for_objects(
        video_path,
        sample_frames=[100, 500, 1000, 1500]
    )
    
    # Compare YOLO vs Gemini
    print("\n" + "="*80)
    print("DETECTION COMPARISON: YOLO vs Gemini")
    print("="*80)
    
    if gemini_results:
        print("\nGemini provided ground truth for object verification")
        print("Review saved frames in results/reid_visualization/")
    else:
        print("\nGemini API not available - using YOLO detections only")
    
    print("\n✓ Phase 2 Complete!")
