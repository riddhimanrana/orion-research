#!/usr/bin/env python3
"""
Debug why spatial-semantic matching still creates duplicates
"""

import cv2
import numpy as np
from ultralytics import YOLO
from orion.managers.model_manager import ModelManager
from orion.perception.enhanced_tracker import EnhancedTracker

video_path = "data/examples/video.mp4"
yolo = YOLO("models/yoloe-11s-seg-pf.pt")
manager = ModelManager.get_instance()

tracker = EnhancedTracker(
    iou_threshold=0.3,
    appearance_threshold=0.5,
    max_age=30,
    min_hits=2,
    ema_alpha=0.9,
    clip_model=manager.clip
)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

print("="*80)
print("SPATIAL COST DEBUG")
print("="*80)

for frame_idx in range(100, 103):  # Just first 3 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    results = yolo(frame, conf=0.35, verbose=False)
    boxes = results[0].boxes
    
    # Build detections
    detections = []
    embeddings = []
    for box, conf, cls_id in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        class_name = yolo.names[int(cls_id)]
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        embedding = manager.clip.encode_image(crop, normalize=True)
        
        detections.append({
            'bbox_3d': np.array([x1, y1, 500, x2-x1, y2-y1, 100], dtype=np.float32),
            'bbox_2d': np.array([x1, y1, x2, y2], dtype=int),
            'class_name': class_name,
            'confidence': float(conf),
            'depth_mm': 500.0
        })
        embeddings.append(embedding)
    
    print(f"\nFrame {frame_idx}: {len(detections)} detections")
    
    # Before update, show cost matrix if we have tracks
    if len(tracker.tracks) > 0:
        print(f"  Existing tracks: {len(tracker.tracks)}")
        
        # Manually compute cost for first few
        frame_diag = np.sqrt(1920**2 + 1080**2)
        
        for t_idx in range(min(3, len(tracker.tracks))):
            track = tracker.tracks[t_idx]
            print(f"\n  Track {track.id} ({track.class_name}):")
            
            for d_idx in range(min(3, len(detections))):
                det = detections[d_idx]
                
                # Spatial
                det_center = np.array([
                    (det['bbox_2d'][0] + det['bbox_2d'][2]) / 2,
                    (det['bbox_2d'][1] + det['bbox_2d'][3]) / 2
                ], dtype=np.float64)
                track_center = np.array([
                    float(track.bbox_3d[0]), float(track.bbox_3d[1])
                ], dtype=np.float64)
                
                if track.velocity is not None:
                    track_center = track_center + track.velocity[:2].astype(np.float64)
                
                spatial_dist = np.linalg.norm(det_center - track_center)
                spatial_cost = min(1.0, spatial_dist / frame_diag)
                
                # Size
                det_area = (det['bbox_2d'][2] - det['bbox_2d'][0]) * \
                          (det['bbox_2d'][3] - det['bbox_2d'][1])
                track_area = track.bbox_3d[3] * track.bbox_3d[4]
                
                size_ratio = det_area / track_area if track_area > 0 else 1.0
                if 0.5 < size_ratio < 2.0:
                    size_cost = abs(1.0 - size_ratio)
                else:
                    size_cost = 1.0
                
                # Semantic
                semantic_cost = 0.0 if det['class_name'] == track.class_name else 1.0
                
                total_cost = 0.5*spatial_cost + 0.2*size_cost + 0.25*semantic_cost + 0.05*0.5
                
                match_str = "✓ MATCH" if total_cost < 0.6 else "✗ NO MATCH"
                print(f"    vs Det {d_idx} ({det['class_name']}): spatial={spatial_cost:.3f} size={size_cost:.3f} semantic={semantic_cost:.3f} → total={total_cost:.3f} {match_str}")
    
    # Update tracker
    tracks = tracker.update(detections, embeddings, frame_idx=frame_idx)
    print(f"  → {len(tracks)} confirmed tracks")

cap.release()

print("\n" + "="*80)
print(f"FINAL: {len(tracker.tracks)} total tracks")
for track in tracker.tracks[:10]:
    print(f"  Track {track.id}: {track.class_name} (hits={track.hits})")
print("="*80)
