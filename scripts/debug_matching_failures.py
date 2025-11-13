#!/usr/bin/env python3
"""
Debug: Why are matches failing for overlapping objects?
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
    min_hits=2,  # Lower to see tracks sooner
    ema_alpha=0.9,
    clip_model=manager.clip
)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

print("="*80)
print("MATCHING DEBUG - Why do IoU=0.97 objects get different track IDs?")
print("="*80)

for frame_idx in range(100, 103):  # Just 3 frames
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
    
    print(f"\nFrame {frame_idx}:")
    print(f"  Detections: {len(detections)}")
    print(f"  Existing tracks: {len(tracker.tracks)}")
    
    # Show which tracks are confirmed
    confirmed = [t for t in tracker.tracks if t.hits >= 2]
    print(f"  Confirmed tracks (hits≥2): {len(confirmed)}")
    for t in confirmed[:5]:
        print(f"    Track {t.id}: {t.class_name} hits={t.hits}")
    
    # Update
    matched_tracks = tracker.update(detections, embeddings, frame_idx=frame_idx)
    print(f"  After update: {len(tracker.tracks)} total, {len(matched_tracks)} confirmed")

cap.release()

print("\n" + "="*80)
print("FINAL ANALYSIS")
print("="*80)

# Find duplicate monitors
monitors = [t for t in tracker.tracks if 'monitor' in tracker._normalize_label(t.class_name).lower()]
print(f"\nMonitor tracks: {len(monitors)}")
for m in monitors:
    bbox = m.bbox_2d
    print(f"  Track {m.id}: hits={m.hits}  bbox=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")

if len(monitors) >= 2:
    print("\nChecking IoU between first 2 monitors:")
    b1 = monitors[0].bbox_2d
    b2 = monitors[1].bbox_2d
    
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    
    if x_right > x_left and y_bottom > y_top:
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        iou = intersection / (area1 + area2 - intersection)
        print(f"  Track {monitors[0].id} ∩ Track {monitors[1].id}: IoU={iou:.3f}")
        
        if iou > 0.5:
            print(f"\n  ❌ PROBLEM: These should have been matched!")
            print(f"     Track {monitors[0].id} created at frame {monitors[0].age}")
            print(f"     Track {monitors[1].id} created at frame {monitors[1].age}")

print("="*80)
