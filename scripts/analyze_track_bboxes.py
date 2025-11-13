#!/usr/bin/env python3
"""
Analyze: Are these really duplicates or multiple objects in scene?
"""

import cv2
import numpy as np
from ultralytics import YOLO
from orion.managers.model_manager import ModelManager
from orion.perception.enhanced_tracker import EnhancedTracker
from collections import defaultdict

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

for frame_idx in range(100, 121):
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
    
    # Update tracker
    tracks = tracker.update(detections, embeddings, frame_idx=frame_idx)

cap.release()

print("="*80)
print("TRACK BBOX ANALYSIS")
print("="*80)

# Group by normalized class
tracks_by_class = defaultdict(list)
for track in tracker.tracks:
    if track.hits >= 2:
        normalized = tracker._normalize_label(track.class_name)
        tracks_by_class[normalized].append(track)

for class_name, class_tracks in sorted(tracks_by_class.items()):
    print(f"\n{class_name.upper()} ({len(class_tracks)} tracks):")
    print("─"*80)
    
    for track in sorted(class_tracks, key=lambda t: t.id):
        bbox = track.bbox_2d
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        print(f"  Track {track.id:2d}: hits={track.hits:2d}  "
              f"center=({center_x:4d},{center_y:4d})  "
              f"size={width:4d}x{height:4d}  "
              f"label={track.class_name}")
    
    # Check if spatially overlapping
    if len(class_tracks) > 1:
        print(f"\n  Overlap analysis:")
        for i, t1 in enumerate(class_tracks):
            for t2 in class_tracks[i+1:]:
                b1 = t1.bbox_2d
                b2 = t2.bbox_2d
                
                # IoU
                x_left = max(b1[0], b2[0])
                y_top = max(b1[1], b2[1])
                x_right = min(b1[2], b2[2])
                y_bottom = min(b1[3], b2[3])
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = intersection / (area1 + area2 - intersection)
                    
                    if iou > 0.1:
                        print(f"    Track {t1.id} ∩ Track {t2.id}: IoU={iou:.3f} ← DUPLICATE?")
                else:
                    # Check distance between centers
                    c1 = np.array([(b1[0]+b1[2])/2, (b1[1]+b1[3])/2])
                    c2 = np.array([(b2[0]+b2[2])/2, (b2[1]+b2[3])/2])
                    dist = np.linalg.norm(c1 - c2)
                    print(f"    Track {t1.id} ↔ Track {t2.id}: dist={dist:.0f}px")

print("\n" + "="*80)
