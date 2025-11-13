#!/usr/bin/env python3
"""
Comparison: Label-based vs Embedding-based Re-ID

Shows why YOLO class names contaminate tracking
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from orion.managers.model_manager import ModelManager
from sklearn.metrics.pairwise import cosine_similarity

video_path = "data/examples/video.mp4"
model_path = "models/yoloe-11s-seg-pf.pt"

yolo = YOLO(model_path)
manager = ModelManager.get_instance()
clip = manager.clip

# Get 3 frames: early, middle, late
cap = cv2.VideoCapture(video_path)
frames_to_test = [50, 100, 150]
frames = []

for frame_idx in frames_to_test:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frames.append((frame_idx, frame))
cap.release()

print("\n" + "="*80)
print("LABEL-BASED vs EMBEDDING-BASED RE-ID COMPARISON")
print("="*80)

# Storage for detections across frames
detections_by_frame = {}

for frame_idx, frame in frames:
    results = yolo(frame, conf=0.35, verbose=False)
    boxes = results[0].boxes
    
    detections = []
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
        
        embedding = clip.encode_image(crop)
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'class': class_name,
            'confidence': float(conf),
            'embedding': embedding,
            'center': ((x1+x2)/2, (y1+y2)/2)
        })
    
    detections_by_frame[frame_idx] = detections

print(f"\nCollected detections from frames: {frames_to_test}")

# Find a stable object (appears in all frames, similar position)
# Let's focus on objects that appear in frame 100

print("\n" + "─"*80)
print("SCENARIO: Track the 'bag' object across frames")
print("─"*80)

frame_100_detections = detections_by_frame[100]

# Find "bag" in frame 100 (YOLO might call it "diaper bag")
bag_det = None
for det in frame_100_detections:
    if 'bag' in det['class'].lower():
        bag_det = det
        break

if not bag_det:
    print("\n❌ No bag detected in frame 100")
else:
    print(f"\n✓ Found in frame 100: '{bag_det['class']}'")
    print(f"  Position: {bag_det['center']}")
    print(f"  Confidence: {bag_det['confidence']:.3f}")
    
    # Try to find it in frame 50 and 150
    print("\n" + "─"*80)
    print("METHOD 1: Label-based matching (UNRELIABLE)")
    print("─"*80)
    
    for other_frame in [50, 150]:
        print(f"\nSearching in frame {other_frame}:")
        detections = detections_by_frame[other_frame]
        
        # Look for objects with same class name
        matches = [d for d in detections if d['class'] == bag_det['class']]
        
        if matches:
            print(f"  ✓ Found {len(matches)} object(s) with class '{bag_det['class']}'")
            for m in matches:
                print(f"    - Position: {m['center']}, Confidence: {m['confidence']:.3f}")
        else:
            print(f"  ❌ No objects with class '{bag_det['class']}' found")
            print(f"  (YOLO might call it something else in this frame!)")
    
    print("\n" + "─"*80)
    print("METHOD 2: Embedding-based matching (RELIABLE)")
    print("─"*80)
    
    for other_frame in [50, 150]:
        print(f"\nSearching in frame {other_frame}:")
        detections = detections_by_frame[other_frame]
        
        # Compare embeddings (visual similarity, ignore labels)
        similarities = []
        for det in detections:
            sim = cosine_similarity(
                bag_det['embedding'].reshape(1, -1),
                det['embedding'].reshape(1, -1)
            )[0][0]
            similarities.append((det, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Top match
        if similarities:
            best_det, best_sim = similarities[0]
            if best_sim > 0.7:  # High similarity threshold
                print(f"  ✓ Found match! (similarity: {best_sim:.3f})")
                print(f"    YOLO class: '{best_det['class']}'")
                print(f"    Position: {best_det['center']}")
                print(f"    Note: Class name might be different, but visually it's the SAME object!")
            else:
                print(f"  ⚠️  Best match has low similarity ({best_sim:.3f})")
                print(f"    Object might not be visible in this frame")
        
        # Show if class name changed
        if similarities and similarities[0][1] > 0.7:
            best_det, _ = similarities[0]
            if best_det['class'] != bag_det['class']:
                print(f"\n  🔍 CLASS NAME INCONSISTENCY:")
                print(f"    Frame 100: '{bag_det['class']}'")
                print(f"    Frame {other_frame}: '{best_det['class']}'")
                print(f"    → Same object, different labels! This is why we can't trust labels.")

print("\n" + "="*80)
print("CONCLUSION:")
print("─"*80)
print("""
Label-based Re-ID (BAD):
  ❌ Fails when YOLO changes its mind about class name
  ❌ "diaper bag" in one frame, "backpack" in another
  ❌ Creates duplicate tracks for same object
  ❌ Contaminates spatial memory and knowledge graph

Embedding-based Re-ID (GOOD):
  ✓ Matches by visual appearance, not class label
  ✓ Robust to YOLO's label inconsistency  
  ✓ Same object = same track, regardless of what YOLO calls it
  ✓ Clean data for downstream systems

RECOMMENDATION:
  Use CLIP embeddings as PRIMARY identity
  Use YOLO labels as WEAK HINTS only (for grouping similar objects)
  Use FastVLM for ACCURATE semantic descriptions (when needed)
  Use USER CORRECTIONS as GROUND TRUTH
""")
print("="*80)
