#!/usr/bin/env python3
"""
Analyze what data we can trust from our detection pipeline
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from orion.managers.model_manager import ModelManager

video_path = "data/examples/video.mp4"
model_path = "models/yoloe-11s-seg-pf.pt"

# Load models
yolo = YOLO(model_path)
manager = ModelManager.get_instance()
clip = manager.clip

# Get frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

# Detect
results = yolo(frame, conf=0.35, verbose=False)
boxes = results[0].boxes
masks = results[0].masks

print("\n" + "="*80)
print("DATA RELIABILITY ANALYSIS")
print("="*80)

print("\n✓ RELIABLE DATA (we can trust this):")
print("─"*80)

for idx, (box, conf, cls_id) in enumerate(zip(
    boxes.xyxy.cpu().numpy(),
    boxes.conf.cpu().numpy(),
    boxes.cls.cpu().numpy()
)):
    x1, y1, x2, y2 = map(int, box)
    class_name = yolo.names[int(cls_id)]
    
    # Get crop
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    
    # RELIABLE: Spatial data
    area = (x2 - x1) * (y2 - y1)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    
    # RELIABLE: Visual embedding (describes appearance, not semantic class)
    embedding = clip.encode_image(crop)
    
    # RELIABLE: Segmentation mask (what pixels are "object" vs "background")
    if masks and idx < len(masks):
        mask = masks[idx].data.cpu().numpy()[0]
        mask_coverage = np.sum(mask > 0.5) / area if area > 0 else 0
    else:
        mask_coverage = 1.0
    
    print(f"\n  Object {idx}:")
    print(f"    Bbox: ({x1}, {y1}, {x2}, {y2})")
    print(f"    Area: {area:.0f} pixels")
    print(f"    Center: ({center_x:.0f}, {center_y:.0f})")
    print(f"    CLIP embedding: 512-dim vector (L2-norm: {np.linalg.norm(embedding):.3f})")
    print(f"    Mask coverage: {mask_coverage:.2f} (how much of bbox is actual object)")
    print(f"    Detection confidence: {conf:.3f}")
    print(f"    ❌ UNRELIABLE class label: '{class_name}' (could be wrong!)")

print("\n" + "="*80)
print("KEY INSIGHT:")
print("─"*80)
print("""
We have EXCELLENT visual data:
  ✓ Precise bounding boxes and segmentation masks
  ✓ High-quality CLIP embeddings (512-dim, captures visual appearance)
  ✓ Spatial positions and sizes
  ✓ Detection confidence scores

But TERRIBLE semantic data:
  ✗ Class labels are unreliable guesses
  ✗ Can't trust "diaper bag", "musical keyboard", "desktop computer"
  ✗ Will contaminate Re-ID, SLAM, and knowledge graphs

SOLUTION:
  Don't use YOLO class names for semantic understanding!
  Use them only as weak hints for initial grouping.
  Build semantic understanding from:
    1. CLIP embeddings (visual similarity)
    2. FastVLM descriptions (visual language model, more accurate)
    3. User corrections ("that's my backpack, not a diaper bag")
    4. Spatial context (on desk → likely work item)
""")
print("="*80)
