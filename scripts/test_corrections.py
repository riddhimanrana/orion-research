#!/usr/bin/env python3
"""Quick detection test with class corrections"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Class corrections
CORRECTIONS = {
    'musical keyboard': 'keyboard',
    'laptop keyboard': 'keyboard',  
    'desktop computer': 'laptop',
    'tv': 'monitor',
    'diaper bag': 'bag',
}

video_path = "data/examples/video.mp4"
model_path = "models/yoloe-11s-seg-pf.pt"

yolo = YOLO(model_path)
print(f"✓ Model: {model_path}\n")

# Get frame 100
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

# Detect
results = yolo(frame, conf=0.35, verbose=False)
boxes = results[0].boxes

print(f"Detections ({len(boxes)}):")
print("─" * 60)

for box, conf, cls_id in zip(
    boxes.xyxy.cpu().numpy(),
    boxes.conf.cpu().numpy(),
    boxes.cls.cpu().numpy()
):
    original = yolo.names[int(cls_id)]
    corrected = CORRECTIONS.get(original, original)
    
    if corrected != original:
        print(f"  {corrected:20s} (was: {original:20s}) | conf={conf:.3f}")
    else:
        print(f"  {corrected:20s} {' '*26} | conf={conf:.3f}")

print("\n✓ Class corrections applied successfully!")
