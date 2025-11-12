#!/usr/bin/env python3
"""Debug YOLO detection on frame 0 with visualization."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

video_path = "data/examples/room.mp4"
if not Path(video_path).exists():
    print(f"❌ Video not found: {video_path}")
    exit(1)

print(f"📹 Extracting frames from {video_path}...")
cap = cv2.VideoCapture(video_path)

# Get first few frames
frames = []
frame_numbers = []
for i in range(5):
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
        frame_numbers.append(i)
    else:
        break
cap.release()

print(f"✓ Extracted {len(frames)} frames")

# Load YOLO
print("\n🔍 Loading YOLO11n...")
yolo = YOLO('models/yolo11x.pt')
print("✓ YOLO loaded")

# Test detection on each frame with different confidence thresholds
print("\n" + "="*70)
print("Testing YOLO Detection on First 5 Frames")
print("="*70)

for frame_num, frame in zip(frame_numbers, frames):
    print(f"\n📊 Frame {frame_num} ({frame.shape[0]}x{frame.shape[1]})")
    
    # Analyze frame
    print(f"  Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
    
    # Test multiple confidence thresholds
    for conf_threshold in [0.1, 0.3, 0.5, 0.7]:
        results = yolo(frame, conf=conf_threshold, verbose=False)
        num_detections = len(results[0].boxes)
        
        print(f"  Conf={conf_threshold}: {num_detections:3d} detections", end="")
        
        if num_detections > 0:
            print(" ✓", end="")
            # Show first few
            for i, box in enumerate(results[0].boxes[:3]):
                cls_id = int(box.cls[0])
                cls_name = yolo.names[cls_id]
                conf = float(box.conf[0])
                print(f"\n    - {cls_name} ({conf:.3f})", end="")
        print()
    
    # Now test with default conf=0.3 and show all
    print(f"\n  Full detection (conf=0.3):")
    results = yolo(frame, conf=0.3, verbose=False)
    detections = results[0].boxes
    
    if len(detections) == 0:
        print(f"    ℹ️  No detections on frame {frame_num}")
        
        # Try with very low confidence
        print(f"    Trying with conf=0.01...")
        results = yolo(frame, conf=0.01, verbose=False)
        if len(results[0].boxes) > 0:
            print(f"    ✓ Found detections with conf=0.01:")
            for box in results[0].boxes[:5]:
                cls_name = yolo.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"      - {cls_name} ({conf:.3f})")
    else:
        print(f"    Found {len(detections)} objects:")
        for box in detections[:10]:
            cls_name = yolo.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"      - {cls_name:15s} ({conf:.3f}) @ ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

print("\n" + "="*70)
print("✓ YOLO Debugging Complete")
print("="*70)
