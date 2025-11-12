#!/usr/bin/env python3
"""Quick YOLO test using pipeline's loaded model."""

import cv2
import sys
from pathlib import Path

# Use the already-initialized pipeline
from full_perception_pipeline import ComprehensivePerceptionPipeline

video_path = "data/examples/room.mp4"

print("📹 Loading pipeline...")
pipeline = ComprehensivePerceptionPipeline(video_path)

print("Extracting frames...")
cap = cv2.VideoCapture(video_path)

print("\nTesting YOLO on first 5 frames:\n")

for frame_num in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    
    print(f"Frame {frame_num}: ", end="", flush=True)
    
    # Test different confidence thresholds
    for conf in [0.01, 0.1, 0.3, 0.5]:
        results = pipeline.yolo(frame, conf=conf, verbose=False)
        num_dets = len(results[0].boxes) if results else 0
        print(f"conf={conf}→{num_dets}  ", end="", flush=True)
    
    print()

cap.release()

# Now extract frame 0 and do detailed analysis
print("\n" + "="*70)
print("Detailed Analysis of Frame 0")
print("="*70)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    print(f"✓ Frame shape: {frame.shape}")
    print(f"  Brightness: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
    
    # Save first frame for inspection
    cv2.imwrite("frame_0.png", frame)
    print(f"  ✓ Saved frame_0.png for inspection")
    
    # Test YOLO with very low threshold
    print(f"\nRunning YOLO with conf=0.01...")
    results = pipeline.yolo(frame, conf=0.01, verbose=False)
    
    if results:
        boxes = results[0].boxes
        print(f"  Found {len(boxes)} detections:")
        
        for i, box in enumerate(boxes[:10]):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = pipeline.yolo.names.get(cls_id, "?")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"    {i+1}. {cls_name:15s} conf={conf:.3f} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    else:
        print("  No results from YOLO")
else:
    print("Failed to read video")
