#!/usr/bin/env python3
"""Test YOLO directly via ultralytics to verify model works"""

import cv2
from ultralytics import YOLO

# Load video and extract frame
video_path = "data/examples/room.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"❌ Could not read frame from {video_path}")
    exit(1)

print(f"✓ Frame loaded: {frame.shape}")

# Test YOLO directly
print("\n🔬 Testing YOLO11m directly via ultralytics...")
model = YOLO("yolo11m.pt")
results = model(frame, conf=0.10, verbose=False)

boxes = results[0].boxes
print(f"\n✓ YOLO detected {len(boxes)} objects:")

if len(boxes) == 0:
    print("  ⚠️  Zero detections!")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame dtype: {frame.dtype}")
    print(f"  Frame range: [{frame.min()}, {frame.max()}]")
    print(f"  Frame mean: {frame.mean():.1f}")
else:
    for i, box in enumerate(boxes[:10]):  # Show first 10
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        label = model.names[cls]
        print(f"  {i+1}. {label}: {conf:.2f} at {[int(x) for x in bbox]}")
