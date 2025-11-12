#!/usr/bin/env python3
"""Direct YOLO test - bypass pipeline."""

import cv2
import sys
from ultralytics import YOLO
from pathlib import Path

print("📹 Loading video...")
cap = cv2.VideoCapture("data/examples/room.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video")
    exit(1)

print(f"✓ Frame: {frame.shape}, dtype={frame.dtype}")

print("🤖 Loading YOLO11n...")
model = YOLO("yolo11n.pt")
print("✓ YOLO loaded")

# Test inference
print("\n🔍 Running inference...")

# Try different approaches
print("\n1. Standard inference (conf=0.3):")
results = model(frame, conf=0.3, verbose=False)
print(f"   Results type: {type(results)}")
print(f"   Results length: {len(results)}")
if results:
    print(f"   Boxes: {len(results[0].boxes)} detections")

print("\n2. With verbose output:")
results = model(frame, conf=0.01, verbose=True)
print(f"   Boxes: {len(results[0].boxes)} detections")

print("\n3. Direct model call:")
results = model.predict(frame, conf=0.01, verbose=False)
print(f"   Results: {type(results)}, len={len(results)}")
if results:
    boxes = results[0].boxes
    print(f"   Detections: {len(boxes)}")
    if len(boxes) > 0:
        for i, box in enumerate(boxes[:5]):
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"     {cls_name}: {conf:.3f}")

print("\n4. Try with post_process=True:")
results = model(source=frame, conf=0.01, imgsz=640, verbose=False)
if results:
    print(f"   Boxes: {len(results[0].boxes)}")

print("\n✓ Done")
