#!/usr/bin/env python3
"""Scan full video for YOLO detections."""

import cv2
from ultralytics import YOLO
from collections import Counter

print("📹 Loading video...")
cap = cv2.VideoCapture("data/examples/room.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✓ Video has {total_frames} frames")

print("\n🤖 Loading YOLO11n...")
model = YOLO("yolo11n.pt")
print("✓ YOLO loaded\n")

# Scan frames
frame_num = 0
detections_per_frame = []
all_classes = Counter()
frames_with_detections = 0

print("Scanning video for detections...")
print("Frame | Detections | Classes Found")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame, conf=0.3, verbose=False)
    boxes = results[0].boxes
    num_dets = len(boxes)
    
    if num_dets > 0:
        frames_with_detections += 1
        classes = [model.names[int(box.cls[0])] for box in boxes]
        all_classes.update(classes)
        
        # Print progress
        if frame_num % 50 == 0 or num_dets > 0:
            print(f"{frame_num:4d}  | {num_dets:10d} | {', '.join(set(classes))}")
    
    detections_per_frame.append(num_dets)
    frame_num += 1
    
    # Print progress every 100 frames
    if frame_num % 100 == 0:
        print(f"  ... processed {frame_num} frames ...")

cap.release()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total frames: {frame_num}")
print(f"Frames with detections: {frames_with_detections} ({100*frames_with_detections/frame_num:.1f}%)")
print(f"Total detections: {sum(detections_per_frame)}")
print(f"Average detections/frame: {sum(detections_per_frame)/frame_num:.2f}")
print(f"Max detections in single frame: {max(detections_per_frame)}")

print(f"\nObject classes found ({len(all_classes)} unique):")
for cls_name, count in all_classes.most_common(20):
    print(f"  {cls_name:20s}: {count:5d} detections")

# Check first frames
print(f"\nFirst 10 frames detection count:")
for i in range(min(10, len(detections_per_frame))):
    print(f"  Frame {i}: {detections_per_frame[i]} objects")
