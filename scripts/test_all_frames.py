#!/usr/bin/env python3
"""Simple test: Check frame 0 vs frame 287"""

import cv2
from ultralytics import YOLO

video_path = "data/examples/room.mp4"
model = YOLO("yolo11m.pt")

print("Testing YOLO11m on room.mp4:")
print("=" * 80)

for frame_idx in [0, 100, 287, 500, 800, 1000]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        results = model(frame, conf=0.10, verbose=False)
        boxes = results[0].boxes
        print(f"Frame {frame_idx:4d}: {len(boxes):2d} detections", end="")
        if len(boxes) > 0:
            # Show top 3
            top_3 = []
            for i in range(min(3, len(boxes))):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                top_3.append(f"{model.names[cls]}:{conf:.2f}")
            print(f"  ({', '.join(top_3)})")
        else:
            print("  (empty frame?)")

print("=" * 80)
print("\n✓ Frame 0 likely empty/dark (common in videos)")
print("✓ Observer should skip initial dark frames or start from first good frame")
