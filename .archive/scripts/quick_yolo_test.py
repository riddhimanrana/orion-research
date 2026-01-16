#!/usr/bin/env python3
"""Quick YOLO-World detection test."""
from ultralytics import YOLO
import cv2
import sys

video = sys.argv[1] if len(sys.argv) > 1 else "data/examples/test.mp4"
vocab_bg = ["monitor", "desk", "chair", "keyboard", "laptop", "picture frame", "bottle", "mouse", "person", ""]

cap = cv2.VideoCapture(video)
print(f"Testing {video}")

for model_name in ["yolov8m-worldv2.pt", "yolov8l-worldv2.pt"]:
    print(f"\n=== {model_name} ===")
    for fid in [10, 100, 500]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        model = YOLO(model_name)
        model.set_classes(vocab_bg)
        results = model.predict(frame, conf=0.10, verbose=False)[0]
        
        dets = sorted(
            [(vocab_bg[int(b.cls[0])], round(float(b.conf[0]), 2)) for b in results.boxes],
            key=lambda x: -x[1]
        )
        print(f"  Frame {fid:4d}: {dets[:5]}")

cap.release()
print("\nDone!")
