#!/usr/bin/env python3
"""Debug Observer detection to find where detections are filtered"""

import cv2
import sys
import logging
from pathlib import Path

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from orion.perception.config import DetectionConfig
from orion.perception.observer import FrameObserver
from orion.managers.model_manager import ModelManager

# Load video
video_path = "data/examples/room.mp4"
cap = cv2.VideoCapture(video_path)

# Jump to frame 287 (where we know there are detections)
cap.set(cv2.CAP_PROP_POS_FRAMES, 287)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read frame 287")
    sys.exit(1)

print(f"✓ Frame 287 loaded: {frame.shape}\n")

# Test 1: Direct YOLO
print("=" * 80)
print("TEST 1: Direct YOLO (baseline)")
print("=" * 80)
model = YOLO("yolo11m.pt")
results = model(frame, conf=0.10, verbose=False)
boxes = results[0].boxes
print(f"✓ YOLO detected {len(boxes)} objects directly\n")

# Test 2: Observer with YOLO backend
print("=" * 80)
print("TEST 2: Observer with YOLO backend")
print("=" * 80)

config = DetectionConfig(
    backend="yolo",
    model="yolo11m",
    confidence_threshold=0.10,
    min_object_size=10,  # Lower size filter
    enable_temporal_filtering=False,  # Disable temporal filter
    enable_adaptive_confidence=False,  # Disable adaptive confidence
    enable_depth_validation=False,  # Disable depth validation
)

# Initialize ModelManager
mgr = ModelManager()
yolo_model = mgr.load_yolo(model_name="yolo11m", device="mps")

observer = FrameObserver(
    config=config,
    detector_backend="yolo",
    yolo_model=yolo_model,
    target_fps=2.0,
    show_progress=False,
)

# Detect on frame 287
frame_width, frame_height = frame.shape[1], frame.shape[0]
detections = observer.detect_objects(
    frame=frame,
    frame_number=287,
    timestamp=287/30.0,
    frame_width=frame_width,
    frame_height=frame_height,
)

print(f"\n✓ Observer detected {len(detections)} objects after filtering")

if len(detections) < len(boxes):
    print(f"\n⚠️  Filters removed {len(boxes) - len(detections)} detections!")
    print("\nChecking which filters are active:")
    print(f"  - min_object_size: {config.min_object_size}")
    print(f"  - enable_temporal_filtering: {config.enable_temporal_filtering}")
    print(f"  - enable_adaptive_confidence: {config.enable_adaptive_confidence}")
    print(f"  - enable_depth_validation: {config.enable_depth_validation}")
    print(f"  - max_bbox_area_ratio: {config.max_bbox_area_ratio}")
    print(f"  - max_aspect_ratio: {config.max_aspect_ratio}")

if len(detections) > 0:
    print("\n✓ Sample detections that passed filters:")
    for i, det in enumerate(detections[:5]):
        print(f"  {i+1}. {det['category']}: {det['confidence']:.2f}")
