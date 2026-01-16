#!/usr/bin/env python3
"""Quick test: run YOLO on a single frame"""

import cv2
import sys
from pathlib import Path

# Extract frame from video
video_path = "data/examples/room.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"❌ Could not read frame from {video_path}")
    sys.exit(1)

print(f"✓ Frame loaded: {frame.shape}")
cv2.imwrite("/tmp/test_frame.jpg", frame)
print("✓ Saved /tmp/test_frame.jpg")

# Test YOLO detection
from orion.perception.detectors.yolo import YOLODetector
from orion.perception.config import DetectionConfig

config = DetectionConfig(
    backend="yolo",
    model="yolo11m",
    confidence_threshold=0.10  # Lower threshold
)

print(f"\n🔬 Testing YOLO detector...")
print(f"   Model: {config.model}")
print(f"   Confidence: {config.confidence_threshold}")
print(f"   Device: mps")

detector = YOLODetector(config, device="mps")
observations = detector.detect(frame)

print(f"\n✓ YOLO detected {len(observations)} objects:")
for obs in observations[:10]:  # Show first 10
    print(f"   - {obs.category}: {obs.confidence:.2f} at {obs.bbox}")

if len(observations) == 0:
    print("\n⚠️  Zero detections! Possible issues:")
    print("   1. Frame is blank/corrupted")
    print("   2. YOLO model issue")
    print("   3. Device mismatch (MPS vs CPU)")
    print("\n   Checking frame statistics:")
    import numpy as np
    print(f"   Frame shape: {frame.shape}")
    print(f"   Frame dtype: {frame.dtype}")
    print(f"   Frame range: [{frame.min()}, {frame.max()}]")
    print(f"   Frame mean: {frame.mean():.1f}")
