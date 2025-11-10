#!/usr/bin/env python3
"""
Quick test of Rerun.io visualization

Tests the Rerun logger with minimal dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rerun as rr
import numpy as np

print("Testing Rerun.io integration...")

# Initialize Rerun
rr.init("orion-test", spawn=True)

print("✓ Rerun initialized - viewer should open in browser!")
print("\nLogging test data...")

# Log some test points
for i in range(10):
    rr.set_time_sequence("frame", i)
    
    # Random 3D points
    points = np.random.rand(20, 3) * 10
    colors = np.random.rand(20, 3) * 255
    
    rr.log(
        "test/points",
        rr.Points3D(points, colors=colors, radii=0.2)
    )
    
    # Camera trajectory
    camera_pos = np.array([i * 0.5, 0, 0])
    rr.log(
        "test/camera",
        rr.Transform3D(translation=camera_pos)
    )
    
    # Trajectory line
    if i > 0:
        trajectory = np.array([[j * 0.5, 0, 0] for j in range(i+1)])
        rr.log(
            "test/trajectory",
            rr.LineStrips3D([trajectory], colors=[0, 255, 0])
        )
    
    # Metrics
    rr.log("metrics/frame", rr.Scalar(i))
    rr.log("metrics/points", rr.Scalar(20))

print("\n✅ Test complete! Check the Rerun viewer in your browser.")
print("You should see:")
print("  • Random 3D points (test/points)")
print("  • Camera moving along X axis (test/camera)")
print("  • Green trajectory line (test/trajectory)")
print("  • Metrics plots (metrics/*)")
print("\nPress Ctrl+C to exit")

# Keep script alive so viewer stays open
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nExiting...")
