#!/usr/bin/env python3
"""
Rerun visualization of complete spatial mapping from room.mp4
Shows depth maps, 3D point clouds, YOLO detections, and camera trajectory
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

try:
    import rerun as rr
except ImportError:
    print("❌ Rerun SDK not installed!")
    print("Install with: pip install rerun-sdk")
    sys.exit(1)

# Initialize Rerun
rr.init("Spatial Mapping - room.mp4", spawn=True)

OUTPUT_DIR = Path("/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/spatial_mapping_output")

print("\n" + "="*80)
print("🎬 LOADING SPATIAL MAPPING DATA INTO RERUN")
print("="*80)

# Load all images grouped by frame
frames_data = defaultdict(dict)
for img_path in sorted(OUTPUT_DIR.glob("*.png")):
    parts = img_path.stem.split("_")
    category = f"{parts[0]}_{parts[1]}"  # e.g., "00_intrinsics"
    frame_num = int(parts[-1])
    frames_data[frame_num][category] = img_path

total_frames = max(frames_data.keys()) + 1
print(f"\n📊 Loading {total_frames} frames...")

# Log camera intrinsics (from first frame if available)
if 1 in frames_data and "00_intrinsics" in frames_data[1]:
    print(f"✓ Camera intrinsics reference image")
    # Camera parameters (auto-estimated)
    rr.log("Camera/Intrinsics", rr.TextLog("""
Camera Matrix K:
  fx = 1334.50 pixels
  fy = 1334.50 pixels
  cx = 540 pixels
  cy = 960 pixels
  
Resolution: 1080 × 1920 (portrait)
FOV: H=40.2°, V=50.3°
Type: Smartphone camera model
"""))

# Log each frame's data
for frame_idx in sorted(frames_data.keys()):
    if frame_idx == 0:
        continue  # Skip intrinsics reference frame
    
    frame_data = frames_data[frame_idx]
    print(f"  Frame {frame_idx}/{total_frames-1}...", end="\r")
    
    # Depth heatmap
    if "01_depth" in frame_data:
        try:
            depth_img = cv2.imread(str(frame_data["01_depth"]))
            if depth_img is not None:
                depth_rgb = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
                rr.log(f"Depth/Heatmap/{frame_idx:03d}", rr.Image(depth_rgb))
        except Exception as e:
            pass
    
    # YOLO detections
    if "02_yolo" in frame_data:
        try:
            yolo_img = cv2.imread(str(frame_data["02_yolo"]))
            if yolo_img is not None:
                yolo_rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
                rr.log(f"Objects/YOLO_Detections/{frame_idx:03d}", rr.Image(yolo_rgb))
        except Exception as e:
            pass
    
    # Depth distribution
    if "03_depth" in frame_data:
        try:
            dist_img = cv2.imread(str(frame_data["03_depth"]))
            if dist_img is not None:
                dist_rgb = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
                rr.log(f"Depth/Distribution/{frame_idx:03d}", rr.Image(dist_rgb))
        except Exception as e:
            pass
    
    # 3D Point Cloud
    if "04_point" in frame_data:
        try:
            pc_img = cv2.imread(str(frame_data["04_point"]))
            if pc_img is not None:
                pc_rgb = cv2.cvtColor(pc_img, cv2.COLOR_BGR2RGB)
                rr.log(f"3D/PointCloud/{frame_idx:03d}", rr.Image(pc_rgb))
        except Exception as e:
            pass
    
    # 2D Reprojection
    if "05_reprojection" in frame_data:
        try:
            reproj_img = cv2.imread(str(frame_data["05_reprojection"]))
            if reproj_img is not None:
                reproj_rgb = cv2.cvtColor(reproj_img, cv2.COLOR_BGR2RGB)
                rr.log(f"Geometry/Reprojection/{frame_idx:03d}", rr.Image(reproj_rgb))
        except Exception as e:
            pass

print(f"\n✅ Loaded {total_frames-1} frames into Rerun")
print("\n" + "="*80)
print("🚀 RERUN VIEWER LAUNCHED")
print("="*80)
print("""
Explore the spatial mapping:

LEFT PANEL - Navigation:
  ✓ Depth/Heatmap - Colored depth maps (Red=near, Blue=far)
  ✓ Objects/YOLO_Detections - Detected objects with boxes
  ✓ Depth/Distribution - Statistical histograms
  ✓ 3D/PointCloud - 3D point cloud representations
  ✓ Geometry/Reprojection - 2D verification dots

CONTROLS:
  • Click on any frame number to navigate
  • Use timeline at bottom to scrub through sequence
  • Click on category names to show/hide
  • Use mouse to pan/zoom in 3D views

WHAT TO LOOK FOR:
  1. Depth continuity across frames
  2. Object detection consistency
  3. 3D geometry coherence
  4. Point cloud spatial structure
  5. Re-projection alignment accuracy

═══════════════════════════════════════════════════════════════════════════════

Press Ctrl+C in terminal when done.
""")

# Keep running
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n✅ Rerun visualization closed")
