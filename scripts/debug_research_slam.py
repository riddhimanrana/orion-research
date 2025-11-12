#!/usr/bin/env python3
"""
Research SLAM Debug Visualization
Outputs: Camera intrinsics, depth heatmaps, spatial maps, point clouds, 3D visualizations
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent))

from orion.perception.depth import DepthEstimator
from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.perception.types import CameraIntrinsics
from ultralytics import YOLO
import time

# Create output directory for debug images
DEBUG_OUTPUT_DIR = Path("debug_output")
DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)


def print_camera_intrinsics(width: int, height: int):
    """Print estimated camera intrinsics"""
    intrinsics = CameraIntrinsics.auto_estimate(width, height)
    
    print("\n" + "="*80)
    print("📷 CAMERA INTRINSICS")
    print("="*80)
    print(f"Resolution: {width}x{height}")
    print(f"Focal Length X: {intrinsics.fx:.2f} px")
    print(f"Focal Length Y: {intrinsics.fy:.2f} px")
    print(f"Principal Point X: {intrinsics.cx:.2f} px")
    print(f"Principal Point Y: {intrinsics.cy:.2f} px")
    print(f"\nIntrinsics Matrix K:")
    K = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1]
    ])
    for row in K:
        print(f"  [{row[0]:8.2f}, {row[1]:8.2f}, {row[2]:8.2f}]")
    return intrinsics


def print_depth_heatmap_stats(depth_map: np.ndarray, frame_num: int):
    """Print depth map statistics"""
    valid_depth = depth_map[depth_map > 0]
    
    if len(valid_depth) == 0:
        print(f"⚠️  Frame {frame_num}: No valid depth data")
        return
    
    print("\n" + "-"*80)
    print(f"🔍 DEPTH ANYTHING V2 - Frame {frame_num}")
    print("-"*80)
    print(f"Shape: {depth_map.shape}")
    print(f"Data type: {depth_map.dtype}")
    print(f"Valid pixels: {len(valid_depth)} / {depth_map.size} ({len(valid_depth)*100/depth_map.size:.1f}%)")
    print(f"Min depth: {valid_depth.min():.0f} mm ({valid_depth.min()/1000:.2f} m)")
    print(f"Max depth: {valid_depth.max():.0f} mm ({valid_depth.max()/1000:.2f} m)")
    print(f"Mean depth: {valid_depth.mean():.0f} mm ({valid_depth.mean()/1000:.2f} m)")
    print(f"Median depth: {np.median(valid_depth):.0f} mm ({np.median(valid_depth)/1000:.2f} m)")
    print(f"Std dev: {valid_depth.std():.0f} mm")
    
    # Histogram
    bins = [100, 500, 1000, 2000, 5000, 10000]
    print(f"\nDepth distribution:")
    prev = 0
    for b in bins:
        count = np.sum((valid_depth >= prev) & (valid_depth < b))
        pct = count * 100 / len(valid_depth)
        print(f"  {prev:5d}-{b:5d} mm: {count:4d} pixels ({pct:5.1f}%)")
        prev = b
    count = np.sum(valid_depth >= prev)
    pct = count * 100 / len(valid_depth)
    print(f"  {prev:5d}+     mm: {count:4d} pixels ({pct:5.1f}%)")


def print_slam_state(slam_result: np.ndarray, frame_num: int):
    """Print SLAM output"""
    print(f"\n📍 SLAM TRACKING - Frame {frame_num}")
    print("-"*80)
    if slam_result is not None:
        print(f"Camera pose matrix (4x4):")
        print(f"  Shape: {slam_result.shape}")
        print(f"  Type: {slam_result.dtype}")
        
        # Extract components
        R = slam_result[:3, :3]
        t = slam_result[:3, 3]
        
        print(f"\nRotation matrix R:")
        for row in R:
            print(f"  {row}")
        
        print(f"\nTranslation vector t (meters):")
        print(f"  x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}")
        
        # Compute pose metrics
        translation_magnitude = np.linalg.norm(t)
        print(f"\nCamera movement: {translation_magnitude:.4f} m")
        
        # Orientation
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        euler = rot.as_euler('xyz', degrees=True)
        print(f"Rotation (Euler XYZ): {euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°")
    else:
        print("⚠️  No SLAM pose (initialization frame)")


def print_yolo_detections(results, frame_num: int, width: int, height: int):
    """Print YOLO detection statistics"""
    print(f"\n🎯 YOLO DETECTIONS - Frame {frame_num}")
    print("-"*80)
    
    if not results or len(results) == 0:
        print("⚠️  No objects detected")
        return
    
    result = results[0]
    boxes = result.boxes
    
    print(f"Total detections: {len(boxes)}")
    print(f"\nTop detections:")
    
    for i, box in enumerate(boxes[:5]):  # Show top 5
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        width_px = x2 - x1
        height_px = y2 - y1
        
        print(f"  {i+1}. Class: {result.names[cls_id]} | "
              f"Conf: {conf:.2f} | "
              f"Box: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)}) | "
              f"Size: {int(width_px)}x{int(height_px)} px")
    
    if len(boxes) > 5:
        print(f"  ... and {len(boxes)-5} more")


def save_depth_heatmap(depth_map: np.ndarray, frame_num: int):
    """Save depth map as heatmap image"""
    if depth_map is None:
        return
    
    # Normalize for visualization
    valid_depth = depth_map[depth_map > 0]
    if len(valid_depth) == 0:
        return
    
    # Normalize to 0-255 range
    depth_normalized = np.clip(depth_map, valid_depth.min(), valid_depth.max())
    depth_normalized = ((depth_normalized - valid_depth.min()) / 
                        (valid_depth.max() - valid_depth.min()) * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    
    # Save
    output_path = DEBUG_OUTPUT_DIR / f"depth_frame_{frame_num:03d}.png"
    cv2.imwrite(str(output_path), depth_colored)
    print(f"  ✓ Saved: {output_path}")


def save_yolo_annotations(frame: np.ndarray, results, frame_num: int):
    """Save frame with YOLO annotations"""
    if not results or len(results) == 0:
        return
    
    # Draw on frame
    annotated_frame = frame.copy()
    result = results[0]
    boxes = result.boxes
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        
        # Draw box
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 2)
        
        # Draw label
        label = f"{result.names[cls_id]} {conf:.2f}"
        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save
    output_path = DEBUG_OUTPUT_DIR / f"yolo_frame_{frame_num:03d}.png"
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"  ✓ Saved: {output_path}")


def save_depth_distribution_chart(depth_map: np.ndarray, frame_num: int):
    """Save depth distribution as visual histogram"""
    valid_depth = depth_map[depth_map > 0]
    if len(valid_depth) == 0:
        return
    
    # Create histogram image
    hist_height, hist_width = 300, 800
    hist_img = np.ones((hist_height, hist_width, 3), dtype=np.uint8) * 255
    
    # Calculate histogram
    bins = np.linspace(valid_depth.min(), valid_depth.max(), 100)
    hist, bin_edges = np.histogram(valid_depth, bins=bins)
    hist = hist / hist.max() * (hist_height - 40)  # Normalize
    
    # Draw bars
    bar_width = hist_width / len(hist)
    for i, count in enumerate(hist):
        x1 = int(i * bar_width)
        x2 = int((i + 1) * bar_width)
        y1 = hist_height - 20
        y2 = hist_height - 20 - int(count)
        cv2.rectangle(hist_img, (x1, y2), (x2, y1), (0, 165, 255), -1)
    
    # Add labels
    cv2.putText(hist_img, f"Depth Distribution - Frame {frame_num}", 
               (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(hist_img, f"Min: {valid_depth.min():.0f}mm | "
               f"Mean: {valid_depth.mean():.0f}mm | "
               f"Max: {valid_depth.max():.0f}mm",
               (20, hist_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save
    output_path = DEBUG_OUTPUT_DIR / f"depth_hist_frame_{frame_num:03d}.png"
    cv2.imwrite(str(output_path), hist_img)
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Research SLAM Debug Visualization")
    parser.add_argument("--video", type=str, default="data/examples/video_short.mp4",
                        help="Input video path")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames to process")
    parser.add_argument("--yolo-model", type=str, default="yolo11n",
                        help="YOLO model size")
    
    args = parser.parse_args()
    video_path = args.video
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🔬 ORION RESEARCH SLAM - COMPREHENSIVE DEBUG OUTPUT")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Frames to process: {args.frames}")
    print(f"YOLO model: {args.yolo_model}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nVideo specs: {width}x{height} @ {fps:.1f} fps")
    
    # Print camera intrinsics
    intrinsics = print_camera_intrinsics(width, height)
    
    # Initialize components
    print("\n" + "="*80)
    print("[1/5] INITIALIZING COMPONENTS")
    print("="*80)
    
    print("Loading YOLO...")
    yolo = YOLO(f"{args.yolo_model}.pt")
    print(f"✅ YOLO11 {args.yolo_model.upper()} loaded")
    
    print("Loading Depth Estimator (MiDaS v2)...")
    depth_estimator = DepthEstimator(model_name="midas")
    print("✅ Depth estimator loaded")
    
    print("Initializing SLAM...")
    slam_config = SLAMConfig()
    slam = OpenCVSLAM(config=slam_config)
    print("✅ SLAM initialized")
    
    # Process frames
    print("\n" + "="*80)
    print("[2/5] PROCESSING VIDEO FRAMES")
    print("="*80)
    
    frame_count = 0
    total_time = 0
    
    while frame_count < args.frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        print(f"\n{'█'*80}")
        print(f"FRAME {frame_count + 1}/{args.frames}")
        print(f"{'█'*80}")
        
        # YOLO Detection
        print("\n[Step 1] YOLO Detection...")
        yolo_results = yolo(frame, verbose=False)
        print_yolo_detections(yolo_results, frame_count + 1, width, height)
        
        # Depth Estimation
        print("\n[Step 2] Depth Estimation (Anything-V2)...")
        depth_result = depth_estimator.estimate(frame)
        if isinstance(depth_result, tuple):
            depth_map, depth_vis = depth_result
        else:
            depth_map = depth_result
        print_depth_heatmap_stats(depth_map, frame_count + 1)
        
        # SLAM Tracking
        print("\n[Step 3] SLAM Tracking...")
        timestamp = frame_count / fps
        slam_result = slam.track(frame, timestamp=timestamp, frame_idx=frame_count, depth_map=depth_map)
        print_slam_state(slam_result, frame_count + 1)
        
        # Export images
        print(f"\n[Step 4] Exporting debug images...")
        save_depth_heatmap(depth_map, frame_count + 1)
        save_depth_distribution_chart(depth_map, frame_count + 1)
        save_yolo_annotations(frame, yolo_results, frame_count + 1)
        
        frame_time = time.time() - start_time
        total_time += frame_time
        print(f"\n⏱️  Frame processing time: {frame_time:.2f}s")
        
        frame_count += 1
    
    cap.release()
    
    # Summary
    print("\n" + "="*80)
    print("[3/5] PROCESSING COMPLETE")
    print("="*80)
    print(f"Frames processed: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {frame_count / total_time:.2f}")
    
    print("\n" + "="*80)
    print("✅ DEBUG OUTPUT COMPLETE")
    print("="*80)
    print(f"\nGenerated data in {DEBUG_OUTPUT_DIR}/:")
    print(f"  • depth_frame_*.png - Depth heatmaps (TURBO colormap)")
    print(f"  • depth_hist_frame_*.png - Depth distribution histograms")
    print(f"  • yolo_frame_*.png - YOLO detections with annotations")
    print(f"  • Console output - Camera intrinsics, statistics, SLAM poses")
    print(f"\nTotal files exported: {(len(list(DEBUG_OUTPUT_DIR.glob('*.png'))))}")


if __name__ == "__main__":
    main()
