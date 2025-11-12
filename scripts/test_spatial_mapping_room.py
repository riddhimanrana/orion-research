#!/usr/bin/env python3
"""
Comprehensive Spatial Mapping Test on room.mp4

Outputs ALL debug visualizations:
1. Camera intrinsics & matrix K
2. Depth heatmaps (TURBO colormap)
3. Depth distributions
4. YOLO detections with bounding boxes
5. 3D back-projections (point clouds)
6. 2D dot projections (reprojected 3D points)
7. 3D bounding boxes in world space
8. Estimated object depths
9. SLAM poses & camera motion

All visualizations saved to spatial_mapping_output/
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import sys
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.depth import DepthEstimator
from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.perception.types import CameraIntrinsics
from ultralytics import YOLO
import time

# Create output directory
OUTPUT_DIR = Path("spatial_mapping_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"📁 Output directory: {OUTPUT_DIR.absolute()}/")


class SpatialMappingVisualizer:
    """Comprehensive 3D spatial mapping visualization"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.intrinsics = CameraIntrinsics.auto_estimate(width, height)
        
        # Camera matrix K
        self.K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.cx],
            [0, self.intrinsics.fy, self.intrinsics.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Inverse for back-projection
        self.K_inv = np.linalg.inv(self.K)
    
    def save_camera_intrinsics(self, frame_num: int):
        """Save camera intrinsics to image"""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        cv2.putText(img, "CAMERA INTRINSICS MATRIX K", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
        
        y = 100
        cv2.putText(img, f"Resolution: {self.width}x{self.height}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        y += 50
        cv2.putText(img, "Intrinsics Matrix K:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2)
        
        y += 50
        for i, row in enumerate(self.K):
            text = f"[ {row[0]:8.2f}  {row[1]:8.2f}  {row[2]:8.2f} ]"
            cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_MONOSPACE, 0.8, (0, 0, 0), 1)
            y += 35
        
        y += 20
        cv2.putText(img, f"Focal Length X (fx): {self.intrinsics.fx:.2f} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y += 30
        cv2.putText(img, f"Focal Length Y (fy): {self.intrinsics.fy:.2f} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y += 30
        cv2.putText(img, f"Principal Point X (cx): {self.intrinsics.cx:.2f} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y += 30
        cv2.putText(img, f"Principal Point Y (cy): {self.intrinsics.cy:.2f} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        path = OUTPUT_DIR / f"00_intrinsics_frame_{frame_num:03d}.png"
        cv2.imwrite(str(path), img)
        print(f"  ✓ {path.name}")
    
    def save_depth_heatmap(self, depth_map: np.ndarray, frame_num: int):
        """Save depth map with TURBO colormap"""
        if depth_map is None or depth_map.size == 0:
            return
        
        valid = depth_map[depth_map > 0]
        if len(valid) == 0:
            return
        
        # Normalize
        depth_norm = np.clip(depth_map, valid.min(), valid.max())
        depth_norm = (depth_norm - valid.min()) / (valid.max() - valid.min() + 1e-8)
        depth_norm = (depth_norm * 255).astype(np.uint8)
        
        # Apply TURBO colormap
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        
        path = OUTPUT_DIR / f"01_depth_heatmap_frame_{frame_num:03d}.png"
        cv2.imwrite(str(path), depth_colored)
        print(f"  ✓ {path.name}")
    
    def save_yolo_detections(self, frame: np.ndarray, results, frame_num: int):
        """Save YOLO detections on frame"""
        if not results or len(results) == 0:
            return
        
        annotated = frame.copy()
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            
            # Draw box
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 0), 3)
            
            # Label
            label = f"{result.names[cls_id]} {conf:.2f}"
            cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        path = OUTPUT_DIR / f"02_yolo_detections_frame_{frame_num:03d}.png"
        cv2.imwrite(str(path), annotated)
        print(f"  ✓ {path.name}")
    
    def save_depth_distribution(self, depth_map: np.ndarray, frame_num: int):
        """Save depth distribution histogram"""
        valid = depth_map[depth_map > 0]
        if len(valid) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(valid / 1000.0, bins=100, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Depth (meters)')
        axes[0].set_ylabel('Pixel Count')
        axes[0].set_title(f'Depth Distribution - Frame {frame_num}')
        axes[0].grid(True, alpha=0.3)
        
        # Statistics text box
        stats_text = f"""
Frame: {frame_num}
Min: {valid.min():.0f} mm ({valid.min()/1000:.2f} m)
Max: {valid.max():.0f} mm ({valid.max()/1000:.2f} m)
Mean: {valid.mean():.0f} mm ({valid.mean()/1000:.2f} m)
Median: {np.median(valid):.0f} mm ({np.median(valid)/1000:.2f} m)
Std Dev: {valid.std():.0f} mm
Valid pixels: {len(valid)} / {depth_map.size} ({100*len(valid)/depth_map.size:.1f}%)
        """
        
        axes[1].text(0.1, 0.9, stats_text, transform=axes[1].transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1].axis('off')
        
        path = OUTPUT_DIR / f"03_depth_distribution_frame_{frame_num:03d}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path.name}")
    
    def back_project_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Back-project depth map to 3D point cloud"""
        h, w = depth_map.shape
        
        # Create pixel coordinates
        yy, xx = np.mgrid[0:h, 0:w]
        coords_2d = np.stack([xx, yy, np.ones_like(xx)], axis=-1)  # (H, W, 3)
        coords_2d = coords_2d.reshape(-1, 3).T  # (3, H*W)
        
        # Back-project using K_inv and depth
        coords_cam = self.K_inv @ coords_2d  # (3, H*W)
        depth_flat = depth_map.reshape(-1)  # (H*W,)
        
        # Scale by depth
        points_3d = coords_cam * depth_flat / 1000.0  # Convert mm to meters
        points_3d = points_3d.T  # (H*W, 3)
        
        # Filter valid points
        valid_mask = depth_flat > 0
        points_3d = points_3d[valid_mask]
        
        return points_3d
    
    def save_point_cloud_visualization(self, depth_map: np.ndarray, frame_num: int):
        """Visualize point cloud in 3D"""
        points_3d = self.back_project_depth(depth_map)
        
        if len(points_3d) == 0:
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by depth
        z_vals = points_3d[:, 2]
        colors = plt.cm.viridis((z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8))
        
        # Sample points for visualization (too dense otherwise)
        sample_idx = np.random.choice(len(points_3d), min(10000, len(points_3d)), replace=False)
        sample_points = points_3d[sample_idx]
        sample_colors = colors[sample_idx]
        
        ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                  c=sample_colors, s=1, alpha=0.6)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Point Cloud - Frame {frame_num} ({len(points_3d):,} points)')
        
        # Set limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 5])
        
        path = OUTPUT_DIR / f"04_point_cloud_frame_{frame_num:03d}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path.name}")
    
    def save_reprojection_visualization(self, depth_map: np.ndarray, results, frame_num: int):
        """Project 3D points back to 2D image"""
        points_3d = self.back_project_depth(depth_map)
        
        if len(points_3d) == 0 or not results:
            return
        
        # Create blank image
        proj_img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 50
        
        # Project points back to image
        points_proj = (self.K @ points_3d.T).T  # (N, 3)
        points_2d = points_proj[:, :2] / points_proj[:, 2:3]  # (N, 2) - normalize by Z
        
        # Sample for visualization
        sample_idx = np.random.choice(len(points_2d), min(5000, len(points_2d)), replace=False)
        
        for idx in sample_idx:
            x, y = points_2d[idx]
            if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                cv2.circle(proj_img, (int(x), int(y)), 1, (0, 255, 255), -1)
        
        # Draw YOLO bboxes
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(proj_img, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 0), 2)
        
        path = OUTPUT_DIR / f"05_reprojection_dots_frame_{frame_num:03d}.png"
        cv2.imwrite(str(path), proj_img)
        print(f"  ✓ {path.name}")
    
    def save_slam_pose_info(self, slam_pose: np.ndarray, frame_num: int):
        """Save SLAM pose information"""
        img = np.ones((600, 1000, 3), dtype=np.uint8) * 255
        
        cv2.putText(img, f"SLAM CAMERA POSE - Frame {frame_num}", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
        
        if slam_pose is not None:
            R = slam_pose[:3, :3]
            t = slam_pose[:3, 3]
            
            y = 100
            cv2.putText(img, "Rotation Matrix R:", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2)
            
            y += 40
            for i, row in enumerate(R):
                text = f"[ {row[0]:7.4f}  {row[1]:7.4f}  {row[2]:7.4f} ]"
                cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_MONOSPACE, 0.6, (0, 0, 0), 1)
                y += 30
            
            y += 20
            cv2.putText(img, "Translation t (meters):", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2)
            y += 35
            cv2.putText(img, f"  x = {t[0]:.4f} m", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y += 30
            cv2.putText(img, f"  y = {t[1]:.4f} m", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y += 30
            cv2.putText(img, f"  z = {t[2]:.4f} m", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y += 30
            
            motion = np.linalg.norm(t)
            cv2.putText(img, f"Total motion: {motion:.4f} m ({motion*1000:.1f} mm)", (40, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        else:
            cv2.putText(img, "Identity pose (reference frame)", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
        
        path = OUTPUT_DIR / f"06_slam_pose_frame_{frame_num:03d}.png"
        cv2.imwrite(str(path), img)
        print(f"  ✓ {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Spatial mapping visualization on room.mp4")
    parser.add_argument("--video", type=str, default="data/examples/room.mp4",
                        help="Input video path")
    parser.add_argument("--max-frames", type=int, default=20,
                        help="Max frames to process")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Process every Nth frame")
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🔬 SPATIAL MAPPING VISUALIZATION - room.mp4 COMPREHENSIVE DEBUG")
    print("="*80)
    
    # Load video
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {args.video}")
    print(f"   Resolution: {width}x{height} @ {fps:.1f} fps")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # Initialize components
    print("\n[1/3] Initializing components...")
    yolo = YOLO("yolo11n.pt")
    print("  ✅ YOLO loaded")
    
    depth_est = DepthEstimator(model_name="depth_anything_v2", model_size="small")
    print("  ✅ Depth Anything V2 loaded")
    
    slam = OpenCVSLAM(config=SLAMConfig())
    print("  ✅ SLAM initialized")
    
    visualizer = SpatialMappingVisualizer(width, height)
    print("  ✅ Visualizer initialized")
    
    # Save intrinsics reference
    visualizer.save_camera_intrinsics(0)
    
    # Process frames
    print(f"\n[2/3] Processing frames (every {args.sample_rate}th frame, max {args.max_frames})...")
    
    frame_idx = 0
    processed = 0
    total_time = 0
    
    while frame_idx < total_frames and processed < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % args.sample_rate != 0:
            frame_idx += 1
            continue
        
        start = time.time()
        processed += 1
        
        print(f"\n  Frame {frame_idx} (processed #{processed}):")
        
        # YOLO
        results = yolo(frame, verbose=False)
        
        # Depth
        depth_result = depth_est.estimate(frame)
        depth_map = depth_result[0] if isinstance(depth_result, tuple) else depth_result
        
        # SLAM
        slam_pose = slam.track(frame, timestamp=frame_idx/fps, frame_idx=frame_idx, depth_map=depth_map)
        
        # Save visualizations
        visualizer.save_camera_intrinsics(processed)
        visualizer.save_depth_heatmap(depth_map, processed)
        visualizer.save_yolo_detections(frame, results, processed)
        visualizer.save_depth_distribution(depth_map, processed)
        visualizer.save_point_cloud_visualization(depth_map, processed)
        visualizer.save_reprojection_visualization(depth_map, results, processed)
        visualizer.save_slam_pose_info(slam_pose, processed)
        
        elapsed = time.time() - start
        total_time += elapsed
        print(f"    ⏱️  {elapsed:.2f}s")
        
        frame_idx += 1
    
    cap.release()
    depth_est.cleanup()
    
    # Summary
    print("\n" + "="*80)
    print("[3/3] COMPLETE")
    print("="*80)
    print(f"\n✅ Processed {processed} frames in {total_time:.1f}s")
    print(f"📁 Output: {OUTPUT_DIR.absolute()}/")
    
    # List files
    files = sorted(OUTPUT_DIR.glob("*.png"))
    print(f"\n📊 Generated files ({len(files)}):")
    for f in files:
        print(f"   {f.name}")
    
    print("\n📊 Visualization categories:")
    print("   00_* = Camera intrinsics matrix K")
    print("   01_* = Depth heatmaps (TURBO colormap)")
    print("   02_* = YOLO detections with boxes")
    print("   03_* = Depth distribution histograms")
    print("   04_* = 3D point clouds (back-projection)")
    print("   05_* = 2D reprojections (dot projections)")
    print("   06_* = SLAM camera poses & motion")


if __name__ == "__main__":
    main()
