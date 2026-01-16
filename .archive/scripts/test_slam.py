#!/usr/bin/env python3
"""
Test SLAM Visual Odometry
=========================

Tests the new SLAM implementation on a video file.
Outputs trajectory visualization and optional point cloud.

Usage:
    python scripts/test_slam.py --video data/examples/test.mp4 --output results/slam_test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Enable MPS fallback for operations not implemented on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.slam import SLAMEngine, SLAMConfig
from orion.perception.depth import DepthEstimator


def run_slam_test(
    video_path: str,
    output_dir: str,
    use_depth: bool = True,
    max_frames: int = 500,
    sample_rate: int = 2,  # Process every Nth frame
    visualize: bool = True
):
    """Run SLAM on a video and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SLAM
    config = SLAMConfig(
        enable_slam=True,
        use_depth=use_depth,
        method="orb_vo",
        num_features=2000,
        build_point_cloud=use_depth,
        point_cloud_subsample=8,  # Subsample for speed
        max_points=200000,
    )
    slam = SLAMEngine(config)
    
    # Initialize depth estimator if needed
    depth_estimator = None
    if use_depth:
        print("Initializing Depth Estimator (Depth Anything V3)...")
        try:
            depth_estimator = DepthEstimator(
                model_name="depth_anything_3",
                model_size="small"
            )
            print("✓ Depth estimator ready")
        except Exception as e:
            print(f"Warning: Could not initialize depth estimator: {e}")
            depth_estimator = None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Processing: every {sample_rate} frames, max {max_frames}")
    
    # Process frames
    frame_idx = 0
    processed = 0
    start_time = time.time()
    
    print("\nProcessing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        if processed >= max_frames:
            break
        
        timestamp = frame_idx / fps
        
        # Estimate depth if available
        depth_map = None
        if depth_estimator is not None:
            try:
                depth_map, _ = depth_estimator.estimate(frame)
            except Exception as e:
                if processed == 0:
                    print(f"Depth estimation failed: {e}")
        
        # Run SLAM
        pose = slam.process_frame(frame, depth_map, timestamp)
        
        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Frame {frame_idx}/{total_frames} ({processed} processed, {elapsed:.1f}s)")
        
        frame_idx += 1
    
    cap.release()
    elapsed = time.time() - start_time
    
    # Get results
    stats = slam.get_stats()
    trajectory = slam.get_trajectory()
    point_cloud = slam.get_point_cloud()
    
    print(f"\n=== SLAM Results ===")
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Keyframes: {stats['num_keyframes']}")
    print(f"Trajectory length: {len(trajectory)} poses")
    print(f"Point cloud: {stats['num_points']} points")
    print(f"Processing time: {elapsed:.1f}s ({processed/elapsed:.1f} fps)")
    
    # Save trajectory
    trajectory_path = output_dir / "trajectory.txt"
    slam.save_trajectory(str(trajectory_path), format="tum")
    
    # Save trajectory as numpy
    np.save(output_dir / "trajectory.npy", trajectory)
    
    # Save point cloud
    if point_cloud is not None and len(point_cloud) > 0:
        slam.save_point_cloud(str(output_dir / "point_cloud.ply"))
    
    # Save stats
    with open(output_dir / "slam_stats.json", 'w') as f:
        # Convert numpy types for JSON serialization
        stats_json = {k: int(v) if isinstance(v, np.integer) else v for k, v in stats.items()}
        json.dump(stats_json, f, indent=2)
    
    # Visualize trajectory
    if visualize and len(trajectory) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY plane (top-down view)
        ax = axes[0]
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=0.5)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory (Top-Down XY)')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        
        # XZ plane (side view)
        ax = axes[1]
        ax.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=0.5)
        ax.plot(trajectory[0, 0], trajectory[0, 2], 'go', markersize=10)
        ax.plot(trajectory[-1, 0], trajectory[-1, 2], 'ro', markersize=10)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('Trajectory (Side View XZ)')
        ax.axis('equal')
        ax.grid(True)
        
        # Translation over time
        ax = axes[2]
        frames = np.arange(len(trajectory))
        ax.plot(frames, trajectory[:, 0], 'r-', label='X', alpha=0.7)
        ax.plot(frames, trajectory[:, 1], 'g-', label='Y', alpha=0.7)
        ax.plot(frames, trajectory[:, 2], 'b-', label='Z', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Position (m)')
        ax.set_title('Position over Time')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "trajectory_plot.png", dpi=150)
        plt.close()
        print(f"\n✓ Trajectory plot saved to {output_dir / 'trajectory_plot.png'}")
    
    # 3D visualization of point cloud
    if visualize and point_cloud is not None and len(point_cloud) > 100:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for visualization
        if len(point_cloud) > 10000:
            idx = np.random.choice(len(point_cloud), 10000, replace=False)
            pts = point_cloud[idx]
        else:
            pts = point_cloud
        
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='viridis', s=0.1, alpha=0.3)
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', linewidth=2, label='Camera')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Cloud with Camera Trajectory')
        ax.legend()
        
        plt.savefig(output_dir / "point_cloud_3d.png", dpi=150)
        plt.close()
        print(f"✓ Point cloud visualization saved to {output_dir / 'point_cloud_3d.png'}")
    
    print(f"\n✓ All results saved to {output_dir}")
    return trajectory, point_cloud


def main():
    parser = argparse.ArgumentParser(description="Test SLAM on video")
    parser.add_argument("--video", type=str, default="data/examples/test.mp4",
                        help="Path to video file")
    parser.add_argument("--output", type=str, default="results/slam_test",
                        help="Output directory")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable depth estimation")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Maximum frames to process")
    parser.add_argument("--sample-rate", type=int, default=2,
                        help="Process every Nth frame")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization")
    
    args = parser.parse_args()
    
    run_slam_test(
        video_path=args.video,
        output_dir=args.output,
        use_depth=not args.no_depth,
        max_frames=args.max_frames,
        sample_rate=args.sample_rate,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()
