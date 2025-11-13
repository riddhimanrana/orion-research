"""
analyze_with_visualizations.py
==============================

Run the perception pipeline with SLAM and tracking, then generate comprehensive visualizations:
- 3D camera trajectory from SLAM
- Camera frustums showing field of view
- Object tracking events (detections, re-identifications)
- Timeline of object visibility
- Spatial distribution of objects

Usage:
    python scripts/analyze_with_visualizations.py --video data/examples/video.mp4 --mode accurate

Author: Orion Research Team
Date: November 2025
"""
import argparse
import sys
import logging
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.config import get_accurate_config, get_fast_config, get_balanced_config
from orion.perception.engine import PerceptionEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline_with_export(video_path: str, mode: str = "accurate"):
    """Run perception pipeline and export visualization data."""
    logger.info(f"Running pipeline in {mode} mode with visualization export...")
    
    # Get config
    if mode == "fast":
        config = get_fast_config()
    elif mode == "balanced":
        config = get_balanced_config()
    else:
        config = get_accurate_config()
    
    # Ensure 3D + tracking + SLAM are enabled
    config.enable_3d = True
    config.enable_tracking = True
    
    # Run pipeline
    engine = PerceptionEngine(config)
    result = engine.process_video(video_path, save_visualizations=True, output_dir="results")
    
    logger.info(f"Pipeline complete: {result.unique_entities} entities, {result.total_detections} detections")
    return result


def plot_camera_frustum(ax, pose, K, scale=0.3, color='cyan', alpha=0.3):
    """Draw a camera frustum at the given pose (4x4) using intrinsics K."""
    w, h = K['width'], K['height']
    fx, fy = K['fx'], K['fy']
    cx, cy = K['cx'], K['cy']
    z = scale
    
    # Camera frustum corners in camera frame
    corners = np.array([
        [(0-cx)*z/fx, (0-cy)*z/fy, z],
        [(w-cx)*z/fx, (0-cy)*z/fy, z],
        [(w-cx)*z/fx, (h-cy)*z/fy, z],
        [(0-cx)*z/fx, (h-cy)*z/fy, z],
        [0, 0, 0]
    ])
    
    # Transform to world
    R = pose[:3, :3]
    t = pose[:3, 3]
    corners_w = (R @ corners.T).T + t
    
    # Draw pyramid lines
    for i in range(4):
        ax.plot([corners_w[i,0], corners_w[4,0]], 
                [corners_w[i,1], corners_w[4,1]], 
                [corners_w[i,2], corners_w[4,2]], 
                color=color, alpha=alpha, linewidth=0.5)
    for i in range(4):
        ax.plot([corners_w[i,0], corners_w[(i+1)%4,0]], 
                [corners_w[i,1], corners_w[(i+1)%4,1]], 
                [corners_w[i,2], corners_w[(i+1)%4,2]], 
                color=color, alpha=alpha, linewidth=0.5)


def visualize_slam_trajectory(output_dir: str = "results"):
    """Generate 3D visualization of SLAM trajectory with camera frustums."""
    logger.info("Generating SLAM trajectory visualization...")
    
    # Load data
    traj_file = Path(output_dir) / "slam_trajectory.npy"
    intrinsics_file = Path(output_dir) / "camera_intrinsics.json"
    entities_file = Path(output_dir) / "entities.json"
    
    if not traj_file.exists():
        logger.warning(f"SLAM trajectory not found at {traj_file}")
        return
    
    traj = np.load(traj_file)
    with open(intrinsics_file, 'r') as f:
        K = json.load(f)
    with open(entities_file, 'r') as f:
        entities = json.load(f)
    
    # Extract camera positions
    if traj.ndim == 3:  # (N, 4, 4) poses
        xyz = traj[:, :3, 3]
    else:  # (N, 3) positions
        xyz = traj
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', linewidth=2, label='Camera Path', alpha=0.8)
    ax1.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color='green', s=100, marker='o', label='Start')
    ax1.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color='red', s=100, marker='x', label='End')
    
    # Plot camera frustums at intervals
    if traj.ndim == 3:
        interval = max(1, len(traj) // 15)
        for i in range(0, len(traj), interval):
            plot_camera_frustum(ax1, traj[i], K, scale=0.2, color='cyan', alpha=0.4)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Camera Trajectory with Frustums')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top-down view (XY plane)
    ax2 = fig.add_subplot(222)
    ax2.plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=2, alpha=0.8)
    ax2.scatter(xyz[0, 0], xyz[0, 1], color='green', s=100, marker='o', label='Start')
    ax2.scatter(xyz[-1, 0], xyz[-1, 1], color='red', s=100, marker='x', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (XY Plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Side view (XZ plane)
    ax3 = fig.add_subplot(223)
    ax3.plot(xyz[:, 0], xyz[:, 2], 'b-', linewidth=2, alpha=0.8)
    ax3.scatter(xyz[0, 0], xyz[0, 2], color='green', s=100, marker='o', label='Start')
    ax3.scatter(xyz[-1, 0], xyz[-1, 2], color='red', s=100, marker='x', label='End')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ Plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Entity timeline
    ax4 = fig.add_subplot(224)
    for i, entity in enumerate(entities['entities']):
        first = entity['first_frame']
        last = entity['last_frame']
        ax4.barh(i, last - first, left=first, height=0.8, 
                label=f"{entity['class']} ({entity['observation_count']} obs)")
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Entity ID')
    ax4.set_title('Object Tracking Timeline')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    output_file = Path(output_dir) / "slam_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization: {output_file}")
    
    plt.show()


def print_summary(output_dir: str = "results"):
    """Print summary statistics from the run."""
    entities_file = Path(output_dir) / "entities.json"
    
    if not entities_file.exists():
        logger.warning(f"Entities file not found at {entities_file}")
        return
    
    with open(entities_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("PERCEPTION & TRACKING SUMMARY")
    print("="*80)
    print(f"Total Entities: {data['total_entities']}")
    
    if 'tracking_stats' in data:
        stats = data['tracking_stats']
        print(f"Total Tracks: {stats.get('total_tracks', 'N/A')}")
        print(f"Confirmed Tracks: {stats.get('confirmed_tracks', 'N/A')}")
        print(f"ID Switches: {stats.get('id_switches', 'N/A')}")
    
    print("\nEntity Details:")
    print("-" * 80)
    for entity in data['entities']:
        print(f"  [{entity['id']}] {entity['class']}: "
              f"{entity['observation_count']} observations "
              f"(frames {entity['first_frame']}-{entity['last_frame']}, "
              f"conf={entity['confidence']:.2f})")
        if entity['description'] != "No description":
            print(f"      Description: {entity['description'][:100]}...")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run pipeline with comprehensive visualizations')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--mode', type=str, default='accurate', choices=['fast', 'balanced', 'accurate'],
                       help='Processing mode (default: accurate)')
    parser.add_argument('--skip-pipeline', action='store_true', 
                       help='Skip pipeline run and only generate visualizations from existing data')
    args = parser.parse_args()
    
    # Run pipeline
    if not args.skip_pipeline:
        result = run_pipeline_with_export(args.video, args.mode)
    
    # Generate visualizations
    visualize_slam_trajectory()
    
    # Print summary
    print_summary()


if __name__ == '__main__':
    main()
