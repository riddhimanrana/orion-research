"""
visualize_slam_and_tracking.py
==============================

Visualizes SLAM camera trajectory, camera intrinsics, and object tracking events (re-tracking, ID switches) in 3D.

Usage:
    python scripts/visualize_slam_and_tracking.py --slam-results results/slam_trajectory.npy --tracking results/entities.json --intrinsics results/camera_intrinsics.json

Features:
- 3D plot of camera trajectory (from SLAM)
- Camera frustum visualization at intervals (using intrinsics)
- Markers for object re-tracking events (ID switches, re-identifications)
- Optionally, plot object 3D positions and tracks
- Timeline plot of object visibility and re-tracking

Author: Orion Research Team
Date: November 2025
"""
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_trajectory(path):
    return np.load(path)  # shape: (N, 3) or (N, 4, 4)

def load_intrinsics(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_tracking(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_camera_frustum(ax, pose, K, scale=0.2, color='b'):
    """Draws a camera frustum at the given pose (4x4) using intrinsics K."""
    # Camera frustum corners in camera frame
    w, h = 640, 480  # Default, override with K if available
    fx, fy = K['fx'], K['fy']
    cx, cy = K['cx'], K['cy']
    z = scale
    corners = np.array([
        [ (0-cx)*z/fx, (0-cy)*z/fy, z],
        [ (w-cx)*z/fx, (0-cy)*z/fy, z],
        [ (w-cx)*z/fx, (h-cy)*z/fy, z],
        [ (0-cx)*z/fx, (h-cy)*z/fy, z],
        [0,0,0]
    ])
    # Transform to world
    R = pose[:3,:3]
    t = pose[:3,3]
    corners_w = (R @ corners.T).T + t
    # Draw lines
    for i in range(4):
        ax.plot([corners_w[i,0], corners_w[4,0]], [corners_w[i,1], corners_w[4,1]], [corners_w[i,2], corners_w[4,2]], color)
    for i in range(4):
        ax.plot([corners_w[i,0], corners_w[(i+1)%4,0]], [corners_w[i,1], corners_w[(i+1)%4,1]], [corners_w[i,2], corners_w[(i+1)%4,2]], color)

def main(args):
    traj = load_trajectory(args.slam_results)  # (N, 4, 4) or (N, 3)
    K = load_intrinsics(args.intrinsics)
    tracking = load_tracking(args.tracking)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    if traj.ndim == 3:
        xyz = traj[:, :3, 3]
    else:
        xyz = traj
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], label='Camera Trajectory', color='blue')

    # Plot camera frustums at intervals
    if traj.ndim == 3:
        for i in range(0, len(traj), max(1, len(traj)//20)):
            plot_camera_frustum(ax, traj[i], K, scale=0.2, color='cyan')

    # Plot object re-tracking events
    for entity in tracking.get('entities', []):
        if 'retrack_events' in entity:
            for evt in entity['retrack_events']:
                idx = evt['frame_idx']
                if idx < len(xyz):
                    ax.scatter(xyz[idx,0], xyz[idx,1], xyz[idx,2], color='red', marker='o', s=60, label='Re-tracked' if i==0 else None)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('SLAM Camera Trajectory & Object Re-Tracking')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize SLAM trajectory and object tracking events.')
    parser.add_argument('--slam-results', type=str, required=True, help='Path to SLAM trajectory (npy, (N,4,4) or (N,3))')
    parser.add_argument('--tracking', type=str, required=True, help='Path to tracking results (entities.json)')
    parser.add_argument('--intrinsics', type=str, required=True, help='Path to camera intrinsics (json)')
    args = parser.parse_args()
    main(args)
