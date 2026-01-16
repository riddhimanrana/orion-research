"""
3D Projection Utilities for SLAM
=================================

Converts 2D bounding boxes + depth maps + camera poses → 3D world coordinates.

Author: Orion Research Team
Date: November 2025
"""

import numpy as np
from typing import Optional


def project_bbox_to_3d(
    bbox: tuple[int, int, int, int],  # (x1, y1, x2, y2)
    depth_map: np.ndarray,  # HxW depth in mm
    camera_intrinsics: np.ndarray,  # 3x3 K matrix
    camera_pose: np.ndarray,  # 4x4 pose matrix (camera to world)
) -> Optional[np.ndarray]:
    """
    Project 2D bounding box center to 3D world coordinates.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2) in pixels
        depth_map: Depth map in mm (HxW)
        camera_intrinsics: Camera intrinsics K (3x3)
        camera_pose: Camera pose in world frame (4x4)
    
    Returns:
        3D position [x, y, z] in mm (world frame), or None if invalid
    """
    x1, y1, x2, y2 = bbox
    
    # Get bbox center
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    
    # Clamp to image bounds
    h, w = depth_map.shape
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    
    # Sample depth at center (with 3x3 median for robustness)
    half_size = 1
    y_start = max(0, cy - half_size)
    y_end = min(h, cy + half_size + 1)
    x_start = max(0, cx - half_size)
    x_end = min(w, cx + half_size + 1)
    
    depth_patch = depth_map[y_start:y_end, x_start:x_end]
    
    # Remove invalid depths
    valid_depths = depth_patch[(depth_patch > 10.0) & (depth_patch < 10000.0)]
    if len(valid_depths) == 0:
        return None
    
    depth_value = np.median(valid_depths)
    
    # Unproject to camera frame
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx_k = camera_intrinsics[0, 2]
    cy_k = camera_intrinsics[1, 2]
    
    x_cam = (cx - cx_k) * depth_value / fx
    y_cam = (cy - cy_k) * depth_value / fy
    z_cam = depth_value
    
    # Convert to homogeneous coordinates
    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    
    # Transform to world frame
    point_world = camera_pose @ point_cam
    
    return point_world[:3]  # Return [x, y, z]


def project_points_to_3d(
    points_2d: np.ndarray,  # Nx2 array of (x, y) pixel coordinates
    depth_map: np.ndarray,  # HxW depth in mm
    camera_intrinsics: np.ndarray,  # 3x3 K matrix
    camera_pose: np.ndarray,  # 4x4 pose matrix
) -> np.ndarray:
    """
    Project multiple 2D points to 3D world coordinates.
    
    Args:
        points_2d: Nx2 array of (x, y) pixel coordinates
        depth_map: Depth map in mm
        camera_intrinsics: Camera intrinsics K
        camera_pose: Camera pose in world frame
    
    Returns:
        Nx3 array of 3D points in world frame
    """
    h, w = depth_map.shape
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    
    points_3d = []
    
    for x, y in points_2d:
        # Clamp to image bounds
        x_int = int(np.clip(x, 0, w - 1))
        y_int = int(np.clip(y, 0, h - 1))
        
        depth = depth_map[y_int, x_int]
        
        if depth < 10.0 or depth > 10000.0:
            # Invalid depth, use placeholder
            points_3d.append([0, 0, 0])
            continue
        
        # Unproject to camera frame
        x_cam = (x - cx) * depth / fx
        y_cam = (y - cy) * depth / fy
        z_cam = depth
        
        # Transform to world frame
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        point_world = camera_pose @ point_cam
        
        points_3d.append(point_world[:3])
    
    return np.array(points_3d)


def compute_3d_velocity(
    prev_position: Optional[np.ndarray],
    curr_position: np.ndarray,
    time_delta: float,  # seconds
) -> np.ndarray:
    """
    Compute 3D velocity from consecutive positions.
    
    Args:
        prev_position: Previous 3D position [x, y, z] in mm
        curr_position: Current 3D position [x, y, z] in mm
        time_delta: Time between frames in seconds
    
    Returns:
        3D velocity [vx, vy, vz] in mm/s
    """
    if prev_position is None or time_delta <= 0:
        return np.array([0.0, 0.0, 0.0])
    
    displacement = curr_position - prev_position
    velocity = displacement / time_delta
    
    return velocity


def estimate_bbox_depth_robust(
    bbox: tuple[int, int, int, int],
    depth_map: np.ndarray,
    method: str = "median",  # "median", "mean", "mode"
) -> Optional[float]:
    """
    Robustly estimate depth for a bounding box.
    
    Samples multiple points within bbox and uses robust statistics.
    
    Args:
        bbox: (x1, y1, x2, y2)
        depth_map: Depth map in mm
        method: Aggregation method
    
    Returns:
        Estimated depth in mm, or None if invalid
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape
    
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Sample grid of points within bbox (not just center)
    # This is more robust for partially occluded objects
    grid_size = 5
    xs = np.linspace(x1, x2, grid_size, dtype=int)
    ys = np.linspace(y1, y2, grid_size, dtype=int)
    
    depths = []
    for x in xs:
        for y in ys:
            if 0 <= y < h and 0 <= x < w:
                d = depth_map[y, x]
                if 10.0 < d < 10000.0:  # Valid range
                    depths.append(d)
    
    if len(depths) == 0:
        return None
    
    depths = np.array(depths)
    
    if method == "median":
        return float(np.median(depths))
    elif method == "mean":
        return float(np.mean(depths))
    elif method == "mode":
        # Use histogram mode (most common depth)
        hist, bins = np.histogram(depths, bins=20)
        mode_idx = np.argmax(hist)
        return float((bins[mode_idx] + bins[mode_idx + 1]) / 2)
    else:
        return float(np.median(depths))
