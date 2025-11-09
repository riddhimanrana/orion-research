"""
Camera intrinsics and 3D backprojection utilities.
"""

from typing import Tuple, Dict, Any
import numpy as np
from .types import CameraIntrinsics


def backproject_point(
    u: float,
    v: float,
    depth_z: float,
    intrinsics: CameraIntrinsics
) -> Tuple[float, float, float]:
    """
    Convert 2D pixel + depth to 3D world coordinates.
    
    Uses pinhole camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth
    
    Args:
        u, v: Pixel coordinates (x, y)
        depth_z: Depth value in millimeters
        intrinsics: Camera intrinsics
        
    Returns:
        (X_3d, Y_3d, Z_3d) in millimeters
    """
    X_3d = (u - intrinsics.cx) * depth_z / intrinsics.fx
    Y_3d = (v - intrinsics.cy) * depth_z / intrinsics.fy
    Z_3d = depth_z
    
    return (X_3d, Y_3d, Z_3d)


def backproject_bbox(
    bbox_2d: Tuple[int, int, int, int],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics
) -> Dict[str, Any]:
    """
    Backproject 2D bounding box using depth map to get 3D information.
    
    Args:
        bbox_2d: (x1, y1, x2, y2) in pixels
        depth_map: Depth map (H, W) in millimeters
        intrinsics: Camera intrinsics
        
    Returns:
        Dictionary with:
            - centroid_3d: (X, Y, Z) in mm
            - bbox_3d_min: (X_min, Y_min, Z_min) in mm
            - bbox_3d_max: (X_max, Y_max, Z_max) in mm
            - depth_mean: Mean depth in mm
            - depth_variance: Depth variance in mm²
    """
    x1, y1, x2, y2 = bbox_2d
    
    # Convert to integers and clamp to image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(depth_map.shape[1] - 1, int(x2))
    y2 = min(depth_map.shape[0] - 1, int(y2))
    
    # Extract depth region
    depth_region = depth_map[y1:y2, x1:x2]
    
    # Filter out invalid depths (0 or NaN)
    valid_depths = depth_region[(depth_region > 0) & np.isfinite(depth_region)]
    
    if len(valid_depths) == 0:
        # No valid depth, return defaults
        return {
            "centroid_3d": (0.0, 0.0, 0.0),
            "bbox_3d_min": (0.0, 0.0, 0.0),
            "bbox_3d_max": (0.0, 0.0, 0.0),
            "depth_mean": 0.0,
            "depth_variance": 0.0,
        }
    
    # Compute statistics
    depth_mean = float(np.mean(valid_depths))
    depth_std = float(np.std(valid_depths))
    depth_variance = depth_std ** 2
    
    # Backproject bbox center
    center_u = (x1 + x2) / 2.0
    center_v = (y1 + y2) / 2.0
    centroid_3d = backproject_point(center_u, center_v, depth_mean, intrinsics)
    
    # Backproject corners to get 3D bounding box
    corners_2d = [
        (x1, y1), (x2, y1),
        (x1, y2), (x2, y2),
    ]
    
    corners_3d = []
    for u, v in corners_2d:
        # Use local depth at corner if available
        u_int = int(np.clip(u, 0, depth_map.shape[1] - 1))
        v_int = int(np.clip(v, 0, depth_map.shape[0] - 1))
        corner_depth = depth_map[v_int, u_int]
        
        if corner_depth > 0 and np.isfinite(corner_depth):
            corner_3d = backproject_point(u, v, corner_depth, intrinsics)
        else:
            corner_3d = backproject_point(u, v, depth_mean, intrinsics)
        
        corners_3d.append(corner_3d)
    
    # Compute 3D bounding box
    corners_array = np.array(corners_3d)
    bbox_3d_min = tuple(corners_array.min(axis=0).tolist())
    bbox_3d_max = tuple(corners_array.max(axis=0).tolist())
    
    return {
        "centroid_3d": centroid_3d,
        "bbox_3d_min": bbox_3d_min,
        "bbox_3d_max": bbox_3d_max,
        "depth_mean": depth_mean,
        "depth_variance": depth_variance,
    }
