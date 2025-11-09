"""
Depth-based visual odometry using ICP for low-texture scenes.
Fallback when feature-based SLAM fails.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DepthOdometryConfig:
    """Configuration for depth-based odometry"""
    min_depth_mm: int = 100      # Minimum valid depth
    max_depth_mm: int = 8000     # Maximum valid depth
    downsample_factor: int = 4   # Downsample for speed (e.g., 4 = 1/16 points)
    icp_max_iterations: int = 50
    icp_tolerance: float = 1e-5
    min_points: int = 100        # Minimum points for ICP


class DepthOdometry:
    """Depth-based visual odometry using ICP on point clouds"""
    
    def __init__(self, K: np.ndarray, config: Optional[DepthOdometryConfig] = None):
        """
        Args:
            K: 3x3 camera intrinsics matrix
            config: Optional configuration
        """
        self.K = K
        self.config = config or DepthOdometryConfig()
        
        # Camera parameters
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        
        # State
        self.prev_depth = None
        self.cumulative_pose = np.eye(4)
        
    def depth_to_pointcloud(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D point cloud.
        
        Returns:
            points: Nx3 array of 3D points
            valid_mask: HxW boolean mask of valid points
        """
        h, w = depth.shape
        
        # Downsample for speed
        ds = self.config.downsample_factor
        depth_ds = depth[::ds, ::ds]
        h_ds, w_ds = depth_ds.shape
        
        # Create meshgrid
        u, v = np.meshgrid(np.arange(w_ds), np.arange(h_ds))
        u = u * ds
        v = v * ds
        
        # Valid depth mask
        valid = (depth_ds > self.config.min_depth_mm) & (depth_ds < self.config.max_depth_mm)
        
        # Back-project to 3D
        z = depth_ds[valid].astype(np.float32)
        x = (u[valid] - self.cx) * z / self.fx
        y = (v[valid] - self.cy) * z / self.fy
        
        points = np.stack([x, y, z], axis=1)  # Nx3
        
        return points, valid
    
    def simple_icp(self, source: np.ndarray, target: np.ndarray, 
                   init_transform: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Simple ICP implementation for point cloud alignment.
        
        Args:
            source: Nx3 source points (current frame)
            target: Mx3 target points (previous frame)
            init_transform: Initial 4x4 transformation guess
            
        Returns:
            4x4 transformation matrix or None if failed
        """
        if len(source) < self.config.min_points or len(target) < self.config.min_points:
            return None
        
        # Initialize
        if init_transform is None:
            transform = np.eye(4)
        else:
            transform = init_transform.copy()
        
        prev_error = float('inf')
        
        for iteration in range(self.config.icp_max_iterations):
            # Apply current transform
            source_h = np.hstack([source, np.ones((len(source), 1))])
            source_transformed = (transform @ source_h.T).T[:, :3]
            
            # Find nearest neighbors (simple brute force for small point sets)
            distances = np.linalg.norm(
                source_transformed[:, np.newaxis, :] - target[np.newaxis, :, :],
                axis=2
            )
            closest_indices = np.argmin(distances, axis=1)
            
            # Compute error
            error = np.mean(np.min(distances, axis=1))
            
            # Check convergence
            if abs(prev_error - error) < self.config.icp_tolerance:
                break
            prev_error = error
            
            # Estimate transformation
            source_matched = source_transformed
            target_matched = target[closest_indices]
            
            # Compute centroids
            source_centroid = np.mean(source_matched, axis=0)
            target_centroid = np.mean(target_matched, axis=0)
            
            # Center the points
            source_centered = source_matched - source_centroid
            target_centered = target_matched - target_centroid
            
            # Compute cross-covariance matrix
            H = source_centered.T @ target_centered
            
            # SVD
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Handle reflection case
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = target_centroid - R @ source_centroid
            
            # Update transform
            delta_transform = np.eye(4)
            delta_transform[:3, :3] = R
            delta_transform[:3, 3] = t
            transform = delta_transform @ transform
        
        # Validate result
        if prev_error > 100:  # > 10cm average error
            return None
        
        return transform
    
    def track(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """
        Track camera motion using depth-based ICP.
        
        Args:
            depth: HxW depth map in mm
            
        Returns:
            4x4 cumulative pose matrix or None if tracking failed
        """
        if self.prev_depth is None:
            # First frame - just store
            self.prev_depth = depth.copy()
            return self.cumulative_pose.copy()
        
        try:
            # Convert to point clouds
            curr_points, _ = self.depth_to_pointcloud(depth)
            prev_points, _ = self.depth_to_pointcloud(self.prev_depth)
            
            # Run ICP
            delta_pose = self.simple_icp(curr_points, prev_points)
            
            if delta_pose is None:
                return None
            
            # Update cumulative pose
            self.cumulative_pose = self.cumulative_pose @ delta_pose
            
            # Store for next frame
            self.prev_depth = depth.copy()
            
            # Compute motion magnitude for logging
            translation = np.linalg.norm(delta_pose[:3, 3])
            
            print(f"[Depth Odometry] Motion: {translation:.1f}mm")
            
            return self.cumulative_pose.copy()
            
        except Exception as e:
            print(f"[Depth Odometry] Failed: {e}")
            return None
    
    def reset(self):
        """Reset odometry state"""
        self.prev_depth = None
        self.cumulative_pose = np.eye(4)
