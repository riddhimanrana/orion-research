"""
Depth processing utilities for improved SLAM scale estimation.

This module provides:
1. Depth uncertainty estimation
2. Temporal depth filtering  
3. Robust scale estimation from depth
4. Depth-visual fusion utilities
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DepthQuality:
    """Quality metrics for a depth map"""
    uncertainty_map: np.ndarray  # Per-pixel uncertainty [0,1]
    avg_uncertainty: float  # Average uncertainty
    valid_ratio: float  # Ratio of valid depth pixels
    edge_ratio: float  # Ratio of edge pixels


class DepthUncertaintyEstimator:
    """
    Estimate confidence/uncertainty for each depth pixel.
    
    Uncertainty factors:
    - Edge proximity (edges = unreliable depth transitions)
    - Texture (low texture = uncertain depth from monocular)
    - Depth gradient (sharp changes = object boundaries)
    - Distance (far = more uncertain due to scale ambiguity)
    """
    
    def __init__(
        self,
        edge_weight: float = 0.3,
        texture_weight: float = 0.2,
        gradient_weight: float = 0.3,
        distance_weight: float = 0.2,
    ):
        """
        Initialize uncertainty estimator.
        
        Args:
            edge_weight: Weight for edge-based uncertainty
            texture_weight: Weight for texture-based uncertainty
            gradient_weight: Weight for depth gradient uncertainty
            distance_weight: Weight for distance-based uncertainty
        """
        self.edge_weight = edge_weight
        self.texture_weight = texture_weight
        self.gradient_weight = gradient_weight
        self.distance_weight = distance_weight
    
    def estimate(
        self,
        depth_map: np.ndarray,
        rgb_frame: np.ndarray,
    ) -> DepthQuality:
        """
        Estimate uncertainty for each depth pixel.
        
        Args:
            depth_map: Depth map (HxW) in mm
            rgb_frame: RGB frame (HxWx3)
        
        Returns:
            DepthQuality with uncertainty map and metrics
        """
        h, w = depth_map.shape
        uncertainty = np.zeros((h, w), dtype=np.float32)
        
        # 1. Edge-based uncertainty
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, None, iterations=2)
        edge_uncertainty = edges_dilated.astype(float) / 255.0
        uncertainty += edge_uncertainty * self.edge_weight
        
        # 2. Texture-based uncertainty
        # Low texture areas are uncertain for monocular depth
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        texture_mag = np.abs(texture)
        # Invert: low texture = high uncertainty
        texture_uncertainty = np.clip(1.0 - texture_mag / 50.0, 0, 1)
        uncertainty += texture_uncertainty * self.texture_weight
        
        # 3. Depth gradient uncertainty
        # Sharp depth changes indicate object boundaries (uncertain)
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_uncertainty = np.clip(grad_mag / 1000.0, 0, 1)
        uncertainty += gradient_uncertainty * self.gradient_weight
        
        # 4. Distance-based uncertainty
        # Farther points are more uncertain (scale ambiguity)
        valid_mask = (depth_map > 100) & (depth_map < 10000)
        normalized_depth = np.zeros_like(depth_map, dtype=float)
        normalized_depth[valid_mask] = depth_map[valid_mask] / 10000.0
        distance_uncertainty = normalized_depth
        uncertainty += distance_uncertainty * self.distance_weight
        
        # Clip to [0, 1]
        uncertainty = np.clip(uncertainty, 0, 1)
        
        # Compute quality metrics
        avg_uncertainty = float(np.mean(uncertainty))
        valid_ratio = float(np.sum(valid_mask) / (h * w))
        edge_ratio = float(np.sum(edges > 0) / (h * w))
        
        return DepthQuality(
            uncertainty_map=uncertainty,
            avg_uncertainty=avg_uncertainty,
            valid_ratio=valid_ratio,
            edge_ratio=edge_ratio,
        )


class TemporalDepthFilter:
    """
    Smooth depth maps across time using exponential moving average.
    
    This helps reduce temporal flickering and noise in depth estimates.
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize temporal filter.
        
        Args:
            alpha: Weight for current frame (0=only history, 1=only current)
        """
        self.alpha = alpha
        self.filtered_depth: Optional[np.ndarray] = None
        self.frame_count = 0
    
    def update(
        self,
        new_depth: np.ndarray,
        uncertainty_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Update filter with new depth measurement.
        
        Args:
            new_depth: New depth map (HxW) in mm
            uncertainty_map: Optional uncertainty map [0,1]
        
        Returns:
            Filtered depth map
        """
        if self.filtered_depth is None:
            self.filtered_depth = new_depth.copy().astype(np.float32)
            self.frame_count = 1
            return self.filtered_depth
        
        # Compute adaptive weights based on uncertainty
        if uncertainty_map is not None:
            # Higher certainty → trust new measurement more
            certainty = 1.0 - uncertainty_map
            # Adaptive alpha: certain pixels use alpha, uncertain use lower
            adaptive_alpha = certainty * self.alpha + (1 - certainty) * 0.3
        else:
            adaptive_alpha = self.alpha
        
        # Exponential moving average
        self.filtered_depth = (
            adaptive_alpha * new_depth.astype(np.float32) +
            (1 - adaptive_alpha) * self.filtered_depth
        )
        
        self.frame_count += 1
        
        return self.filtered_depth
    
    def reset(self):
        """Reset filter state"""
        self.filtered_depth = None
        self.frame_count = 0


class ScaleKalmanFilter:
    """
    Kalman filter for temporal scale smoothing.
    
    Tracks scale over time with prediction and measurement updates.
    Helps maintain consistent scale across frames.
    """
    
    def __init__(
        self,
        initial_scale: float = 1.0,
        process_noise: float = 0.01,
    ):
        """
        Initialize Kalman filter.
        
        Args:
            initial_scale: Initial scale estimate
            process_noise: How much scale can change per frame
        """
        self.state = initial_scale  # Scale estimate
        self.covariance = 1.0  # Uncertainty in estimate
        self.process_noise = process_noise
    
    def predict(self):
        """
        Predict step: assume scale doesn't change much.
        Increases uncertainty slightly.
        """
        self.covariance += self.process_noise
    
    def update(
        self,
        measurement: float,
        measurement_confidence: float = 1.0,
    ) -> float:
        """
        Update step: incorporate new scale measurement.
        
        Args:
            measurement: Measured scale value
            measurement_confidence: Confidence in measurement [0,1]
        
        Returns:
            Updated scale estimate
        """
        # Measurement noise (inversely proportional to confidence)
        measurement_noise = max(0.01, 1.0 - measurement_confidence)
        
        # Kalman gain: how much to trust measurement vs prediction
        K = self.covariance / (self.covariance + measurement_noise)
        
        # Update state (weighted average)
        self.state = self.state + K * (measurement - self.state)
        
        # Update covariance (reduce uncertainty)
        self.covariance = (1 - K) * self.covariance
        
        return self.state
    
    def get_state(self) -> Tuple[float, float]:
        """
        Get current filter state.
        
        Returns:
            (scale, covariance)
        """
        return self.state, self.covariance
    
    def reset(self, scale: float = 1.0):
        """Reset filter to initial state"""
        self.state = scale
        self.covariance = 1.0


def backproject_point(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
) -> np.ndarray:
    """
    Backproject 2D pixel + depth to 3D point.
    
    Args:
        u, v: Pixel coordinates
        depth: Depth value in mm
        K: Camera intrinsic matrix (3x3)
    
    Returns:
        3D point [x, y, z] in camera frame (mm)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    return np.array([x, y, z])


def project_point(
    point_3d: np.ndarray,
    K: np.ndarray,
) -> Tuple[float, float]:
    """
    Project 3D point to 2D pixel.
    
    Args:
        point_3d: 3D point [x, y, z] in camera frame (mm)
        K: Camera intrinsic matrix (3x3)
    
    Returns:
        (u, v) pixel coordinates
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x, y, z = point_3d
    
    u = fx * x / z + cx
    v = fy * y / z + cy
    
    return u, v


def check_depth_consistency(
    pts1: np.ndarray,
    pts2: np.ndarray,
    depth1: np.ndarray,
    depth2: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    reprojection_threshold: float = 3.0,
    depth_threshold: float = 0.3,
) -> np.ndarray:
    """
    Check if depth values are consistent with visual odometry.
    
    Uses epipolar geometry: backproject points from frame 1,
    transform to frame 2, and check if they match observations.
    
    Args:
        pts1: Points in frame 1 (Nx2)
        pts2: Points in frame 2 (Nx2)
        depth1: Depth map of frame 1 (HxW)
        depth2: Depth map of frame 2 (HxW)
        pose: Relative pose from frame 1 to 2 (4x4)
        K: Camera intrinsic matrix (3x3)
        reprojection_threshold: Max reprojection error in pixels
        depth_threshold: Max relative depth error (0.3 = 30%)
    
    Returns:
        Boolean mask (N,) - True if consistent, False if outlier
    """
    n_points = len(pts1)
    consistent = np.zeros(n_points, dtype=bool)
    
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    for i in range(n_points):
        try:
            # Backproject point in frame 1
            u1, v1 = pts1[i]
            d1 = depth1[int(v1), int(u1)]
            
            if d1 < 100 or d1 > 10000:
                continue
            
            p1_3d = backproject_point(u1, v1, d1, K)
            
            # Transform to frame 2
            p2_3d_predicted = R @ p1_3d + t
            
            # Project to frame 2
            u2_pred, v2_pred = project_point(p2_3d_predicted, K)
            
            # Check if it matches observed point
            u2_obs, v2_obs = pts2[i]
            
            reprojection_error = np.sqrt(
                (u2_pred - u2_obs)**2 + (v2_pred - v2_obs)**2
            )
            
            # Also check depth consistency
            d2_obs = depth2[int(v2_obs), int(u2_obs)]
            
            if d2_obs < 100 or d2_obs > 10000:
                continue
            
            d2_pred = p2_3d_predicted[2]
            depth_error = abs(d2_obs - d2_pred) / max(d2_pred, 1.0)
            
            # Consistent if both errors are small
            if reprojection_error < reprojection_threshold and depth_error < depth_threshold:
                consistent[i] = True
                
        except (IndexError, ValueError):
            # Out of bounds or invalid
            continue
    
    return consistent
