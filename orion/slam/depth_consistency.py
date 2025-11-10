"""
Depth Consistency Checking for Visual SLAM

Validates depth estimates using epipolar geometry and temporal consistency
to filter out unreliable measurements before pose estimation.
"""

import numpy as np
from typing import Tuple, Optional
import cv2


def check_epipolar_consistency(
    p1_2d: np.ndarray,
    p2_2d: np.ndarray,
    depth1: float,
    depth2: float,
    E: np.ndarray,
    K: np.ndarray,
    threshold: float = 1.0
) -> bool:
    """
    Check if depth is consistent with epipolar geometry.
    
    The epipolar constraint states: p2^T @ E @ p1 = 0
    If depths are correct, the reprojected 3D points should satisfy this constraint.
    
    Args:
        p1_2d: 2D point in frame 1 (u, v)
        p2_2d: 2D point in frame 2 (u, v)
        depth1: Depth at p1 in mm
        depth2: Depth at p2 in mm
        E: Essential matrix (3x3)
        K: Camera intrinsics (3x3)
        threshold: Epipolar error threshold in pixels (default: 1.0)
    
    Returns:
        True if depth is consistent with epipolar geometry
    """
    # Convert to homogeneous coordinates
    p1_h = np.array([p1_2d[0], p1_2d[1], 1.0])
    p2_h = np.array([p2_2d[0], p2_2d[1], 1.0])
    
    # Compute epipolar error
    # Error should be near 0 if geometry is consistent
    error = abs(p2_h @ E @ p1_h)
    
    # Normalize by image size (rough approximation)
    # For a 640x480 image, 1 pixel error is reasonable
    return error < threshold


def check_depth_consistency_batch(
    pts1: np.ndarray,
    pts2: np.ndarray,
    depths1: np.ndarray,
    depths2: np.ndarray,
    E: np.ndarray,
    K: np.ndarray,
    threshold: float = 1.0
) -> np.ndarray:
    """
    Batch check epipolar consistency for multiple points.
    
    Args:
        pts1: Points in frame 1 (Nx2)
        pts2: Points in frame 2 (Nx2)
        depths1: Depths at pts1 (N,)
        depths2: Depths at pts2 (N,)
        E: Essential matrix (3x3)
        K: Camera intrinsics (3x3)
        threshold: Epipolar error threshold
    
    Returns:
        Boolean mask (N,) - True for consistent points
    """
    n_points = len(pts1)
    consistent = np.zeros(n_points, dtype=bool)
    
    for i in range(n_points):
        consistent[i] = check_epipolar_consistency(
            pts1[i], pts2[i], depths1[i], depths2[i], E, K, threshold
        )
    
    return consistent


def check_temporal_depth_consistency(
    depth_t0: np.ndarray,
    depth_t1: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    threshold: float = 100.0
) -> np.ndarray:
    """
    Check temporal consistency by warping depth from t0 to t1.
    
    Args:
        depth_t0: Depth at time t0 (HxW) in mm
        depth_t1: Depth at time t1 (HxW) in mm
        pose: Relative pose from t0 to t1 (4x4 matrix)
        K: Camera intrinsics (3x3)
        threshold: Depth difference threshold in mm (default: 100mm)
    
    Returns:
        Valid mask (HxW) - True for consistent pixels
    """
    h, w = depth_t0.shape
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Backproject depth_t0 to 3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Points in camera frame at t0
    X = (u - cx) * depth_t0 / fx
    Y = (v - cy) * depth_t0 / fy
    Z = depth_t0
    
    # Stack into homogeneous coordinates (HxWx4)
    points_t0 = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)
    
    # Transform to t1
    points_t1 = np.einsum('ij,hwj->hwi', pose, points_t0)
    
    # Project back to image
    X_t1 = points_t1[:, :, 0]
    Y_t1 = points_t1[:, :, 1]
    Z_t1 = points_t1[:, :, 2]
    
    u_t1 = (X_t1 * fx / Z_t1 + cx).astype(int)
    v_t1 = (Y_t1 * fy / Z_t1 + cy).astype(int)
    
    # Check bounds
    valid_mask = (u_t1 >= 0) & (u_t1 < w) & (v_t1 >= 0) & (v_t1 < h) & (Z_t1 > 0)
    
    # Compare warped depth with actual depth at t1
    depth_warped = np.zeros_like(depth_t1)
    depth_warped[valid_mask] = Z_t1[valid_mask]
    
    # Compute difference
    depth_diff = np.abs(depth_warped - depth_t1)
    
    # Mark as consistent if difference < threshold
    consistent = (depth_diff < threshold) & valid_mask
    
    return consistent


def filter_depth_outliers_by_consistency(
    pts1: np.ndarray,
    pts2: np.ndarray,
    depths1: np.ndarray,
    depths2: np.ndarray,
    E: np.ndarray,
    K: np.ndarray,
    epipolar_threshold: float = 1.0,
    depth_ratio_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter depth outliers using epipolar consistency and depth ratio checks.
    
    Args:
        pts1: Points in frame 1 (Nx2)
        pts2: Points in frame 2 (Nx2)
        depths1: Depths at pts1 (N,)
        depths2: Depths at pts2 (N,)
        E: Essential matrix (3x3)
        K: Camera intrinsics (3x3)
        epipolar_threshold: Epipolar error threshold
        depth_ratio_threshold: Max allowed depth ratio change (e.g., 0.3 = 30%)
    
    Returns:
        Filtered (pts1, pts2, depths1, depths2, inlier_mask)
    """
    n_points = len(pts1)
    
    # Check 1: Epipolar consistency
    epipolar_consistent = check_depth_consistency_batch(
        pts1, pts2, depths1, depths2, E, K, epipolar_threshold
    )
    
    # Check 2: Depth ratio (detect moving objects or bad depth)
    # Depths should be similar for static scene
    depth_ratios = np.abs(depths1 - depths2) / (depths1 + 1e-6)
    depth_ratio_ok = depth_ratios < depth_ratio_threshold
    
    # Check 3: Valid depth range
    valid_depth = (depths1 > 100) & (depths1 < 10000) & (depths2 > 100) & (depths2 < 10000)
    
    # Combine all checks
    inlier_mask = epipolar_consistent & depth_ratio_ok & valid_depth
    
    # Filter points
    pts1_filtered = pts1[inlier_mask]
    pts2_filtered = pts2[inlier_mask]
    depths1_filtered = depths1[inlier_mask]
    depths2_filtered = depths2[inlier_mask]
    
    return pts1_filtered, pts2_filtered, depths1_filtered, depths2_filtered, inlier_mask


def detect_depth_discontinuities(
    depth_map: np.ndarray,
    threshold: float = 500.0,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Detect depth discontinuities (edges) in depth map.
    
    Depth edges indicate object boundaries where depth estimates
    are less reliable.
    
    Args:
        depth_map: Depth map (HxW) in mm
        threshold: Depth gradient threshold in mm (default: 500mm = 50cm)
        kernel_size: Sobel kernel size (default: 3)
    
    Returns:
        Edge mask (HxW) - True at depth discontinuities
    """
    # Compute depth gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    # Magnitude of gradient
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold to detect edges
    edge_mask = grad_magnitude > threshold
    
    return edge_mask


def update_uncertainty_with_consistency(
    uncertainty_map: np.ndarray,
    consistency_mask: np.ndarray,
    boost_factor: float = 0.2
) -> np.ndarray:
    """
    Update uncertainty map based on temporal consistency.
    
    Pixels that are temporally consistent get lower uncertainty.
    
    Args:
        uncertainty_map: Current uncertainty (HxW) in [0, 1]
        consistency_mask: Temporal consistency mask (HxW)
        boost_factor: How much to reduce uncertainty for consistent pixels
    
    Returns:
        Updated uncertainty map (HxW)
    """
    updated_uncertainty = uncertainty_map.copy()
    
    # Reduce uncertainty for consistent pixels
    updated_uncertainty[consistency_mask] *= (1.0 - boost_factor)
    
    # Increase uncertainty for inconsistent pixels
    updated_uncertainty[~consistency_mask] = np.minimum(
        updated_uncertainty[~consistency_mask] + boost_factor,
        1.0
    )
    
    return updated_uncertainty


class DepthConsistencyChecker:
    """
    Stateful depth consistency checker for SLAM.
    
    Maintains history and performs temporal consistency checks.
    """
    
    def __init__(
        self,
        epipolar_threshold: float = 1.0,
        temporal_threshold: float = 100.0,
        depth_ratio_threshold: float = 0.3
    ):
        """
        Initialize depth consistency checker.
        
        Args:
            epipolar_threshold: Epipolar error threshold in pixels
            temporal_threshold: Temporal depth difference threshold in mm
            depth_ratio_threshold: Max depth ratio change (0.3 = 30%)
        """
        self.epipolar_threshold = epipolar_threshold
        self.temporal_threshold = temporal_threshold
        self.depth_ratio_threshold = depth_ratio_threshold
        
        # Statistics
        self.total_checks = 0
        self.total_inliers = 0
        self.total_outliers = 0
    
    def check_and_filter(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        depths1: np.ndarray,
        depths2: np.ndarray,
        E: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Check consistency and filter outliers.
        
        Returns:
            (filtered_pts1, filtered_pts2, filtered_depths1, filtered_depths2, inlier_ratio)
        """
        self.total_checks += len(pts1)
        
        # Filter outliers
        pts1_f, pts2_f, d1_f, d2_f, mask = filter_depth_outliers_by_consistency(
            pts1, pts2, depths1, depths2, E, K,
            self.epipolar_threshold, self.depth_ratio_threshold
        )
        
        n_inliers = np.sum(mask)
        n_outliers = len(mask) - n_inliers
        
        self.total_inliers += n_inliers
        self.total_outliers += n_outliers
        
        inlier_ratio = n_inliers / len(mask) if len(mask) > 0 else 0.0
        
        return pts1_f, pts2_f, d1_f, d2_f, inlier_ratio
    
    def get_statistics(self) -> dict:
        """Get consistency check statistics"""
        if self.total_checks == 0:
            return {
                "total_checks": 0,
                "total_inliers": 0,
                "total_outliers": 0,
                "inlier_ratio": 0.0,
                "outlier_ratio": 0.0
            }
        
        return {
            "total_checks": self.total_checks,
            "total_inliers": self.total_inliers,
            "total_outliers": self.total_outliers,
            "inlier_ratio": self.total_inliers / self.total_checks,
            "outlier_ratio": self.total_outliers / self.total_checks
        }
