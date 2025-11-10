"""
Multi-frame depth fusion with sliding window.

Phase 3 Week 6 Day 3 - Aggregates depth estimates across multiple frames
to improve depth accuracy and consistency.

Author: Orion Research Team
Date: November 9, 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


def warp_depth_to_frame(
    depth_src: np.ndarray,
    pose_src: np.ndarray,
    pose_tgt: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp depth map from source frame to target frame using poses.
    
    Args:
        depth_src: Source depth map (H, W)
        pose_src: Source camera pose (4, 4)
        pose_tgt: Target camera pose (4, 4)
        K: Camera intrinsics (3, 3)
    
    Returns:
        warped_depth: Warped depth map (H, W)
        valid_mask: Mask of valid warped pixels (H, W)
    """
    H, W = depth_src.shape
    
    # Compute relative pose: tgt -> src
    pose_rel = np.linalg.inv(pose_tgt) @ pose_src
    R = pose_rel[:3, :3]
    t = pose_rel[:3, 3]
    
    # Create pixel grid
    y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)
    
    # Convert to normalized camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_norm = (x_coords - cx) / fx
    y_norm = (y_coords - cy) / fy
    
    # Unproject to 3D using source depth
    z = depth_src
    x_3d = x_norm * z
    y_3d = y_norm * z
    z_3d = z
    
    # Stack into (H, W, 3)
    points_3d = np.stack([x_3d, y_3d, z_3d], axis=-1)
    
    # Transform to target frame
    points_3d_flat = points_3d.reshape(-1, 3)
    points_tgt = (R @ points_3d_flat.T).T + t  # (N, 3)
    points_tgt = points_tgt.reshape(H, W, 3)
    
    # Project to target image
    x_tgt = points_tgt[:, :, 0]
    y_tgt = points_tgt[:, :, 1]
    z_tgt = points_tgt[:, :, 2]
    
    u_tgt = (x_tgt / z_tgt) * fx + cx
    v_tgt = (y_tgt / z_tgt) * fy + cy
    
    # Check valid projections
    valid_mask = (
        (z_tgt > 0) &
        (u_tgt >= 0) & (u_tgt < W) &
        (v_tgt >= 0) & (v_tgt < H)
    )
    
    # Initialize warped depth
    warped_depth = np.zeros((H, W), dtype=np.float32)
    
    # Fill in warped depth (simple nearest neighbor)
    u_int = u_tgt[valid_mask].astype(int)
    v_int = v_tgt[valid_mask].astype(int)
    z_warped = z_tgt[valid_mask]
    
    # Handle multiple points projecting to same pixel (take minimum depth)
    for i in range(len(u_int)):
        u, v, z = u_int[i], v_int[i], z_warped[i]
        if warped_depth[v, u] == 0 or z < warped_depth[v, u]:
            warped_depth[v, u] = z
    
    # Create final valid mask (where warped depth is non-zero)
    warped_valid_mask = warped_depth > 0
    
    return warped_depth, warped_valid_mask


def fuse_depth_maps_weighted(
    depth_maps: List[np.ndarray],
    confidence_maps: List[np.ndarray],
    valid_masks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse multiple depth maps using confidence-weighted averaging.
    
    Args:
        depth_maps: List of N depth maps (H, W)
        confidence_maps: List of N confidence maps (H, W) in [0, 1]
        valid_masks: List of N valid masks (H, W) boolean
    
    Returns:
        fused_depth: Fused depth map (H, W)
        fused_confidence: Fused confidence map (H, W)
    """
    if not depth_maps:
        raise ValueError("depth_maps cannot be empty")
    
    H, W = depth_maps[0].shape
    
    # Initialize accumulators
    weighted_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    
    # Accumulate weighted depths
    for depth, confidence, valid in zip(depth_maps, confidence_maps, valid_masks):
        # Weight by confidence
        weight = confidence * valid.astype(np.float32)
        weighted_sum += depth * weight
        weight_sum += weight
    
    # Compute fused depth (avoid division by zero)
    fused_depth = np.zeros((H, W), dtype=np.float32)
    valid_pixels = weight_sum > 1e-6
    fused_depth[valid_pixels] = weighted_sum[valid_pixels] / weight_sum[valid_pixels]
    
    # Compute fused confidence (average of contributing confidences)
    fused_confidence = np.zeros((H, W), dtype=np.float32)
    num_contributions = sum(valid.astype(np.float32) for valid in valid_masks)
    valid_pixels = num_contributions > 0
    
    # Average confidence where we have contributions
    for confidence, valid in zip(confidence_maps, valid_masks):
        fused_confidence += confidence * valid.astype(np.float32)
    
    fused_confidence[valid_pixels] /= num_contributions[valid_pixels]
    
    return fused_depth, fused_confidence


def reject_depth_outliers_temporal(
    depth_maps: List[np.ndarray],
    valid_masks: List[np.ndarray],
    threshold_mm: float = 100.0,
) -> List[np.ndarray]:
    """
    Reject temporal outliers by comparing with median depth.
    
    Args:
        depth_maps: List of N depth maps (H, W)
        valid_masks: List of N valid masks (H, W)
        threshold_mm: Outlier threshold in mm
    
    Returns:
        refined_masks: List of N refined valid masks
    """
    if len(depth_maps) < 3:
        # Need at least 3 frames for median
        return valid_masks
    
    H, W = depth_maps[0].shape
    
    # Stack depth maps (H, W, N)
    depth_stack = np.stack(depth_maps, axis=-1)
    valid_stack = np.stack([m.astype(np.float32) for m in valid_masks], axis=-1)
    
    # Compute median depth per pixel (only over valid observations)
    # Mask invalid depths with NaN
    depth_masked = depth_stack.copy()
    depth_masked[valid_stack == 0] = np.nan
    
    # Median ignoring NaN
    median_depth = np.nanmedian(depth_masked, axis=-1)
    
    # Check each depth map against median
    refined_masks = []
    for i, (depth, valid) in enumerate(zip(depth_maps, valid_masks)):
        # Compute residual
        residual = np.abs(depth - median_depth)
        
        # Mark as inlier if within threshold
        inlier_mask = (residual < threshold_mm) & valid & (median_depth > 0)
        refined_masks.append(inlier_mask)
    
    return refined_masks


class MultiFrameDepthFusion:
    """
    Fuses depth estimates across multiple frames using a sliding window.
    
    Maintains a window of recent depth maps and their poses, and fuses them
    by warping to the current frame and performing confidence-weighted averaging.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        outlier_threshold_mm: float = 100.0,
        min_confidence: float = 0.3,
    ):
        """
        Initialize multi-frame depth fusion.
        
        Args:
            window_size: Number of frames to keep in sliding window
            outlier_threshold_mm: Temporal outlier threshold (mm)
            min_confidence: Minimum confidence to use a depth estimate
        """
        self.window_size = window_size
        self.outlier_threshold_mm = outlier_threshold_mm
        self.min_confidence = min_confidence
        
        # Sliding window storage
        self.depth_window = deque(maxlen=window_size)
        self.pose_window = deque(maxlen=window_size)
        self.confidence_window = deque(maxlen=window_size)
        self.frame_idx_window = deque(maxlen=window_size)
        
        # Statistics
        self.total_fusions = 0
        self.total_frames_used = 0
        self.avg_frames_per_fusion = 0.0
    
    def add_frame(
        self,
        depth_map: np.ndarray,
        pose: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        frame_idx: int = 0,
    ):
        """
        Add a new frame to the sliding window.
        
        Args:
            depth_map: Depth map (H, W)
            pose: Camera pose (4, 4)
            confidence_map: Confidence map (H, W), defaults to uniform 1.0
            frame_idx: Frame index
        """
        if confidence_map is None:
            confidence_map = np.ones_like(depth_map)
        
        self.depth_window.append(depth_map.copy())
        self.pose_window.append(pose.copy())
        self.confidence_window.append(confidence_map.copy())
        self.frame_idx_window.append(frame_idx)
    
    def fuse_to_current_frame(
        self,
        K: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Fuse depth maps from sliding window to the current (most recent) frame.
        
        Args:
            K: Camera intrinsics (3, 3)
        
        Returns:
            fused_depth: Fused depth map (H, W)
            fused_confidence: Fused confidence map (H, W)
            stats: Fusion statistics
        """
        if len(self.depth_window) == 0:
            raise ValueError("No frames in window")
        
        # Current frame is the last one added
        current_idx = len(self.depth_window) - 1
        current_pose = self.pose_window[current_idx]
        current_depth = self.depth_window[current_idx]
        H, W = current_depth.shape
        
        # Collect warped depth maps
        depth_maps = []
        confidence_maps = []
        valid_masks = []
        
        for i in range(len(self.depth_window)):
            depth = self.depth_window[i]
            pose = self.pose_window[i]
            confidence = self.confidence_window[i]
            
            # Check minimum confidence
            avg_confidence = confidence.mean()
            if avg_confidence < self.min_confidence:
                continue
            
            if i == current_idx:
                # Current frame - no warping needed
                depth_maps.append(depth)
                confidence_maps.append(confidence)
                valid_masks.append(depth > 0)
            else:
                # Warp to current frame
                warped_depth, valid_mask = warp_depth_to_frame(
                    depth, pose, current_pose, K
                )
                
                # Only use if enough valid pixels
                valid_ratio = valid_mask.sum() / (H * W)
                if valid_ratio > 0.1:  # At least 10% valid
                    depth_maps.append(warped_depth)
                    confidence_maps.append(confidence)
                    valid_masks.append(valid_mask)
        
        # Reject temporal outliers
        if len(depth_maps) >= 3:
            valid_masks = reject_depth_outliers_temporal(
                depth_maps, valid_masks, self.outlier_threshold_mm
            )
        
        # Fuse with confidence weighting
        if not depth_maps:
            # Fallback: return current frame only
            return current_depth, self.confidence_window[current_idx], {
                "num_frames_used": 1,
                "valid_ratio": (current_depth > 0).sum() / (H * W)
            }
        
        fused_depth, fused_confidence = fuse_depth_maps_weighted(
            depth_maps, confidence_maps, valid_masks
        )
        
        # Update statistics
        self.total_fusions += 1
        self.total_frames_used += len(depth_maps)
        self.avg_frames_per_fusion = self.total_frames_used / self.total_fusions
        
        stats = {
            "num_frames_used": len(depth_maps),
            "valid_ratio": (fused_depth > 0).sum() / (H * W),
            "avg_frames_per_fusion": self.avg_frames_per_fusion,
        }
        
        return fused_depth, fused_confidence, stats
    
    def get_statistics(self) -> Dict:
        """Get fusion statistics"""
        return {
            "total_fusions": self.total_fusions,
            "total_frames_used": self.total_frames_used,
            "avg_frames_per_fusion": self.avg_frames_per_fusion,
            "window_size": len(self.depth_window),
        }
    
    def clear(self):
        """Clear the sliding window"""
        self.depth_window.clear()
        self.pose_window.clear()
        self.confidence_window.clear()
        self.frame_idx_window.clear()
