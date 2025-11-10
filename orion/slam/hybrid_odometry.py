"""
Hybrid Visual-Depth Odometry with Confidence-Based Fusion

Combines visual odometry (accurate rotation) with depth odometry (accurate scale)
using uncertainty-weighted fusion for optimal pose estimation.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R


class HybridOdometry:
    """
    Fuses visual and depth-based pose estimates with confidence weighting.
    
    Strategy:
    - Visual odometry: Better for rotation and textured scenes
    - Depth odometry: Better for scale and low-texture scenes
    - Fusion: Confidence-weighted combination of both
    """
    
    def __init__(
        self,
        rotation_weight_visual: float = 0.8,  # Prefer visual for rotation
        translation_fusion_mode: str = "weighted",  # "weighted", "visual", "depth"
        min_confidence_threshold: float = 0.3,  # Switch to single mode if other fails
    ):
        """
        Initialize hybrid odometry system.
        
        Args:
            rotation_weight_visual: Weight for visual rotation (0-1)
            translation_fusion_mode: How to fuse translations
            min_confidence_threshold: Minimum confidence to use a pose estimate
        """
        self.rotation_weight_visual = rotation_weight_visual
        self.translation_fusion_mode = translation_fusion_mode
        self.min_confidence_threshold = min_confidence_threshold
        
        # Statistics
        self.total_fusions = 0
        self.visual_only_count = 0
        self.depth_only_count = 0
        self.fusion_count = 0
    
    def fuse_poses(
        self,
        visual_pose: Optional[np.ndarray],
        depth_pose: Optional[np.ndarray],
        visual_confidence: float,
        depth_confidence: float,
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Fuse visual and depth poses with confidence weighting.
        
        Args:
            visual_pose: 4x4 pose matrix from visual odometry
            depth_pose: 4x4 pose matrix from depth odometry
            visual_confidence: Confidence in visual estimate [0, 1]
            depth_confidence: Confidence in depth estimate [0, 1]
        
        Returns:
            (fused_pose, mode) where mode is "visual", "depth", or "fusion"
        """
        self.total_fusions += 1
        
        # Handle missing estimates
        if visual_pose is None and depth_pose is None:
            return None, "none"
        
        # Use only available estimate
        if visual_pose is None:
            self.depth_only_count += 1
            return depth_pose, "depth"
        
        if depth_pose is None:
            self.visual_only_count += 1
            return visual_pose, "visual"
        
        # Check confidence thresholds
        if visual_confidence < self.min_confidence_threshold:
            self.depth_only_count += 1
            return depth_pose, "depth"
        
        if depth_confidence < self.min_confidence_threshold:
            self.visual_only_count += 1
            return visual_pose, "visual"
        
        # Both estimates available and confident - fuse them
        self.fusion_count += 1
        
        # Extract rotation and translation
        R_visual = visual_pose[:3, :3]
        t_visual = visual_pose[:3, 3]
        
        R_depth = depth_pose[:3, :3]
        t_depth = depth_pose[:3, 3]
        
        # Fuse rotation (prefer visual, but blend with SLERP)
        R_fused = self._fuse_rotations(R_visual, R_depth, self.rotation_weight_visual)
        
        # Fuse translation (confidence-weighted)
        t_fused = self._fuse_translations(
            t_visual, t_depth, visual_confidence, depth_confidence
        )
        
        # Build fused pose
        fused_pose = np.eye(4)
        fused_pose[:3, :3] = R_fused
        fused_pose[:3, 3] = t_fused
        
        return fused_pose, "fusion"
    
    def _fuse_rotations(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        weight1: float
    ) -> np.ndarray:
        """
        Fuse two rotation matrices using SLERP (Spherical Linear Interpolation).
        
        Args:
            R1: First rotation matrix (3x3)
            R2: Second rotation matrix (3x3)
            weight1: Weight for R1 [0, 1], weight2 = 1 - weight1
        
        Returns:
            Fused rotation matrix (3x3)
        """
        try:
            # Convert to quaternions
            rot1 = R.from_matrix(R1)
            rot2 = R.from_matrix(R2)
            
            # SLERP interpolation
            # weight1=1.0 → pure R1, weight1=0.0 → pure R2
            key_times = [0, 1]
            key_rots = R.from_quat([rot1.as_quat(), rot2.as_quat()])
            slerp = R.from_quat(key_rots.as_quat())
            
            # Interpolate at weight1
            fused_rot = slerp.from_quat(
                rot1.as_quat() * weight1 + rot2.as_quat() * (1 - weight1)
            )
            
            return fused_rot.as_matrix()
        
        except Exception as e:
            # Fallback: use weighted average (less accurate but robust)
            print(f"[HybridOdom] SLERP failed: {e}, using weighted average")
            return R1 * weight1 + R2 * (1 - weight1)
    
    def _fuse_translations(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        conf1: float,
        conf2: float
    ) -> np.ndarray:
        """
        Fuse two translation vectors with confidence weighting.
        
        Args:
            t1: First translation (3,)
            t2: Second translation (3,)
            conf1: Confidence in t1 [0, 1]
            conf2: Confidence in t2 [0, 1]
        
        Returns:
            Fused translation (3,)
        """
        if self.translation_fusion_mode == "visual":
            return t1
        elif self.translation_fusion_mode == "depth":
            return t2
        else:  # "weighted"
            # Normalize confidences
            total_conf = conf1 + conf2
            if total_conf < 1e-6:
                # Equal weighting if both very low confidence
                w1, w2 = 0.5, 0.5
            else:
                w1 = conf1 / total_conf
                w2 = conf2 / total_conf
            
            return w1 * t1 + w2 * t2
    
    def estimate_visual_confidence(
        self,
        inlier_ratio: float,
        num_matches: int,
        texture_score: Optional[float] = None,
    ) -> float:
        """
        Estimate confidence in visual odometry estimate.
        
        Args:
            inlier_ratio: Ratio of inliers in RANSAC [0, 1]
            num_matches: Number of feature matches
            texture_score: Optional texture richness score [0, 1]
        
        Returns:
            Confidence score [0, 1]
        """
        # Base confidence from inliers (60% weight)
        conf = inlier_ratio * 0.6
        
        # Boost for many matches (30% weight)
        # Assume 100+ matches is ideal
        match_factor = min(num_matches / 100.0, 1.0) * 0.3
        conf += match_factor
        
        # Texture score (10% weight) if available
        if texture_score is not None:
            conf += texture_score * 0.1
        else:
            # Default texture score
            conf += 0.05
        
        return min(conf, 1.0)
    
    def estimate_depth_confidence(
        self,
        uncertainty_map: Optional[np.ndarray],
        valid_ratio: float,
        depth_range_ok: bool = True,
    ) -> float:
        """
        Estimate confidence in depth odometry estimate.
        
        Args:
            uncertainty_map: Per-pixel uncertainty [0, 1] (HxW)
            valid_ratio: Ratio of valid depth pixels [0, 1]
            depth_range_ok: Whether depth is in valid range
        
        Returns:
            Confidence score [0, 1]
        """
        if not depth_range_ok:
            return 0.0
        
        # Base confidence from valid pixels (40% weight)
        conf = valid_ratio * 0.4
        
        # Average certainty (60% weight)
        if uncertainty_map is not None:
            # Only consider pixels with reasonable uncertainty
            valid_uncert = uncertainty_map[uncertainty_map < 0.8]
            if len(valid_uncert) > 0:
                avg_certainty = np.mean(1.0 - valid_uncert)
                conf += avg_certainty * 0.6
            else:
                # No valid uncertainty data
                conf += 0.3  # Medium confidence
        else:
            # No uncertainty map - assume medium confidence
            conf += 0.4
        
        return min(conf, 1.0)
    
    def get_statistics(self) -> dict:
        """Get fusion statistics"""
        if self.total_fusions == 0:
            return {
                "total_fusions": 0,
                "visual_only": 0.0,
                "depth_only": 0.0,
                "fusion": 0.0,
            }
        
        return {
            "total_fusions": self.total_fusions,
            "visual_only": self.visual_only_count / self.total_fusions,
            "depth_only": self.depth_only_count / self.total_fusions,
            "fusion": self.fusion_count / self.total_fusions,
        }


def compute_texture_score(image: np.ndarray) -> float:
    """
    Compute texture richness score for an image.
    
    Higher score = more texture = better for visual odometry
    
    Args:
        image: Grayscale image (HxW)
    
    Returns:
        Texture score [0, 1]
    """
    import cv2
    
    # Compute Laplacian variance (measure of sharpness/texture)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize (empirically, variance > 100 is textured)
    texture_score = min(variance / 100.0, 1.0)
    
    return texture_score


def check_depth_range(depth_map: np.ndarray, min_depth: float = 100.0, max_depth: float = 10000.0) -> bool:
    """
    Check if depth map has sufficient valid depth measurements.
    
    Args:
        depth_map: Depth map in millimeters (HxW)
        min_depth: Minimum valid depth (default: 100mm = 10cm)
        max_depth: Maximum valid depth (default: 10000mm = 10m)
    
    Returns:
        True if >50% of pixels have valid depth
    """
    if depth_map is None or depth_map.size == 0:
        return False
    
    # Count pixels in valid range
    valid_depth = depth_map[(depth_map > min_depth) & (depth_map < max_depth)]
    valid_ratio = len(valid_depth) / depth_map.size
    
    # Require at least 50% valid pixels
    return valid_ratio > 0.5
