"""
Absolute Scale Recovery for Monocular SLAM

This module recovers absolute metric scale using object size priors.
Monocular SLAM can only estimate relative scale; this makes it absolute.

Strategy:
1. Detect objects with known real-world sizes
2. Measure their pixel/depth dimensions
3. Estimate scale factor: real_size / (pixel_size * depth)
4. Average multiple estimates for robustness
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Real-world object size priors (in meters)
# These are average/typical sizes for common indoor objects
OBJECT_SIZE_PRIORS = {
    # People
    'person': {'height': 1.70, 'width': 0.50, 'confidence': 0.85},
    
    # Furniture
    'chair': {'height': 0.90, 'width': 0.50, 'depth': 0.50, 'confidence': 0.75},
    'couch': {'height': 0.85, 'width': 2.00, 'depth': 0.90, 'confidence': 0.80},
    'sofa': {'height': 0.85, 'width': 2.00, 'depth': 0.90, 'confidence': 0.80},
    'bed': {'height': 0.60, 'width': 1.50, 'depth': 2.00, 'confidence': 0.75},
    'dining table': {'height': 0.75, 'width': 1.50, 'depth': 0.90, 'confidence': 0.80},
    'table': {'height': 0.75, 'width': 1.20, 'depth': 0.80, 'confidence': 0.70},
    
    # Electronics
    'tv': {'width': 1.20, 'height': 0.70, 'confidence': 0.70},  # ~55 inch
    'laptop': {'width': 0.35, 'depth': 0.25, 'height': 0.02, 'confidence': 0.90},
    'keyboard': {'width': 0.45, 'depth': 0.15, 'confidence': 0.85},
    'mouse': {'width': 0.06, 'depth': 0.10, 'confidence': 0.80},
    'cell phone': {'height': 0.15, 'width': 0.07, 'confidence': 0.85},
    
    # Appliances
    'refrigerator': {'height': 1.80, 'width': 0.70, 'depth': 0.70, 'confidence': 0.85},
    'microwave': {'width': 0.50, 'depth': 0.40, 'height': 0.30, 'confidence': 0.80},
    'oven': {'width': 0.60, 'height': 0.85, 'depth': 0.60, 'confidence': 0.80},
    
    # Architecture (most reliable!)
    'door': {'height': 2.10, 'width': 0.90, 'confidence': 0.95},
    'window': {'height': 1.50, 'width': 1.20, 'confidence': 0.75},
    
    # Small objects
    'bottle': {'height': 0.25, 'diameter': 0.07, 'confidence': 0.75},
    'cup': {'height': 0.12, 'diameter': 0.08, 'confidence': 0.70},
    'wine glass': {'height': 0.20, 'diameter': 0.08, 'confidence': 0.75},
    'book': {'height': 0.23, 'width': 0.15, 'confidence': 0.70},
    
    # Plants
    'potted plant': {'height': 0.40, 'width': 0.25, 'confidence': 0.60},
}


@dataclass
class ScaleEstimate:
    """Single scale estimation from one object"""
    scale: float  # meters per unit
    confidence: float  # 0-1
    source_class: str  # Object class used
    method: str  # 'height', 'width', 'depth'
    frame_idx: int


class ScaleEstimator:
    """
    Estimates absolute metric scale from object size priors
    
    Strategy:
    - Collect multiple scale estimates from different objects
    - Filter outliers
    - Take robust average (median)
    - Require minimum confidence before committing
    """
    
    def __init__(
        self,
        min_estimates: int = 10,
        confidence_threshold: float = 0.7,
        outlier_threshold: float = 2.0  # standard deviations
    ):
        """
        Initialize scale estimator
        
        Args:
            min_estimates: Minimum number of estimates before committing to scale
            confidence_threshold: Minimum confidence to accept estimate
            outlier_threshold: Z-score threshold for outlier removal
        """
        self.min_estimates = min_estimates
        self.confidence_threshold = confidence_threshold
        self.outlier_threshold = outlier_threshold
        
        # Storage
        self.estimates: List[ScaleEstimate] = []
        self.committed_scale: Optional[float] = None
        self.scale_locked = False
        
        logger.info(f"[ScaleEstimator] Initialized (min_estimates={min_estimates}, threshold={confidence_threshold})")
    
    def estimate_from_object(
        self,
        bbox: Tuple[float, float, float, float],  # x1, y1, x2, y2
        depth_roi: np.ndarray,  # Depth map region for this object
        class_name: str,
        frame_idx: int
    ) -> Optional[ScaleEstimate]:
        """
        Estimate scale from a single detected object
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2) in pixels
            depth_roi: Depth values within bbox (in mm)
            class_name: YOLO class name
            frame_idx: Current frame number
            
        Returns:
            ScaleEstimate if successful, None otherwise
        """
        # Check if we have size priors for this class
        if class_name not in OBJECT_SIZE_PRIORS:
            return None
        
        size_prior = OBJECT_SIZE_PRIORS[class_name]
        
        # Get bbox dimensions in pixels
        bbox_width_px = bbox[2] - bbox[0]
        bbox_height_px = bbox[3] - bbox[1]
        
        # Get median depth (robust to outliers)
        valid_depth = depth_roi[depth_roi > 0]
        if len(valid_depth) < 10:  # Too few depth points
            return None
        
        median_depth_mm = np.median(valid_depth)
        median_depth_m = median_depth_mm / 1000.0
        
        # Skip if depth is unrealistic
        if median_depth_m < 0.3 or median_depth_m > 15.0:
            return None
        
        # Estimate scale using different dimensions
        scale_estimates = []
        
        # Method 1: Height-based (most reliable for vertical objects)
        if 'height' in size_prior and bbox_height_px > 50:  # Min size threshold
            # In SLAM, we have 3D positions in mm
            # The bbox height in 3D space (in mm) ≈ bbox_height_px * depth_mm / focal_length_px
            # For simplicity, assume focal_length ≈ image_height (typical for normalized cameras)
            # Approximate: 3D_height_mm ≈ bbox_height_px * depth_mm / 1000
            # 
            # We want: scale = real_height_m / 3D_height_mm
            # So: scale ≈ real_height_m / (bbox_height_px * depth_mm / 1000)
            # Simplify: scale = real_height_m * 1000 / (bbox_height_px * depth_mm)
            
            scale_h = (size_prior['height'] * 1000.0) / (bbox_height_px * median_depth_mm)
            scale_estimates.append(('height', scale_h, 0.9 * size_prior['confidence']))
        
        # Method 2: Width-based
        if 'width' in size_prior and bbox_width_px > 50:
            scale_w = (size_prior['width'] * 1000.0) / (bbox_width_px * median_depth_mm)
            scale_estimates.append(('width', scale_w, 0.85 * size_prior['confidence']))
        
        # Method 3: Depth-based (for objects with known depth)
        # (More complex, skipping for now)
        
        if not scale_estimates:
            return None
        
        # Weighted average of estimates
        total_weight = sum(s[2] for s in scale_estimates)
        avg_scale = sum(s[1] * s[2] for s in scale_estimates) / total_weight
        avg_confidence = total_weight / len(scale_estimates)
        best_method = max(scale_estimates, key=lambda x: x[2])[0]
        
        # Sanity check: scale should be reasonable
        # Typical scale for monocular SLAM: 0.1 - 10.0 meters/unit
        if not (0.05 < avg_scale < 20.0):
            return None
        
        return ScaleEstimate(
            scale=avg_scale,
            confidence=avg_confidence,
            source_class=class_name,
            method=best_method,
            frame_idx=frame_idx
        )
    
    def add_estimate(self, estimate: ScaleEstimate):
        """
        Add a scale estimate to the collection
        
        Args:
            estimate: Scale estimate from object
        """
        if estimate.confidence < self.confidence_threshold:
            return  # Reject low confidence
        
        self.estimates.append(estimate)
        
        # Try to commit scale if we have enough estimates
        if not self.scale_locked and len(self.estimates) >= self.min_estimates:
            self._try_commit_scale()
    
    def _try_commit_scale(self):
        """Try to commit to a final scale estimate"""
        if len(self.estimates) < self.min_estimates:
            return
        
        # Extract scale values
        scales = np.array([e.scale for e in self.estimates])
        confidences = np.array([e.confidence for e in self.estimates])
        
        # Remove outliers using MAD (Median Absolute Deviation)
        median_scale = np.median(scales)
        mad = np.median(np.abs(scales - median_scale))
        
        if mad > 0:
            # Modified z-score
            modified_z = 0.6745 * (scales - median_scale) / mad
            inlier_mask = np.abs(modified_z) < self.outlier_threshold
            
            scales_filtered = scales[inlier_mask]
            confidences_filtered = confidences[inlier_mask]
        else:
            scales_filtered = scales
            confidences_filtered = confidences
        
        # Require at least 50% inliers
        if len(scales_filtered) < len(scales) * 0.5:
            logger.warning(f"[ScaleEstimator] Too many outliers ({len(scales) - len(scales_filtered)}/{len(scales)})")
            return
        
        # Weighted average (by confidence)
        final_scale = np.average(scales_filtered, weights=confidences_filtered)
        
        # Compute confidence (based on agreement)
        scale_std = np.std(scales_filtered)
        agreement_confidence = np.exp(-scale_std / final_scale)  # Higher std = lower confidence
        
        # Overall confidence
        avg_confidence = np.mean(confidences_filtered)
        final_confidence = 0.7 * avg_confidence + 0.3 * agreement_confidence
        
        # Commit if confident enough
        if final_confidence > 0.7:
            self.committed_scale = float(final_scale)
            self.scale_locked = True
            
            logger.info(f"✓ Absolute scale established: {final_scale:.3f} m/unit")
            logger.info(f"  Confidence: {final_confidence:.2f}")
            logger.info(f"  Based on {len(scales_filtered)} estimates")
            logger.info(f"  Sources: {set(e.source_class for e in self.estimates)}")
    
    def get_scale(self) -> Optional[float]:
        """
        Get committed scale, or best estimate if not yet committed
        
        Returns:
            Scale factor (meters/unit), or None if not enough data
        """
        if self.committed_scale is not None:
            return self.committed_scale
        
        # Return provisional estimate if we have some data
        if len(self.estimates) >= 3:
            scales = np.array([e.scale for e in self.estimates])
            return float(np.median(scales))
        
        return None
    
    def is_locked(self) -> bool:
        """Check if scale has been committed"""
        return self.scale_locked
    
    def get_statistics(self) -> Dict:
        """Get scale estimation statistics"""
        return {
            'total_estimates': len(self.estimates),
            'scale_locked': self.scale_locked,
            'committed_scale': self.committed_scale,
            'provisional_scale': np.median([e.scale for e in self.estimates]) if self.estimates else None,
            'source_classes': list(set(e.source_class for e in self.estimates)),
        }
