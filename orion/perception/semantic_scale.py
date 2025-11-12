#!/usr/bin/env python3
"""
Semantic Scale Recovery & Ground Plane Detection
=================================================

Fixes the scale ambiguity problem in monocular SLAM by:
1. Using semantic priors (known object sizes)
2. Detecting ground plane for scale reference
3. Temporal depth fusion for stability

Real-time optimized: ~5-10ms overhead

Author: Orion Research
Date: November 11, 2025
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from collections import deque


# COCO object size priors (in meters: width, height, depth)
# Based on typical real-world measurements
OBJECT_SIZE_PRIORS = {
    # Electronics
    'laptop': (0.35, 0.25, 0.02),
    'keyboard': (0.45, 0.15, 0.03),
    'mouse': (0.10, 0.06, 0.04),
    'cell phone': (0.07, 0.15, 0.01),
    'remote': (0.05, 0.18, 0.02),
    'tv': (1.0, 0.6, 0.15),
    'monitor': (0.6, 0.4, 0.1),
    
    # Furniture
    'chair': (0.5, 0.9, 0.5),
    'couch': (2.0, 0.8, 0.9),
    'bed': (2.0, 0.6, 1.5),
    'dining table': (1.5, 0.75, 0.9),
    'desk': (1.2, 0.75, 0.6),
    
    # Kitchen
    'bottle': (0.08, 0.25, 0.08),
    'cup': (0.08, 0.10, 0.08),
    'bowl': (0.15, 0.08, 0.15),
    'microwave': (0.5, 0.3, 0.4),
    'oven': (0.6, 0.6, 0.6),
    'refrigerator': (0.8, 1.7, 0.7),
    
    # Personal items
    'backpack': (0.3, 0.45, 0.2),
    'handbag': (0.35, 0.3, 0.15),
    'suitcase': (0.5, 0.7, 0.25),
    'book': (0.15, 0.23, 0.03),
    
    # People & animals
    'person': (0.5, 1.7, 0.3),
    'dog': (0.4, 0.6, 0.3),
    'cat': (0.25, 0.25, 0.15),
    
    # Default for unknown
    'unknown': (0.3, 0.3, 0.3)
}


class SemanticScaleRecovery:
    """
    Recovers metric scale using semantic object priors
    
    Real-time: ~2-5ms per frame
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Args:
            confidence_threshold: Min confidence to use for scale
        """
        self.confidence_threshold = confidence_threshold
        self.scale_history = deque(maxlen=30)  # 1 second at 30fps
        
    def correct_depth(self,
                     detections: List[Dict],
                     depth_map: np.ndarray,
                     intrinsics: Dict[str, float]) -> Tuple[np.ndarray, Dict]:
        """
        Correct depth map using semantic priors
        
        Args:
            detections: List of detections with bbox, class, score
            depth_map: Raw depth map (H, W) in meters
            intrinsics: Camera intrinsics (fx, fy, cx, cy)
        
        Returns:
            corrected_depth: Scale-corrected depth map
            stats: Statistics dict
        """
        if not detections:
            return depth_map, {'scale_factor': 1.0, 'n_anchors': 0}
        
        # Collect scale estimates from high-confidence objects
        scale_estimates = []
        
        fx = intrinsics.get('fx', 500)
        fy = intrinsics.get('fy', 500)
        
        for det in detections:
            if det['score'] < self.confidence_threshold:
                continue
            
            class_name = det['class']
            if class_name not in OBJECT_SIZE_PRIORS:
                # print(f"[ScaleRecovery] Skipping {class_name} - no size prior")
                continue
            
            # Get expected size
            expected_w, expected_h, expected_d = OBJECT_SIZE_PRIORS[class_name]
            
            # Get bbox
            x1, y1, x2, y2 = det['bbox']
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # Skip tiny boxes (likely false positives)
            if bbox_w < 20 or bbox_h < 20:
                continue
            
            # Get median depth in bbox
            y1_int, y2_int = int(y1), int(y2)
            x1_int, x2_int = int(x1), int(x2)
            
            # Clip to image bounds
            y1_int = max(0, min(depth_map.shape[0]-1, y1_int))
            y2_int = max(0, min(depth_map.shape[0]-1, y2_int))
            x1_int = max(0, min(depth_map.shape[1]-1, x1_int))
            x2_int = max(0, min(depth_map.shape[1]-1, x2_int))
            
            if y2_int <= y1_int or x2_int <= x1_int:
                continue
            
            depth_roi = depth_map[y1_int:y2_int, x1_int:x2_int]
            if depth_roi.size == 0:
                continue
                
            depth_median = np.median(depth_roi)
            
            if depth_median < 0.1 or depth_median > 50:
                continue
            
            # Estimate depth from expected size (pinhole camera model)
            # depth = (focal_length * real_size) / pixel_size
            depth_from_width = (fx * expected_w) / bbox_w
            depth_from_height = (fy * expected_h) / bbox_h
            
            # Average the two estimates
            depth_expected = (depth_from_width + depth_from_height) / 2
            
            # Compute scale factor
            if depth_median > 0.1:
                scale = depth_expected / depth_median
                
                # Sanity check (scale shouldn't be crazy)
                # Allow 0.2x to 5.0x scale adjustments (indoor scenes can vary a lot)
                if 0.2 < scale < 5.0:
                    weight = det['score']  # Weight by confidence
                    scale_estimates.append((scale, weight))
                    # print(f"[ScaleRecovery] ✓ {class_name}: scale={scale:.2f}, depth_median={depth_median:.2f}m, expected={depth_expected:.2f}m")
                # else:
                #     print(f"[ScaleRecovery] ✗ {class_name}: scale {scale:.2f} out of range [0.2, 5.0]")
            # else:
            #     print(f"[ScaleRecovery] {class_name}: depth_median {depth_median:.2f}m too small")
        
        # Compute weighted median scale
        if len(scale_estimates) >= 2:
            scales = np.array([s for s, w in scale_estimates])
            weights = np.array([w for s, w in scale_estimates])
            
            # Weighted median
            sorted_idx = np.argsort(scales)
            scales_sorted = scales[sorted_idx]
            weights_sorted = weights[sorted_idx]
            cumsum = np.cumsum(weights_sorted)
            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
            scale_factor = scales_sorted[median_idx]
            
            # Add to history
            self.scale_history.append(scale_factor)
            
            # Temporal smoothing (EMA)
            if len(self.scale_history) > 5:
                scale_factor = np.median(list(self.scale_history)[-10:])
        else:
            # Use previous scale or 1.0
            scale_factor = self.scale_history[-1] if self.scale_history else 1.0
        
        # Apply scale correction
        corrected_depth = depth_map * scale_factor
        
        stats = {
            'scale_factor': float(scale_factor),
            'n_anchors': len(scale_estimates),
            'scale_std': float(np.std([s for s, w in scale_estimates])) if scale_estimates else 0.0
        }
        
        return corrected_depth, stats


class GroundPlaneDetector:
    """
    Detects ground plane from depth map for scale reference
    
    Real-time: ~3-8ms per frame
    """
    
    def __init__(self, 
                 camera_height_prior: float = 1.5,
                 plane_tolerance: float = 0.1):
        """
        Args:
            camera_height_prior: Expected camera height above ground (meters)
            plane_tolerance: RANSAC tolerance for plane fitting (meters)
        """
        self.camera_height_prior = camera_height_prior
        self.plane_tolerance = plane_tolerance
        self.ground_plane_history = deque(maxlen=10)
        
    def detect(self,
               depth_map: np.ndarray,
               camera_pose: Optional[np.ndarray] = None,
               downsample: int = 4) -> Optional[Dict]:
        """
        Detect ground plane using RANSAC on depth map
        
        Args:
            depth_map: Depth map (H, W) in meters
            camera_pose: 4x4 camera pose matrix (optional)
            downsample: Downsample factor for speed
        
        Returns:
            plane_params: Dict with normal, point, height, confidence
        """
        h, w = depth_map.shape
        
        # Downsample for speed
        depth_small = depth_map[::downsample, ::downsample]
        h_small, w_small = depth_small.shape
        
        # Focus on bottom third (likely ground)
        y_start = int(h_small * 0.6)
        depth_ground = depth_small[y_start:, :]
        
        # Generate 3D points (simple pinhole model)
        fx = fy = 500  # Approximate
        cx, cy = w_small / 2, h_small / 2
        
        y_coords, x_coords = np.meshgrid(
            np.arange(y_start, h_small),
            np.arange(w_small),
            indexing='ij'
        )
        
        # Valid depth mask
        valid = (depth_ground > 0.3) & (depth_ground < 10.0)
        
        if valid.sum() < 100:
            return None
        
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        depths = depth_ground[valid]
        
        # Backproject to 3D
        X = (x_coords - cx) * depths / fx
        Y = (y_coords - cy) * depths / fy
        Z = depths
        
        points = np.stack([X, Y, Z], axis=-1)
        
        # RANSAC plane fitting
        best_plane = None
        best_inliers = 0
        n_iterations = 50
        
        for _ in range(n_iterations):
            # Sample 3 random points
            if len(points) < 3:
                break
                
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
                
            normal = normal / np.linalg.norm(normal)
            
            # Compute distances to plane
            distances = np.abs((points - p1) @ normal)
            
            # Count inliers
            inliers = distances < self.plane_tolerance
            n_inliers = inliers.sum()
            
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_plane = {
                    'normal': normal,
                    'point': p1,
                    'inliers': inliers,
                    'confidence': n_inliers / len(points)
                }
        
        if best_plane is None or best_plane['confidence'] < 0.3:
            return None
        
        # Estimate camera height above ground
        # Assume ground normal points up (Y-axis in camera coords)
        camera_height = abs(np.dot(best_plane['point'], best_plane['normal']))
        
        best_plane['height'] = camera_height
        
        # Add to history for stability
        self.ground_plane_history.append(best_plane)
        
        return best_plane
    
    def get_stable_ground(self) -> Optional[Dict]:
        """Get temporally averaged ground plane"""
        if not self.ground_plane_history:
            return None
        
        # Average normals and heights
        normals = [p['normal'] for p in self.ground_plane_history]
        heights = [p['height'] for p in self.ground_plane_history]
        
        avg_normal = np.mean(normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        avg_height = np.median(heights)
        
        return {
            'normal': avg_normal,
            'height': avg_height,
            'confidence': np.mean([p['confidence'] for p in self.ground_plane_history])
        }


class TemporalDepthFusion:
    """
    Fuses depth maps temporally for stability
    
    Real-time: ~5-15ms per frame (depending on resolution)
    """
    
    def __init__(self, window_size: int = 5, weight_decay: float = 0.8):
        """
        Args:
            window_size: Number of frames to fuse
            weight_decay: Exponential decay for older frames
        """
        self.window_size = window_size
        self.weight_decay = weight_decay
        self.depth_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)
        
    def fuse(self,
             depth_map: np.ndarray,
             confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fuse current depth with history
        
        Args:
            depth_map: Current depth map (H, W)
            confidence: Optional confidence map (H, W)
        
        Returns:
            fused_depth: Temporally smoothed depth
        """
        if confidence is None:
            confidence = np.ones_like(depth_map)
        
        # Add to buffer
        self.depth_buffer.append(depth_map.copy())
        self.confidence_buffer.append(confidence.copy())
        
        if len(self.depth_buffer) == 1:
            return depth_map
        
        # Weighted average with exponential decay
        weights = np.array([self.weight_decay ** i for i in range(len(self.depth_buffer))])
        weights = weights[::-1]  # Recent frames have higher weight
        weights = weights / weights.sum()
        
        # Fuse depths
        fused = np.zeros_like(depth_map, dtype=np.float32)
        
        for i, (depth, conf) in enumerate(zip(self.depth_buffer, self.confidence_buffer)):
            fused += depth * conf * weights[i]
        
        # Normalize by confidence sum
        conf_sum = sum(conf * weights[i] for i, conf in enumerate(self.confidence_buffer))
        fused = fused / (conf_sum + 1e-8)
        
        return fused
    
    def reset(self):
        """Clear history"""
        self.depth_buffer.clear()
        self.confidence_buffer.clear()
