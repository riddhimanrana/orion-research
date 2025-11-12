#!/usr/bin/env python3
"""
Motion Blur Detection and Quality Assessment
=============================================

Detects motion blur and provides frame quality metrics.
Used to skip or weight frames appropriately.

Author: Orion Research
Date: November 11, 2025
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class MotionBlurDetector:
    """
    Detects motion blur using Laplacian variance
    
    Thresholds (Laplacian variance):
    - > 100: Sharp, good quality
    - 50-100: Slight blur, acceptable
    - 20-50: Moderate blur, reduced confidence
    - < 20: Severe blur, skip frame
    """
    
    def __init__(self,
                 sharp_threshold: float = 100.0,
                 blur_threshold: float = 20.0):
        """
        Args:
            sharp_threshold: Threshold for "sharp" classification
            blur_threshold: Threshold below which frame is considered blurred
        """
        self.sharp_threshold = sharp_threshold
        self.blur_threshold = blur_threshold
    
    def assess_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Assess frame quality
        
        Args:
            frame: RGB or BGR frame
        
        Returns:
            Dict with 'sharpness', 'is_sharp', 'is_blurred', 'confidence_weight'
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Compute Laplacian variance (higher = sharper)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Classify
        is_sharp = sharpness >= self.sharp_threshold
        is_blurred = sharpness < self.blur_threshold
        
        # Compute confidence weight (0-1, based on sharpness)
        if sharpness >= self.sharp_threshold:
            confidence_weight = 1.0
        elif sharpness >= self.blur_threshold:
            # Linear interpolation
            confidence_weight = (sharpness - self.blur_threshold) / (
                self.sharp_threshold - self.blur_threshold
            )
        else:
            confidence_weight = 0.2  # Very low but not zero
        
        return {
            'sharpness': float(sharpness),
            'is_sharp': bool(is_sharp),
            'is_blurred': bool(is_blurred),
            'confidence_weight': float(confidence_weight),
            'quality': self._get_quality_label(sharpness)
        }
    
    def _get_quality_label(self, sharpness: float) -> str:
        """Get human-readable quality label"""
        if sharpness >= 100:
            return "excellent"
        elif sharpness >= 50:
            return "good"
        elif sharpness >= 20:
            return "fair"
        else:
            return "poor"
    
    def should_skip_frame(self, frame: np.ndarray) -> bool:
        """Check if frame should be skipped due to blur"""
        assessment = self.assess_frame(frame)
        return assessment['is_blurred']
    
    def get_confidence_multiplier(self, frame: np.ndarray) -> float:
        """Get confidence multiplier for detections (0-1)"""
        assessment = self.assess_frame(frame)
        return assessment['confidence_weight']


class TemporalDepthFusion:
    """
    Fuses depth maps across time for stability
    
    Uses optical flow to align depths and averages them.
    """
    
    def __init__(self, buffer_size: int = 5):
        """
        Args:
            buffer_size: Number of frames to keep in history
        """
        self.buffer_size = buffer_size
        self.depth_buffer = []
        self.frame_buffer = []
        self.flow_estimator = None  # Optional: add RAFT-Lite
    
    def add_frame(self, frame: np.ndarray, depth: np.ndarray):
        """Add frame and depth to buffer"""
        self.frame_buffer.append(frame)
        self.depth_buffer.append(depth)
        
        # Keep only last N frames
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
            self.depth_buffer.pop(0)
    
    def get_fused_depth(self) -> np.ndarray:
        """
        Get temporally fused depth map
        
        Returns:
            Fused depth map (average of aligned depths)
        """
        if not self.depth_buffer:
            return None
        
        if len(self.depth_buffer) == 1:
            return self.depth_buffer[0]
        
        # Simple averaging (TODO: add optical flow alignment)
        weights = np.exp(np.linspace(-1, 0, len(self.depth_buffer)))
        weights /= weights.sum()
        
        fused = np.zeros_like(self.depth_buffer[0])
        for depth, weight in zip(self.depth_buffer, weights):
            fused += depth * weight
        
        return fused
    
    def reset(self):
        """Clear buffer"""
        self.frame_buffer.clear()
        self.depth_buffer.clear()
