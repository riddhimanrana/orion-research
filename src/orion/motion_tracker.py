"""
Motion Tracking Utilities
==========================

This module provides utilities for tracking object motion across frames,
calculating velocities, and detecting directional movement patterns.
These are essential for the Causal Influence Score (CIS) computation.

Author: Orion Research Team
Date: October 2025
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MotionData:
    """
    Encapsulates motion information for an object across frames
    """
    
    centroid: Tuple[float, float]  # (x, y) center of bounding box
    velocity: Tuple[float, float]  # (vx, vy) pixels per second
    speed: float  # Magnitude of velocity
    direction: float  # Angle in radians (0 = right, π/2 = up)
    timestamp: float  # Frame timestamp in seconds
    
    def is_moving_towards(
        self, 
        target_centroid: Tuple[float, float],
        angle_threshold: float = math.pi / 4  # 45 degrees
    ) -> bool:
        """
        Determine if this object is moving towards a target location
        
        Args:
            target_centroid: (x, y) position of target
            angle_threshold: Maximum angle deviation (radians) to consider "towards"
            
        Returns:
            True if moving towards target within angle threshold
        """
        if self.speed < 1.0:  # Essentially stationary
            return False
        
        # Vector from current position to target
        dx = target_centroid[0] - self.centroid[0]
        dy = target_centroid[1] - self.centroid[1]
        
        if abs(dx) < 1 and abs(dy) < 1:  # Already at target
            return False
        
        # Angle to target
        angle_to_target = math.atan2(dy, dx)
        
        # Angular difference between velocity direction and target direction
        angle_diff = abs(self.direction - angle_to_target)
        
        # Normalize to [0, π]
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        return angle_diff < angle_threshold


class MotionTracker:
    """
    Tracks motion of objects across frames by computing velocities
    from bounding box positions
    """
    
    def __init__(self, smoothing_window: int = 3):
        """
        Args:
            smoothing_window: Number of frames to average for velocity smoothing
        """
        self.smoothing_window = smoothing_window
        self.history: Dict[str, List[Tuple[float, Tuple[float, float]]]] = {}
        # Maps temp_id -> [(timestamp, (x, y)), ...]
    
    def update(
        self,
        temp_id: str,
        timestamp: float,
        bounding_box: List[int]
    ) -> Optional[MotionData]:
        """
        Update motion tracking for an object and compute current motion data
        
        Args:
            temp_id: Temporary object identifier
            timestamp: Frame timestamp in seconds
            bounding_box: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            MotionData if velocity can be computed, None otherwise
        """
        # Calculate centroid
        x1, y1, x2, y2 = bounding_box
        centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        
        # Initialize history for new objects
        if temp_id not in self.history:
            self.history[temp_id] = []
        
        # Add current observation
        self.history[temp_id].append((timestamp, centroid))
        
        # Keep only recent history
        max_history = self.smoothing_window + 1
        if len(self.history[temp_id]) > max_history:
            self.history[temp_id] = self.history[temp_id][-max_history:]
        
        # Need at least 2 observations to compute velocity
        if len(self.history[temp_id]) < 2:
            return MotionData(
                centroid=centroid,
                velocity=(0.0, 0.0),
                speed=0.0,
                direction=0.0,
                timestamp=timestamp
            )
        
        # Compute velocity using recent history
        return self._compute_velocity(temp_id, timestamp, centroid)
    
    def _compute_velocity(
        self,
        temp_id: str,
        current_timestamp: float,
        current_centroid: Tuple[float, float]
    ) -> MotionData:
        """
        Compute smoothed velocity from recent position history
        
        Args:
            temp_id: Object identifier
            current_timestamp: Current frame timestamp
            current_centroid: Current centroid position
            
        Returns:
            MotionData with computed velocity
        """
        history = self.history[temp_id]
        
        # Use linear regression for smoothed velocity estimation
        # across recent observations
        timestamps = np.array([t for t, _ in history])
        x_coords = np.array([c[0] for _, c in history])
        y_coords = np.array([c[1] for _, c in history])
        
        # Handle edge case: all same timestamp
        if np.std(timestamps) < 1e-6:
            return MotionData(
                centroid=current_centroid,
                velocity=(0.0, 0.0),
                speed=0.0,
                direction=0.0,
                timestamp=current_timestamp
            )
        
        # Fit linear trend: position = velocity * time + offset
        vx = np.polyfit(timestamps, x_coords, 1)[0]  # dx/dt
        vy = np.polyfit(timestamps, y_coords, 1)[0]  # dy/dt
        
        # Compute speed and direction
        speed = math.sqrt(vx**2 + vy**2)
        direction = math.atan2(vy, vx) if speed > 0.1 else 0.0
        
        return MotionData(
            centroid=current_centroid,
            velocity=(vx, vy),
            speed=speed,
            direction=direction,
            timestamp=current_timestamp
        )
    
    def get_motion_at_time(
        self,
        temp_id: str,
        target_timestamp: float,
        tolerance: float = 0.5
    ) -> Optional[MotionData]:
        """
        Retrieve motion data for an object at a specific timestamp
        
        Args:
            temp_id: Object identifier
            target_timestamp: Desired timestamp
            tolerance: Maximum time difference to accept (seconds)
            
        Returns:
            MotionData if available within tolerance, None otherwise
        """
        if temp_id not in self.history or not self.history[temp_id]:
            return None
        
        # Find closest timestamp
        history = self.history[temp_id]
        closest_idx = min(
            range(len(history)),
            key=lambda i: abs(history[i][0] - target_timestamp)
        )
        
        closest_time, closest_centroid = history[closest_idx]
        
        # Check if within tolerance
        if abs(closest_time - target_timestamp) > tolerance:
            return None
        
        # Recompute velocity at this point if we have enough history
        if closest_idx == len(history) - 1:
            # This is the most recent, use stored computation
            return self._compute_velocity(temp_id, closest_time, closest_centroid)
        else:
            # Historical point - use surrounding points
            # Simplified: just use the stored centroid with zero velocity
            # (full implementation would reconstruct velocity from surrounding points)
            return MotionData(
                centroid=closest_centroid,
                velocity=(0.0, 0.0),
                speed=0.0,
                direction=0.0,
                timestamp=closest_time
            )
    
    def clear_old_history(self, cutoff_timestamp: float):
        """
        Remove tracking history older than cutoff timestamp to free memory
        
        Args:
            cutoff_timestamp: Remove observations before this time
        """
        for temp_id in list(self.history.keys()):
            self.history[temp_id] = [
                (t, c) for t, c in self.history[temp_id]
                if t >= cutoff_timestamp
            ]
            # Remove completely if empty
            if not self.history[temp_id]:
                del self.history[temp_id]


def calculate_distance(
    pos1: Tuple[float, float],
    pos2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two positions
    
    Args:
        pos1: (x, y) position 1
        pos2: (x, y) position 2
        
    Returns:
        Distance in pixels
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.sqrt(dx**2 + dy**2)


def bbox_to_centroid(bounding_box: List[int]) -> Tuple[float, float]:
    """
    Convert bounding box to centroid
    
    Args:
        bounding_box: [x1, y1, x2, y2]
        
    Returns:
        (x, y) centroid
    """
    x1, y1, x2, y2 = bounding_box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_overlap_area(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate intersection area between two bounding boxes
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        Overlap area in pixels²
    """
    x1_max = max(bbox1[0], bbox2[0])
    y1_max = max(bbox1[1], bbox2[1])
    x2_min = min(bbox1[2], bbox2[2])
    y2_min = min(bbox1[3], bbox2[3])
    
    if x2_min <= x1_max or y2_min <= y1_max:
        return 0.0
    
    return (x2_min - x1_max) * (y2_min - y1_max)


def bbox_area(bounding_box: List[int]) -> float:
    """
    Calculate area of a bounding box
    
    Args:
        bounding_box: [x1, y1, x2, y2]
        
    Returns:
        Area in pixels²
    """
    x1, y1, x2, y2 = bounding_box
    return (x2 - x1) * (y2 - y1)
