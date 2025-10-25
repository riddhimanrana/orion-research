"""
Unit Tests for Motion Tracker
==============================

Tests for motion tracking, velocity estimation, and direction detection.

Author: Orion Research Team
Date: October 2025
"""

import math
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.tracker import (
    MotionTracker,
    MotionData,
    calculate_distance,
    bbox_to_centroid,
    bbox_overlap_area,
    bbox_area,
)


class TestMotionData:
    """Test MotionData dataclass"""
    
    def test_motion_data_creation(self):
        motion = MotionData(
            centroid=(100.0, 200.0),
            velocity=(5.0, -3.0),
            speed=math.sqrt(5**2 + 3**2),
            direction=math.atan2(-3.0, 5.0),
            timestamp=1.0
        )
        
        assert motion.centroid == (100.0, 200.0)
        assert motion.velocity == (5.0, -3.0)
        assert abs(motion.speed - math.sqrt(34)) < 0.01
        assert motion.timestamp == 1.0
    
    def test_is_moving_towards(self):
        # Object moving right (5, 0)
        motion = MotionData(
            centroid=(100.0, 100.0),
            velocity=(5.0, 0.0),
            speed=5.0,
            direction=0.0,  # Moving right
            timestamp=1.0
        )
        
        # Target to the right
        target_right = (150.0, 100.0)
        assert motion.is_moving_towards(target_right) is True
        
        # Target to the left
        target_left = (50.0, 100.0)
        assert motion.is_moving_towards(target_left) is False
        
        # Target at 30 degrees (should still be True with 45 degree threshold)
        target_diagonal = (150.0, 130.0)
        assert motion.is_moving_towards(target_diagonal) is True
        
        # Target at 90 degrees (should be False)
        target_perpendicular = (100.0, 150.0)
        assert motion.is_moving_towards(target_perpendicular) is False
    
    def test_is_moving_towards_stationary(self):
        # Stationary object
        motion = MotionData(
            centroid=(100.0, 100.0),
            velocity=(0.0, 0.0),
            speed=0.5,  # Below threshold
            direction=0.0,
            timestamp=1.0
        )
        
        target = (150.0, 100.0)
        assert motion.is_moving_towards(target) is False


class TestMotionTracker:
    """Test MotionTracker class"""
    
    def test_initialization(self):
        tracker = MotionTracker(smoothing_window=3)
        assert tracker.smoothing_window == 3
        assert len(tracker.history) == 0
    
    def test_single_update(self):
        tracker = MotionTracker()
        
        bbox = [100, 100, 150, 150]
        motion = tracker.update("obj1", 0.0, bbox)
        
        assert motion is not None
        assert motion.centroid == (125.0, 125.0)
        assert motion.speed == 0.0  # First observation, no velocity
    
    def test_velocity_estimation(self):
        tracker = MotionTracker()
        
        # Move object from (100, 100) to (110, 105) over 1 second
        # Expected velocity: (10, 5) pixels/second
        
        motion1 = tracker.update("obj1", 0.0, [95, 95, 105, 105])  # centroid (100, 100)
        motion2 = tracker.update("obj1", 1.0, [105, 100, 115, 110])  # centroid (110, 105)
        
        assert motion2 is not None
        # Velocity should be approximately (10, 5)
        vx, vy = motion2.velocity
        assert abs(vx - 10.0) < 1.0
        assert abs(vy - 5.0) < 1.0
        assert motion2.speed > 0
    
    def test_smoothing_with_multiple_observations(self):
        tracker = MotionTracker(smoothing_window=3)
        
        # Create a consistent rightward motion
        bboxes = [
            [100, 100, 110, 110],  # t=0, centroid (105, 105)
            [110, 100, 120, 110],  # t=0.5, centroid (115, 105)
            [120, 100, 130, 110],  # t=1.0, centroid (125, 105)
            [130, 100, 140, 110],  # t=1.5, centroid (135, 105)
        ]
        
        timestamps = [0.0, 0.5, 1.0, 1.5]
        
        for i, (ts, bbox) in enumerate(zip(timestamps, bboxes)):
            motion = tracker.update("obj1", ts, bbox)
            if i >= 1:  # After first observation
                # Should detect rightward motion
                assert motion.velocity[0] > 0  # Positive x velocity
                assert abs(motion.velocity[1]) < 2.0  # Near-zero y velocity
    
    def test_multiple_objects(self):
        tracker = MotionTracker()
        
        # Track two different objects
        motion1 = tracker.update("obj1", 0.0, [100, 100, 110, 110])
        motion2 = tracker.update("obj2", 0.0, [200, 200, 210, 210])
        
        assert motion1.centroid == (105.0, 105.0)
        assert motion2.centroid == (205.0, 205.0)
        
        # Update first object
        motion1_v2 = tracker.update("obj1", 1.0, [110, 100, 120, 110])
        
        # Both should be tracked independently
        assert "obj1" in tracker.history
        assert "obj2" in tracker.history
        assert len(tracker.history["obj1"]) == 2
        assert len(tracker.history["obj2"]) == 1
    
    def test_get_motion_at_time(self):
        tracker = MotionTracker()
        
        tracker.update("obj1", 0.0, [100, 100, 110, 110])
        tracker.update("obj1", 1.0, [110, 100, 120, 110])
        tracker.update("obj1", 2.0, [120, 100, 130, 110])
        
        # Get motion at t=1.0
        motion = tracker.get_motion_at_time("obj1", 1.0, tolerance=0.1)
        assert motion is not None
        assert motion.centroid == (115.0, 105.0)
        
        # Get motion at non-existent time (out of tolerance)
        motion_far = tracker.get_motion_at_time("obj1", 10.0, tolerance=0.1)
        assert motion_far is None
    
    def test_clear_old_history(self):
        tracker = MotionTracker()
        
        # Add observations at different times
        tracker.update("obj1", 0.0, [100, 100, 110, 110])
        tracker.update("obj1", 1.0, [110, 100, 120, 110])
        tracker.update("obj1", 5.0, [120, 100, 130, 110])
        tracker.update("obj2", 0.5, [200, 200, 210, 210])
        
        # Clear history before t=2.0
        tracker.clear_old_history(cutoff_timestamp=2.0)
        
        # obj1 should have only one observation (t=5.0)
        assert len(tracker.history["obj1"]) == 1
        # obj2 should be removed completely
        assert "obj2" not in tracker.history


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_calculate_distance(self):
        pos1 = (0.0, 0.0)
        pos2 = (3.0, 4.0)
        
        distance = calculate_distance(pos1, pos2)
        assert abs(distance - 5.0) < 0.01  # 3-4-5 triangle
    
    def test_bbox_to_centroid(self):
        bbox = [100, 200, 150, 250]
        centroid = bbox_to_centroid(bbox)
        
        assert centroid == (125.0, 225.0)
    
    def test_bbox_overlap_area(self):
        # Fully overlapping boxes
        bbox1 = [100, 100, 150, 150]
        bbox2 = [100, 100, 150, 150]
        overlap = bbox_overlap_area(bbox1, bbox2)
        assert overlap == 2500.0  # 50x50
        
        # Partially overlapping boxes
        bbox1 = [100, 100, 150, 150]
        bbox2 = [125, 125, 175, 175]
        overlap = bbox_overlap_area(bbox1, bbox2)
        assert overlap == 625.0  # 25x25
        
        # Non-overlapping boxes
        bbox1 = [100, 100, 150, 150]
        bbox2 = [200, 200, 250, 250]
        overlap = bbox_overlap_area(bbox1, bbox2)
        assert overlap == 0.0
    
    def test_bbox_area(self):
        bbox = [100, 100, 150, 200]
        area = bbox_area(bbox)
        assert area == 5000.0  # 50 x 100


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_time_interval(self):
        tracker = MotionTracker()
        
        # Update same object twice with same timestamp
        tracker.update("obj1", 0.0, [100, 100, 110, 110])
        motion = tracker.update("obj1", 0.0, [110, 100, 120, 110])
        
        # Should handle gracefully (no division by zero)
        assert motion is not None
        assert motion.speed == 0.0
    
    def test_empty_bbox(self):
        tracker = MotionTracker()
        
        # Bbox with zero area
        motion = tracker.update("obj1", 0.0, [100, 100, 100, 100])
        assert motion is not None
        assert motion.centroid == (100.0, 100.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
